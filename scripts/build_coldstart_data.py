"""Hard-Prompt Targeted Cold Start data builder.

Frontier-Shaping pipeline (AGOPD project):
  1. Sample N prompts from the DAPO train split.
  2. Estimate per-prompt success p_hat with the student model at n=8.
  3. Bucket prompts: Easy (p_hat>=0.75) / Frontier (0.125..0.75) / Hard (0).
  4. Hard prompts: rejection sampling with the student at n=32 to find
     verifier-confirmed correct trajectories (method 1, on-policy-ish).
  5. Optionally: teacher (8B) generation for prompts with zero student hits
     (method 2, teacher-correct-only).
  6. Assemble verifier-confirmed trajectories into an SFT corpus with the
     DAPO exact prompt, full reasoning and the unified Answer: contract.

Run on a local GPU (e.g. two RTX 4090):
  CUDA_VISIBLE_DEVICES=0 python3 scripts/build_coldstart_data.py \
      --sample-prompts 200 --output outputs/coldstart-pilot
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from agopd.reward.math_reward import score_math_response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Hard-Prompt targeted cold-start SFT data.")
    parser.add_argument("--data", default="data/dapo-verl-v1/train.parquet")
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--teacher-model", type=str, default=None, help="8B teacher for method-2 generation")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip Stage C entirely (use on 24 GiB 4090s where the 8B teacher engine is unstable)")
    parser.add_argument("--sample-prompts", type=int, default=200, help="Number of prompts to bucket")
    parser.add_argument("--est-n", type=int, default=8, help="Student samples per prompt for p_hat estimation")
    parser.add_argument("--rejection-n", type=int, default=32, help="Total student samples for Hard prompts")
    parser.add_argument("--teacher-n", type=int, default=8, help="Teacher samples for still-zero prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/coldstart-pilot"))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=50, help="Prompts per vLLM batch (x n sequences)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--eager", action="store_true", help="Enable eager mode (slow); default is CUDA Graph")
    parser.add_argument("--teacher-eager", action="store_true", help="Teacher engine in eager mode (needed on 24 GiB 4090s)")
    parser.add_argument("--teacher-gmu", type=float, default=None, help="Teacher gpu_memory_utilization override")
    parser.add_argument("--teacher-batched-tokens", type=int, default=8192,
                        help="Teacher max_num_batched_tokens; 32768 blows up 8B warmup on 24 GiB 4090s")
    parser.add_argument("--max-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument(
        "--shard-idx", type=int, default=0, help="Shard index when splitting the sampled prompts across GPUs"
    )
    parser.add_argument(
        "--n-shards", type=int, default=1, help="Total shards; prompts are split contiguously after a global seed sample"
    )
    return parser.parse_args()


def _messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return [{"role": "user", "content": str(value)}]
    return [
        {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
        for message in value
        if isinstance(message, dict)
    ]


def _ground_truth(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("ground_truth", ""))
    if hasattr(value, "as_py"):
        py = value.as_py()
        if isinstance(py, dict):
            return str(py.get("ground_truth", ""))
    return str(value)


def _render(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate_batch(llm, prompts: list[str], n: int, max_tokens: int, temperature: float, top_p: float, top_k: int, seed: int):
    from vllm import SamplingParams

    params = SamplingParams(
        n=n, temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, seed=seed,
    )
    return llm.generate(prompts, params)


def _classify(p_hat: float) -> str:
    if p_hat >= 0.75:
        return "easy"
    if p_hat >= 0.125:
        return "frontier"
    return "hard"


def main() -> None:
    args = _parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    df = pd.read_parquet(args.data)
    if args.sample_prompts is not None and len(df) > args.sample_prompts:
        df = df.sample(n=args.sample_prompts, random_state=args.seed).reset_index(drop=True)
    if args.n_shards > 1:
        # Contiguous split of the globally-seeded sample; shard runs are independent.
        total = len(df)
        per = total // args.n_shards
        start = args.shard_idx * per
        end = start + per if args.shard_idx < args.n_shards - 1 else total
        df = df.iloc[start:end].reset_index(drop=True)
        print(f"[shard] {args.shard_idx}/{args.n_shards}: prompts {start}-{end} ({len(df)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [
        {
            "index": (
                str(r["extra_info"].get("index", ""))
                if isinstance(r.get("extra_info"), dict)
                else ""
            ),
            "prompt_messages": _messages(r["prompt"]),
            "prompt": _render(tokenizer, _messages(r["prompt"])),
            "ground_truth": _ground_truth(r["reward_model"]),
            "data_source": str(r.get("data_source", "")),
        }
        for _, r in df.iterrows()
    ]

    from vllm import LLM

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(4096, args.max_new_tokens + 1024),
        enforce_eager=args.eager,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    def score_run(prompt_rec: dict, completions: list[str]) -> list[dict]:
        scored = []
        for text in completions:
            res = score_math_response(text, prompt_rec["ground_truth"])
            scored.append(
                {
                    "output": text,
                    "correct": res.correct,
                    "predicted": res.predicted_answer,
                    "reason": res.reason,
                }
            )
        return scored

    # ---------- Stage A: p_hat estimation at n=est_n ----------
    print(f"[stage A] estimating p_hat with n={args.est_n} on {len(records)} prompts")
    for i in range(0, len(records), args.chunk):
        chunk = records[i : i + args.chunk]
        outputs = _generate_batch(
            llm, [r["prompt"] for r in chunk], args.est_n,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.seed,
        )
        for rec, out in zip(chunk, outputs):
            texts = [o.text for o in out.outputs]
            rec["est_completions"] = score_run(rec, texts)
            rec["p_hat"] = sum(1 for c in rec["est_completions"] if c["correct"]) / len(texts)
            rec["bucket"] = _classify(rec["p_hat"])
        print(f"  chunk {i // args.chunk + 1}/{(len(records) + args.chunk - 1) // args.chunk} done")

    easy = [r for r in records if r["bucket"] == "easy"]
    frontier = [r for r in records if r["bucket"] == "frontier"]
    hard = [r for r in records if r["bucket"] == "hard"]
    print(
        f"[stage A] easy={len(easy)} frontier={len(frontier)} hard={len(hard)} "
        f"(frontier_share={len(frontier) / len(records):.2f})"
    )

    # ---------- Stage B: rejection sampling on Hard prompts ----------
    # NOTE: batch ALL hard prompts in one vLLM call per chunk (n per prompt),
    # instead of one call per prompt — vLLM schedules the whole batch in
    # parallel and this is ~30-50x faster than per-prompt serial calls.
    extra_n = max(0, args.rejection_n - args.est_n)
    sft_rows: list[dict] = []
    if hard and extra_n > 0:
        for i in range(0, len(hard), args.chunk):
            chunk = hard[i : i + args.chunk]
            outputs = _generate_batch(
                llm, [r["prompt"] for r in chunk], extra_n,
                args.max_new_tokens, args.temperature, args.top_p, args.top_k,
                args.seed + 1 + args.shard_idx,
            )
            for rec, out in zip(chunk, outputs):
                extra = score_run(rec, [o.text for o in out.outputs])
                rec["est_completions"].extend(extra)
            print(f"  rejection chunk {i // args.chunk + 1}/{(len(hard) + args.chunk - 1) // args.chunk} done")
    for rec in hard:
        correct = [c for c in rec["est_completions"] if c["correct"]]
        rec["rejection_correct"] = len(correct)
        for c in correct:
            sft_rows.append(
                {
                    "index": rec["index"],
                    "prompt_messages": rec["prompt_messages"],
                    "ground_truth": rec["ground_truth"],
                    "source": "student_rejection",
                    "output": c["output"],
                }
            )

    # ---------- Stage C: teacher generation for still-zero prompts ----------
    teacher_sft = []
    if args.teacher_model is not None and not args.skip_teacher:
        still_zero = [r for r in hard if r["rejection_correct"] == 0]
        if still_zero:
            print(f"[stage C] teacher {args.teacher_model} on {len(still_zero)} still-zero prompts")
            # Free the student engine first: two LLM engines cannot coexist on a
            # single 24 GiB RTX 4090 (student + teacher weights would OOM).
            import gc

            del llm
            gc.collect()
            torch.cuda.empty_cache()
            teacher_llm = LLM(
                model=args.teacher_model,
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=args.teacher_gmu if args.teacher_gmu is not None else args.gpu_memory_utilization,
                max_model_len=max(4096, args.max_new_tokens + 1024),
                enforce_eager=args.teacher_eager or args.eager,
                max_num_batched_tokens=args.teacher_batched_tokens,
                max_num_seqs=args.max_num_seqs,
            )
            teacher_chunk = max(2, min(args.chunk, 20))  # A100-friendly batch; 4090s use --chunk 10
            for i in range(0, len(still_zero), teacher_chunk):
                chunk = still_zero[i : i + teacher_chunk]
                outputs = _generate_batch(
                    teacher_llm, [r["prompt"] for r in chunk], args.teacher_n,
                    args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.seed + 2 + args.shard_idx,
                )
                for rec, out in zip(chunk, outputs):
                    correct = [
                        c for c in score_run(rec, [o.text for o in out.outputs]) if c["correct"]
                    ]
                    rec["teacher_correct"] = len(correct)
                    for c in correct:
                        teacher_sft.append(
                            {
                                "index": rec["index"],
                                "prompt_messages": rec["prompt_messages"],
                                "ground_truth": rec["ground_truth"],
                                "source": "teacher_correct_only",
                                "output": c["output"],
                            }
                        )
            del teacher_llm

    # Cap at 2 trajectories per prompt for diversity (avoid a single prompt
    # monopolizing the corpus, e.g. a teacher that scores 7/8 on one problem).
    per_prompt: dict[str, int] = {}
    capped: list[dict] = []
    for row in sft_rows + teacher_sft:
        if per_prompt.get(row["index"], 0) < 2:
            capped.append(row)
            per_prompt[row["index"]] = per_prompt.get(row["index"], 0) + 1
    all_sft = capped

    # ---------- Report ----------
    est_all_zero = sum(1 for r in records if r["bucket"] == "hard")
    report = {
        "prompts_total": len(records),
        "easy": len(easy),
        "frontier": len(frontier),
        "hard": len(hard),
        "est_all_zero_rate": est_all_zero / len(records),
        "student_rejection_hits": len(sft_rows),
        "teacher_hits": len(teacher_sft),
        "sft_rows_total": len(all_sft),
        "unique_prompts_in_sft": len({r["index"] for r in all_sft}),
        "sampling": {
            "est_n": args.est_n,
            "rejection_n": args.rejection_n,
            "teacher_n": args.teacher_n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
    }
    print(json.dumps(report, indent=2))

    # ---------- Save ----------
    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with (out_dir / "sft_data.jsonl").open("w") as f:
        for row in all_sft:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "buckets.jsonl").open("w") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "index": r["index"],
                        "p_hat": r["p_hat"],
                        "bucket": r["bucket"],
                        "rejection_correct": r.get("rejection_correct", 0),
                        "teacher_correct": r.get("teacher_correct", 0),
                        "ground_truth": r["ground_truth"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
