"""Teacher backfill for shards that skipped Stage C (e.g. local 4090 shards).

Reads a shard's buckets.jsonl, finds hard prompts with zero student rejection
hits, and generates teacher (8B) completions for them, verifier-filtered.
Outputs sft rows compatible with build_coldstart_data.py.

Run on an A100 (8B teacher is unstable on 24 GiB 4090s):
  CUDA_VISIBLE_DEVICES=0 python3 scripts/teacher_backfill.py \
      --buckets outputs/coldstart-5000-shard4/buckets.jsonl \
      --output outputs/coldstart-5000-shard4/teacher_backfill.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from agopd.reward.math_reward import score_math_response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-correct-only backfill for skipped shards.")
    parser.add_argument("--buckets", required=True, type=Path, help="buckets.jsonl of the shard to backfill")
    parser.add_argument("--data", default="data/dapo-verl-v1/train.parquet")
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--teacher-n", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=40, help="Prompts per batch (x n completions)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # prompts needing backfill: hard bucket, zero rejection hits
    need = []
    for line in open(args.buckets):
        b = json.loads(line)
        if b["bucket"] == "hard" and b.get("rejection_correct", 0) == 0:
            need.append(b["index"])
    print(f"[backfill] {len(need)} prompts need teacher")

    df = pd.read_parquet(args.data)
    df["_idx"] = df["extra_info"].apply(lambda x: str(x.get("index", "")) if isinstance(x, dict) else "")
    by_idx = {str(r["_idx"]): r for _, r in df.iterrows()}
    prompts = []
    for idx in need:
        r = by_idx.get(idx)
        if r is None:
            continue
        prompts.append(r)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(4096, args.max_new_tokens + 1024),
        enforce_eager=True,
        max_num_batched_tokens=8192,
        max_num_seqs=512,
    )

    rows_out = []
    for i in range(0, len(prompts), args.chunk):
        chunk = prompts[i : i + args.chunk]
        rendered = []
        for r in chunk:
            msgs = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in r["prompt"]
            ]
            rendered.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        out = llm.generate(
            rendered,
            SamplingParams(n=args.teacher_n, temperature=0.6, top_p=0.95, top_k=20,
                           max_tokens=args.max_new_tokens, seed=args.seed),
        )
        for r, result in zip(chunk, out):
            gt = str(r["reward_model"].get("ground_truth", "")) if isinstance(r["reward_model"], dict) else ""
            for o in result.outputs:
                res = score_math_response(o.text, gt)
                if res.correct:
                    rows_out.append(
                        {
                            "index": str(r["extra_info"].get("index", "")),
                            "prompt_messages": [
                                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                                for m in r["prompt"]
                            ],
                            "ground_truth": gt,
                            "source": "teacher_correct_only",
                            "output": o.text,
                        }
                    )
        print(f"  chunk {i // args.chunk + 1}/{(len(prompts) + args.chunk - 1) // args.chunk} done, hits so far {len(rows_out)}")

    with args.output.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    uniq = len({r["index"] for r in rows_out})
    print(f"[backfill] saved {len(rows_out)} rows / {uniq} unique prompts to {args.output}")


if __name__ == "__main__":
    main()
