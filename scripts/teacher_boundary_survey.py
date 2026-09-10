"""Teacher capability boundary survey.

Sample N prompts with a given model at n=4 and score with the verifier.
Two passes (4B locally, 8B on A100s) merge into the four-quadrant matrix:
A both correct / B teacher-only / C student-only / D neither.

Output: per-prompt correct counts jsonl for later quadrant analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from agopd.reward.math_reward import score_math_response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher boundary survey pass.")
    parser.add_argument("--data", required=True, help="val.parquet or train.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-prompts", type=int, default=None, help="Random subset (train only)")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--batch-prompts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(args.data)
    if args.sample_prompts is not None and len(df) > args.sample_prompts:
        df = df.sample(n=args.sample_prompts, random_state=args.seed).reset_index(drop=True)
    print(f"[survey] {len(df)} prompts, model {args.model}, n={args.n}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, metas = [], []
    for _, r in df.iterrows():
        msgs = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in r["prompt"]
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        idx = str(r["extra_info"].get("index", "")) if isinstance(r.get("extra_info"), dict) else ""
        gt = str(r["reward_model"].get("ground_truth", "")) if isinstance(r["reward_model"], dict) else ""
        metas.append({"index": idx, "ground_truth": gt})

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=args.eager,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    )

    rows_out = []
    bs = args.batch_prompts
    for start in range(0, len(prompts), bs):
        chunk_p = prompts[start : start + bs]
        out = llm.generate(
            chunk_p,
            SamplingParams(n=args.n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, seed=args.seed),
        )
        for r, meta in zip(out, metas[start : start + bs]):
            hits = sum(1 for o in r.outputs if score_math_response(o.text, meta["ground_truth"]).correct)
            rows_out.append({"index": meta["index"], "correct": hits, "n": args.n})
        print(f"  batch {start // bs + 1}/{(len(prompts) + bs - 1) // bs} done", flush=True)

    with args.output.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    n_hit = sum(1 for r in rows_out if r["correct"] > 0)
    print(f"[survey] saved {len(rows_out)} prompts to {args.output}; hit rate (n>=1): {n_hit}/{len(rows_out)} = {100*n_hit/len(rows_out):.1f}%")


if __name__ == "__main__":
    main()
