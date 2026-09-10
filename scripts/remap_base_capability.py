#!/usr/bin/env python3
"""Evaluate Qwen3-1.7B Base on the fixed 300-prompt capability-map sample."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-1.7B")
    p.add_argument("--data", default="data/dapo-verl-v1/val.parquet")
    p.add_argument("--census", default="reports/figures/val_census_base_all.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def bucket(p: float) -> str:
    if p == 0:
        return "hard(p=0)"
    if p < 0.25:
        return "p<.25"
    if p < 0.5:
        return ".25-.5"
    if p < 0.75:
        return ".5-.75"
    return "easy(>.75)"


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    census = json.loads(Path(args.census).read_text())
    buckets = defaultdict(list)
    for row in census:
        buckets[bucket(row["p_hat"])].append(row["idx"])

    rng = random.Random(args.seed)
    selected = []
    order = ["hard(p=0)", "p<.25", ".25-.5", ".5-.75", "easy(>.75)"]
    for name in order:
        selected.extend((idx, name) for idx in rng.sample(buckets[name], 60))
    wanted = {idx for idx, _ in selected}

    ds = load_dataset("parquet", data_files=args.data, split="train")
    by_idx = {}
    for row in ds:
        idx = str(row.get("extra_info", {}).get("index", ""))
        if idx in wanted:
            messages = row["prompt"]
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            gt = str(row.get("reward_model", {}).get("ground_truth", ""))
            by_idx[idx] = (messages, gt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [tokenizer.apply_chat_template(by_idx[idx][0], tokenize=False, add_generation_prompt=True, enable_thinking=False) for idx, _ in selected]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.82,
        max_model_len=4096,
        enforce_eager=False,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
    )

    from agopd.reward.math_reward import score_math_response

    outputs = llm.generate(
        prompts,
        SamplingParams(n=8, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, seed=args.seed),
    )
    rows = []
    for (idx, name), output in zip(selected, outputs):
        gt = by_idx[idx][1]
        hits = sum(score_math_response(item.text, gt).correct for item in output.outputs)
        rows.append({"idx": idx, "bucket": name, "p_hat8": hits / 8})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    agg = defaultdict(list)
    for row in rows:
        agg[row["bucket"]].append(row["p_hat8"])
    print("base capability map", {name: round(sum(agg[name]) / len(agg[name]), 4) for name in order}, flush=True)
    print(f"saved={args.output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
