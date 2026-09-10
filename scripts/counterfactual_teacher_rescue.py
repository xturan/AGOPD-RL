#!/usr/bin/env python3
"""Counterfactual Root-vs-Path Teacher Rescue (CRPTR).

Stage 1 collects student trajectories. Stage 2 asks the same teacher to solve
from the root prompt or continue from fixed fractions of a wrong student path.
The resulting rescue curve tests whether teacher guidance is damaged by the
student's visited state rather than by teacher capacity alone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("student", "teacher"), required=True)
    p.add_argument("--data", default="data/dapo-verl-v1/train_comfort.parquet")
    p.add_argument("--model", required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-prompts", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--correct-control", action="store_true",
                   help="Run the paired correct-prefix control instead of the full prefix sweep.")
    p.add_argument("--random-control", action="store_true",
                   help="Add a same-token-length neutral filler control.")
    return p.parse_args()


def user_content(row: pd.Series) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    raise ValueError(f"unsupported prompt type: {type(prompt)!r}")


def records_from_data(path: str, limit: int, seed: int) -> list[dict]:
    df = pd.read_parquet(path)
    if len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    result = []
    for _, row in df.iterrows():
        reward = row.get("reward_model") or {}
        result.append({
            "idx": str((row.get("extra_info") or {}).get("index", "")),
            "content": user_content(row),
            "gt": str(reward.get("ground_truth", "")),
            "bucket": "comfort",
        })
    return result


def make_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score(text: str, gt: str) -> bool:
    from agopd.reward.math_reward import score_math_response

    return bool(score_math_response(text, gt).correct)


def collect_student(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = records_from_data(args.data, args.num_prompts, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [make_prompt(tokenizer, r["content"]) for r in records]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        enforce_eager=False,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row, request in zip(records, outputs):
            samples = []
            for out in request.outputs:
                samples.append({"text": out.text, "correct": score(out.text, row["gt"])})
            wrong = next((s["text"] for s in samples if not s["correct"]), None)
            row["student_samples"] = samples
            row["wrong_path"] = wrong
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"student_saved={args.output} prompts={len(records)} wrong_paths={sum(r['wrong_path'] is not None for r in records)}", flush=True)


def strip_no_think(content: str) -> str:
    return content.replace("\n\n/no_think", "").replace("\n/no_think", "")


def condition_content(content: str, prefix: str | None, mode: str) -> str:
    base = strip_no_think(content)
    if mode == "root":
        return base + "\n\n/no_think"
    if mode == "ignore":
        instruction = "The draft below may be wrong. Ignore it completely and solve the original problem independently."
    else:
        instruction = "The draft below may be wrong. Continue the solution, correct any mistakes you detect, and give the final answer."
    return base + "\n\n" + instruction + "\n\n<student_draft>\n" + (prefix or "") + "\n</student_draft>\n\n/no_think"


def prefix_at(tokenizer, text: str, fraction: float) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    end = max(1, int(len(ids) * fraction))
    return tokenizer.decode(ids[:end], skip_special_tokens=True)


def prefix_with_token_length(tokenizer, text: str, token_length: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:token_length], skip_special_tokens=True)


def filler_prefix(tokenizer, token_length: int) -> str:
    filler = "neutral draft placeholder text " * max(16, token_length // 4)
    return prefix_with_token_length(tokenizer, filler, token_length)


def teacher_conditions(
    tokenizer, row: dict, correct_control: bool = False, random_control: bool = False
) -> list[dict]:
    wrong = row.get("wrong_path")
    if not wrong:
        return []
    if correct_control:
        correct = next((s["text"] for s in row.get("student_samples", []) if s.get("correct")), None)
        if not correct:
            return []
        wrong_ids = tokenizer(wrong, add_special_tokens=False)["input_ids"]
        correct_ids = tokenizer(correct, add_special_tokens=False)["input_ids"]
        matched_tokens = max(1, min(len(wrong_ids), len(correct_ids)) // 2)
        wrong_prefix = prefix_with_token_length(tokenizer, wrong, matched_tokens)
        correct_prefix = prefix_with_token_length(tokenizer, correct, matched_tokens)
        conditions = [
            {"condition": "root", "prefix_fraction": 0.0, "content": condition_content(row["content"], None, "root")},
            {"condition": "wrong_path_matched", "prefix_fraction": matched_tokens / len(wrong_ids),
             "prefix_tokens": matched_tokens, "prefix": wrong_prefix,
             "content": condition_content(row["content"], wrong_prefix, "path")},
            {"condition": "correct_path_matched", "prefix_fraction": matched_tokens / len(correct_ids),
             "prefix_tokens": matched_tokens, "prefix": correct_prefix,
             "content": condition_content(row["content"], correct_prefix, "path")},
            {"condition": "ignore_path_matched", "prefix_fraction": matched_tokens / len(wrong_ids),
             "prefix_tokens": matched_tokens, "prefix": wrong_prefix,
             "content": condition_content(row["content"], wrong_prefix, "ignore")},
        ]
        if random_control:
            filler = filler_prefix(tokenizer, matched_tokens)
            conditions.append({
                "condition": "random_filler_matched",
                "prefix_fraction": matched_tokens / len(wrong_ids),
                "prefix_tokens": matched_tokens,
                "prefix": filler,
                "content": condition_content(row["content"], filler, "ignore"),
            })
        return conditions
    result = [{"condition": "root", "prefix_fraction": 0.0, "content": condition_content(row["content"], None, "root")}]
    for fraction in (0.25, 0.5, 0.75):
        prefix = prefix_at(tokenizer, wrong, fraction)
        result.append({
            "condition": f"path_{int(fraction * 100)}",
            "prefix_fraction": fraction,
            "prefix": prefix,
            "content": condition_content(row["content"], prefix, "path"),
        })
    prefix = prefix_at(tokenizer, wrong, 0.5)
    result.append({
        "condition": "ignore_path_50",
        "prefix_fraction": 0.5,
        "prefix": prefix,
        "content": condition_content(row["content"], prefix, "ignore"),
    })
    return result


def collect_teacher(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.input is None:
        raise ValueError("--input is required for teacher stage")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    def eligible(row: dict) -> bool:
        has_wrong = bool(row.get("wrong_path"))
        has_correct = any(s.get("correct") for s in row.get("student_samples", []))
        return has_wrong and (has_correct if args.correct_control else True)

    rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard_idx and eligible(r)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    jobs = []
    for row_idx, row in enumerate(rows):
        for condition in teacher_conditions(tokenizer, row, args.correct_control, args.random_control):
            condition["row_idx"] = row_idx
            jobs.append(condition)
    prompts = [make_prompt(tokenizer, j["content"]) for j in jobs]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        enforce_eager=False,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for job, request in zip(jobs, outputs):
            row = rows[job["row_idx"]]
            samples = [{"text": out.text, "correct": score(out.text, row["gt"])} for out in request.outputs]
            result = {
                "idx": row["idx"],
                "gt": row["gt"],
                "bucket": row.get("bucket", "comfort"),
                "condition": job["condition"],
                "prefix_fraction": job["prefix_fraction"],
                "student_wrong_path": row["wrong_path"],
                "teacher_samples": samples,
                "teacher_correct_rate": sum(s["correct"] for s in samples) / max(1, len(samples)),
            }
            if "prefix" in job:
                result["student_prefix"] = job["prefix"]
            if "prefix_tokens" in job:
                result["prefix_tokens"] = job["prefix_tokens"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"teacher_saved={args.output} rows={len(rows)} jobs={len(jobs)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "student":
        collect_student(args)
    else:
        collect_teacher(args)


if __name__ == "__main__":
    main()
