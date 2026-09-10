#!/usr/bin/env python3
"""Generate Student/Teacher continuations from identical assistant-prefix states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    return p.parse_args()


def score(text: str, gt: str) -> bool:
    from agopd.reward.math_reward import score_math_response

    return bool(score_math_response(text, gt).correct)


def state_ids(tokenizer: Any, prompt: str, response_ids: list[int]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return prompt_ids + response_ids


def main() -> None:
    a = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    requests: list[dict[str, Any]] = []
    unique_requests: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for row in rows:
        response_ids = tokenizer(row["student_response"], add_special_tokens=False)["input_ids"]
        onset = int(row["error_onset_token"])
        offsets = {
            "root": 0,
            "error_onset": onset,
            "post_error_64": min(len(response_ids), onset + 64),
        }
        for state, offset in offsets.items():
            prompt_token_ids = state_ids(tokenizer, row["prompt"], response_ids[:offset])
            key = (row["prompt_id"], state, tuple(prompt_token_ids))
            request = unique_requests.get(key)
            if request is None:
                request = {
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_id": row["prompt_id"],
                    "state": state,
                    "offset": offset,
                    "gt": row["gt"],
                    "aliases": [],
                }
                unique_requests[key] = request
                requests.append(request)
            request["aliases"].append(row.get("trajectory_kind", "student_wrong"))

    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_tokens + 2048,
        enforce_eager=False,
        max_num_batched_tokens=32768,
        max_num_seqs=32,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": request["prompt_token_ids"]} for request in requests],
        SamplingParams(n=a.n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=a.seed),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for request, output in zip(requests, outputs):
            samples = [{"text": item.text, "correct": score(item.text, request["gt"])} for item in output.outputs]
            for trajectory_kind in request["aliases"]:
                record = {
                    "prompt_id": request["prompt_id"],
                    "trajectory_kind": trajectory_kind,
                    "state": request["state"],
                    "response_token_offset": request["offset"],
                    "model": a.model,
                    "state_reused_for_alias": len(request["aliases"]) > 1,
                    "sampling": {"n": a.n, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": a.seed, "max_tokens": a.max_tokens},
                    "samples": samples,
                    "correct_count": sum(int(item["correct"]) for item in samples),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"continuations_saved={a.output} unique_states={len(requests)} aliases={sum(len(r['aliases']) for r in requests)} samples={sum(len(r['aliases']) for r in requests) * a.n}", flush=True)


if __name__ == "__main__":
    main()
