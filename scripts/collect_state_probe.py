#!/usr/bin/env python3
"""Collect real trajectories and same-state top-k probes for Appendix F.

The experiment is deliberately split into three stages:

1. ``student`` samples real on-policy trajectories from the parquet prompt.
2. ``queue`` creates a semantic first-error annotation queue.
3. ``probe`` evaluates Student and Teacher on exactly the same
   ``user prompt + assistant prefix`` state.

The probe stage does not insert a ``student_draft`` instruction.  That would
measure a new prompt condition rather than the state visited by the student.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("student", "queue", "probe"))
    p.add_argument("--data", default="data/dapo-verl-v1/val.parquet")
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-prompts", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--student-model", default="models/Qwen3-1.7B")
    p.add_argument("--teacher-model", default="models/Qwen3-4B-grpo-50step-ckpt2")
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-device", default="cuda:1")
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--require-matched-correct", action="store_true", help="queue only prompts with both a correct and a wrong Student trajectory")
    p.add_argument("--max-wrong-per-prompt", type=int, default=0, help="queue at most this many wrong samples per prompt; 0 means all")
    return p.parse_args()


def prompt_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        for message in value:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    raise TypeError(f"unsupported prompt value: {type(value)!r}")


def load_rows(path: str, count: int, seed: int) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    rng = random.Random(seed)
    indices = list(range(len(frame)))
    if count < len(indices):
        indices = sorted(rng.sample(indices, count))
    rows: list[dict[str, Any]] = []
    for position in indices:
        row = frame.iloc[position]
        reward = row.get("reward_model") or {}
        extra = row.get("extra_info") or {}
        rows.append(
            {
                "row_index": position,
                "prompt_id": str(extra.get("index", position)),
                "prompt": prompt_content(row["prompt"]),
                "gt": str(reward.get("ground_truth", extra.get("ground_truth", ""))),
            }
        )
    return rows


def score(text: str, gt: str) -> bool:
    from agopd.reward.math_reward import score_math_response

    return bool(score_math_response(text, gt).correct)


def collect_student(a: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    rows = load_rows(a.data, a.num_prompts, a.seed)
    llm = LLM(
        model=a.student_model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_tokens + 1024,
        enforce_eager=False,
        max_num_batched_tokens=32768,
        max_num_seqs=128,
    )
    outputs = llm.generate(
        [r["prompt"] for r in rows],
        SamplingParams(
            n=a.n,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=a.max_tokens,
            seed=a.seed,
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    all_samples: list[dict[str, Any]] = []
    mixed = 0
    with a.output.open("w", encoding="utf-8") as handle:
        for row, request in zip(rows, outputs):
            samples = []
            for sample_id, out in enumerate(request.outputs):
                text = out.text
                samples.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "correct": score(text, row["gt"]),
                    }
                )
            all_samples.extend(samples)
            if any(s["correct"] for s in samples) and any(not s["correct"] for s in samples):
                mixed += 1
            record = {**row, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": a.seed}, "student_samples": samples}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    correct = sum(int(s["correct"]) for s in all_samples)
    total = len(all_samples)
    print(f"student_saved={a.output} prompts={len(rows)} trajectories={total} correct={correct}/{total} mixed_prompts={mixed}", flush=True)


def make_queue(a: argparse.Namespace) -> None:
    if a.input is None:
        raise ValueError("--input is required for queue stage")
    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with a.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            correct = next((s for s in row["student_samples"] if s["correct"]), None)
            if a.require_matched_correct and correct is None:
                continue
            queued_for_prompt = 0
            for sample in row["student_samples"]:
                if sample["correct"]:
                    continue
                if a.max_wrong_per_prompt and queued_for_prompt >= a.max_wrong_per_prompt:
                    break
                handle.write(
                    json.dumps(
                        {
                            "prompt_id": row["prompt_id"],
                            "row_index": row["row_index"],
                            "prompt": row["prompt"],
                            "gt": row["gt"],
                            "student_sample_id": sample["sample_id"],
                            "student_response": sample["text"],
                            "matched_correct_response": correct["text"] if correct else None,
                            "student_correct": False,
                            "error_onset_unit": None,
                            "error_onset_token": None,
                            "error_type": None,
                            "error_evidence": None,
                            "error_persists": None,
                            "annotation_confidence": None,
                            "auditor": None,
                            "annotation_status": "needs_semantic_audit",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
                queued_for_prompt += 1
    print(f"queue_saved={a.output} wrong_trajectories={count}", flush=True)


def render_state(tokenizer: Any, prompt: str, response_ids: list[int]) -> list[int]:
    """Return ids for the exact user + assistant-prefix state."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return prompt_ids + response_ids


def probe_one(model: Any, tokenizer: Any, device: str, ids: list[int], topk: int) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, -1].float()
    log_z = torch.logsumexp(logits, dim=-1)
    log_probs = logits - log_z
    values, indices = torch.topk(log_probs, k=min(topk, log_probs.shape[-1]))
    top = []
    for log_prob, index in zip(values.tolist(), indices.tolist()):
        top.append(
            {
                "token_id": int(index),
                "token": tokenizer.decode([index], skip_special_tokens=False),
                "logprob": float(log_prob),
                "prob": float(math.exp(log_prob)),
            }
        )
    probs = values.softmax(dim=0)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    top_mass = float(values.exp().sum().item())
    return {"top": top, "topk_entropy": entropy, "topk_mass": top_mass, "argmax_token_id": int(indices[0].item())}


def probe(a: argparse.Namespace) -> None:
    if a.input is None:
        raise ValueError("--input is required for probe stage")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("error_onset_token") is not None]
    if not rows:
        raise ValueError("no audited rows with error_onset_token; run queue stage and fill semantic annotations first")

    tokenizer = AutoTokenizer.from_pretrained(a.student_model, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(a.student_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(a.student_device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(a.teacher_device).eval()

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            response = row["student_response"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            onset = int(row["error_onset_token"])
            positions = {"root": 0, "pre_error_64": max(0, onset - 64), "pre_error_16": max(0, onset - 16), "error_onset": onset, "post_error_16": min(len(response_ids), onset + 16), "post_error_64": min(len(response_ids), onset + 64)}
            probes = {}
            for name, offset in positions.items():
                ids = render_state(tokenizer, row["prompt"], response_ids[:offset])
                probes[name] = {
                    "response_token_offset": offset,
                    "student": probe_one(student, tokenizer, a.student_device, ids, a.topk),
                    "teacher": probe_one(teacher, tokenizer, a.teacher_device, ids, a.topk),
                }
            handle.write(json.dumps({**row, "probe_protocol": {"topk": a.topk, "same_state": "chat_template(user) + assistant_prefix", "student_model": a.student_model, "teacher_model": a.teacher_model}, "probes": probes}, ensure_ascii=False) + "\n")
            print(f"probed={row['prompt_id']} sample={row['student_sample_id']} onset={onset}", flush=True)
    print(f"probe_saved={a.output} rows={len(rows)}", flush=True)


def main() -> None:
    a = args()
    if a.stage == "student":
        collect_student(a)
    elif a.stage == "queue":
        make_queue(a)
    else:
        probe(a)


if __name__ == "__main__":
    main()
