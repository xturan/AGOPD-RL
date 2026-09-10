#!/usr/bin/env python3
"""Semantic, row-level audit for math trajectories.

The numeric reward answers only whether the extracted final answer matches the
gold answer. This evaluator reads the problem, gold answer, and full response
and records whether the reasoning actually supports the conclusion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RUBRIC = """You are a strict math-trajectory quality auditor. Read the problem,
gold answer, and the model response. Judge the response itself, not just the
provided correct flag. A correct final answer with unsupported, contradictory,
random, copied, or circular reasoning is not a clean trajectory.

Return exactly one JSON object and no markdown. Use these values:
reasoning_quality: sound | partial | unsound | absent | undecidable
trajectory_class: clean_correct | correct_but_unsound | correct_lucky_or_no_reasoning |
  wrong_with_progress | wrong_no_progress | malformed_or_repetitive | truncated_or_incomplete
defects: an array containing only: template_echo, answer_repetition, answer_first,
  missing_answer, wrong_answer, unsupported_leap, contradiction, arithmetic_error,
  irrelevant_reasoning, repetition_loop, truncation, hedging_or_guessing,
  meta_or_courtesy_tail, parse_or_format_error
confidence: a number from 0 to 1
rationale: one concise sentence naming the decisive evidence from the response

Rules:
1. Check the math steps against the problem and gold answer. Do not reward a
   lucky final guess as sound reasoning.
2. A response may be sound without saying the literal word Answer, but missing
   or ambiguous final output is a format defect.
3. Mark truncation only when the response visibly stops mid-sentence/derivation
   or ends at a generation limit; do not infer it from length alone.
4. Mark repetition_loop for pathological repeated phrases, not ordinary use of
   the word Answer several times.
5. Keep the rationale grounded in the actual text, with no invented steps.
"""


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--data", default="data/dapo-verl-v1/val.parquet")
    p.add_argument("--model", default="models/Qwen3-8B")
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-input-tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=220)
    return p.parse_args()


def load_problem_map(path: str) -> dict[int, tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=path, split="train")
    result = {}
    for i, row in enumerate(ds):
        prompt = row["prompt"]
        if isinstance(prompt, list):
            problem = next((x.get("content", "") for x in prompt if x.get("role") == "user"), "")
        else:
            problem = str(prompt)
        reward = row.get("reward_model") or {}
        gt = str(reward.get("ground_truth", ""))
        result[i] = (problem, gt)
    return result


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "reasoning_quality": "undecidable",
        "trajectory_class": "malformed_or_repetitive",
        "defects": ["parse_or_format_error"],
        "confidence": 0.0,
        "rationale": "The semantic evaluator did not return valid JSON.",
    }


def normalize_result(value: dict, record: dict) -> dict:
    allowed_quality = {"sound", "partial", "unsound", "absent", "undecidable"}
    allowed_class = {
        "clean_correct", "correct_but_unsound", "correct_lucky_or_no_reasoning",
        "wrong_with_progress", "wrong_no_progress", "malformed_or_repetitive",
        "truncated_or_incomplete",
    }
    allowed_defects = {
        "template_echo", "answer_repetition", "answer_first", "missing_answer",
        "wrong_answer", "unsupported_leap", "contradiction", "arithmetic_error",
        "irrelevant_reasoning", "repetition_loop", "truncation", "hedging_or_guessing",
        "meta_or_courtesy_tail", "parse_or_format_error",
    }
    quality = value.get("reasoning_quality")
    cls = value.get("trajectory_class")
    defects = value.get("defects")
    if quality not in allowed_quality:
        quality = "undecidable"
    if cls not in allowed_class:
        cls = "malformed_or_repetitive"
    if not isinstance(defects, list):
        defects = []
    defects = [str(x) for x in defects if str(x) in allowed_defects]
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(value.get("rationale", "")).strip()
    if not rationale:
        rationale = "No rationale returned."
    return {
        "idx": record["idx"],
        "correct": bool(record.get("correct")),
        "gt": record.get("gt"),
        "reasoning_quality": quality,
        "trajectory_class": cls,
        "defects": defects,
        "confidence": round(confidence, 3),
        "rationale": rationale,
    }


def main() -> None:
    a = args()
    records = [json.loads(line) for line in Path(a.input).read_text().splitlines() if line.strip()]
    records = records[a.start : (a.start + a.limit if a.limit else None)]
    problem_map = load_problem_map(a.data)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w") as f:
        for n, record in enumerate(records, 1):
            problem, mapped_gt = problem_map[int(record["idx"])]
            prompt = (
                f"Problem:\n{problem}\n\nGold answer: {record.get('gt', mapped_gt)}\n"
                f"Numeric checker correct flag: {record.get('correct')}\n\n"
                f"Model response (read the entire response):\n{record['text']}\n\n"
                "Now apply the rubric and return the JSON object."
            )
            messages = [
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=a.max_input_tokens,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=a.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            result = normalize_result(parse_json(raw), record)
            result["raw_evaluator_output"] = raw
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
            print(f"audited={n}/{len(records)} idx={record['idx']} class={result['trajectory_class']}", flush=True)

    summary = {
        "input": a.input,
        "count": len(results),
        "correct_count": sum(x["correct"] for x in results),
        "reasoning_quality": dict(Counter(x["reasoning_quality"] for x in results)),
        "trajectory_class": dict(Counter(x["trajectory_class"] for x in results)),
        "defects": dict(Counter(d for x in results for d in x["defects"])),
        "correct_by_quality": dict(Counter(x["reasoning_quality"] for x in results if x["correct"])),
        "wrong_by_quality": dict(Counter(x["reasoning_quality"] for x in results if not x["correct"])),
    }
    Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
