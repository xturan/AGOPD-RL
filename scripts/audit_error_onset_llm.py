#!/usr/bin/env python3
"""Propose first semantic error locations for Appendix F trajectories.

This is an annotation assistant, not an automatic correctness oracle.  Every
proposal retains the original response, an exact quote, and a status so that
the proposal can be reviewed before it is used for probing or paper claims.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="models/Qwen3-4B-grpo-50step-ckpt2")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def audit_prompt(row: dict[str, Any]) -> str:
    return f"""You are a strict mathematical trajectory auditor. Identify the FIRST verifiable semantic or mathematical error in the student's response.

Return JSON only, with exactly these keys:
{{"error_quote":"exact contiguous quote copied from the response, 8-80 words", "error_unit":"short description of the first invalid step", "error_type":"arithmetic|algebra|interpretation|unsupported_inference|answer_extraction|other", "reason":"one sentence explaining why it is invalid", "confidence":"high|medium|low"}}

Rules:
1. Read the entire problem and response before deciding.
2. Do not mark harmless wording, a trial explicitly labeled as wrong, or a later repeated mistake if an earlier invalid claim exists.
3. The quote must be copied exactly from the response. If the response is incomplete but contains no verifiable error, use error_quote="" and confidence="low".
4. Distinguish a wrong final answer caused by answer extraction from a wrong mathematical derivation.

Problem:
{row['prompt']}

Reference answer:
{row['gt']}

Student response:
<student_response>
{row['student_response']}
</student_response>

Now output exactly one JSON object and do not solve the problem again.
"""


def parse_json(text: str) -> dict[str, Any]:
    candidates = [line.strip() for line in text.splitlines() if line.strip().startswith("{")]
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            result["raw_auditor_output"] = text
            result["parse_status"] = "ok"
            return result
    return {"raw_auditor_output": text, "parse_status": "invalid_json"}


def main() -> None:
    a = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": audit_prompt(row)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,
        max_num_batched_tokens=16384,
        max_num_seqs=8,
    )
    schema = {
        "type": "object",
        "properties": {
            "error_quote": {"type": "string"},
            "error_unit": {"type": "string"},
            "error_type": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "string"},
        },
        "required": ["error_quote", "error_unit", "error_type", "reason", "confidence"],
        "additionalProperties": False,
    }
    outputs = llm.generate(
        rendered_prompts,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=a.max_new_tokens,
            seed=42,
            guided_decoding=GuidedDecodingParams(json=schema),
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row, request in zip(rows, outputs):
            generated = request.outputs[0].text
            proposal = parse_json(generated)
            handle.write(json.dumps({**row, "auditor_model": a.model, "auditor": proposal}, ensure_ascii=False) + "\n")
    print(f"audit_saved={a.output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
