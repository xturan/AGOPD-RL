#!/usr/bin/env python3
"""Prepare high-confidence semantic error rows for the Appendix-F probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANNOTATIONS = {
    "a854c341-0ac3-41e6-bbc8-c5d87226b71b": {
        "sample_id": 0,
        "error_quote": r"$ \gcd(N, 20) = 1 $, which means $ N $ must not be divisible by 4 or 5.",
        "error_unit": "误读互素条件 gcd(N,20)=1，把排除因子 2 错写为排除因子 4。",
        "error_type": "interpretation",
        "error_evidence": "与 20 互素要求不能被 2 整除且不能被 5 整除，而不是不能被 4 整除。",
    },
    "b9d85a26-59fe-42e0-920a-331dc4b376d9": {
        "sample_id": 5,
        "error_quote": r"\sqrt{410357169} = 20257 \Rightarrow \text{Perfect square!}",
        "error_unit": "错误计算平方根并据此把 n=7 判定为解。",
        "error_type": "arithmetic",
        "error_evidence": "20257^2 不等于 410357169，因此该平方根等式不成立。",
    },
    "7034407a-fb7a-46d5-9987-f58b32a3466c": {
        "sample_id": 0,
        "error_quote": r"g(2) = -2 - g(0)",
        "error_unit": "错误使用中心对称关系，遗漏了中心纵坐标带来的两倍常数。",
        "error_type": "interpretation",
        "error_evidence": "关于 (1,-2) 中心对称应满足 g(2)+g(0)=-4，而不是 -2。",
    },
    "c90baad1-aa60-487d-b05c-bc647455bdc4": {
        "sample_id": 3,
        "error_quote": r"the number of red candies be $ 0.75x $, since they are 10% and 25% respectively.",
        "error_unit": "把原始红色糖果比例 25% 误写为 75%。",
        "error_type": "arithmetic",
        "error_evidence": "题干明确给出原碗红色糖果占 25%，应为 0.25x。",
    },
    "602690b8-58c0-479b-9648-2d47efe768ca": {
        "sample_id": 0,
        "error_quote": r"2 \times 36 = 72",
        "error_unit": "把有序的正负因子对重复计数，导致 distinct line 数量翻倍。",
        "error_type": "interpretation",
        "error_evidence": "根为无序的两个交点；交换两个根不会产生新直线，因此 36 个正因子对应 36 条候选线，而不是 72 条。",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="models/Qwen3-1.7B")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for row in rows:
        annotation = ANNOTATIONS.get(row["prompt_id"])
        if annotation is None or row["student_sample_id"] != annotation["sample_id"]:
            continue
        response = row["student_response"]
        char_offset = response.find(annotation["error_quote"])
        if char_offset < 0:
            raise ValueError(f"quote not found for {row['prompt_id']}: {annotation['error_quote']!r}")
        token_offset = len(tokenizer(response[:char_offset], add_special_tokens=False)["input_ids"])
        selected.append({
            **row,
            **annotation,
            "error_onset_char": char_offset,
            "error_onset_token": token_offset,
            "annotation_confidence": "high",
            "auditor": "Codex manual review",
            "annotation_status": "reviewed_for_probe",
        })
    if len(selected) != len(ANNOTATIONS):
        found = {row["prompt_id"] for row in selected}
        raise ValueError(f"missing annotated rows: {sorted(set(ANNOTATIONS) - found)}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in selected:
        print(f"reviewed={row['prompt_id']} onset_token={row['error_onset_token']} type={row['error_type']}", flush=True)
    print(f"reviewed_probe_set={a.output} rows={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
