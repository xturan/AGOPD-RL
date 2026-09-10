#!/usr/bin/env python3
"""Create a small, manually reviewed Appendix-F probe set.

The quotes below were reviewed against the actual Student responses.  The
script maps the first quoted invalid claim to a response-token offset with
the Student tokenizer, so the downstream probe can use the exact state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANNOTATIONS = {
    "df9c0d93-45cf-465e-bd20-6113282c732a": {
        "sample_id": 0,
        "error_quote": "A **regular tetrahedron** $ P-ABC $, meaning all edges are equal.",
        "error_unit": "将正三棱锥误读为所有棱相等的正四面体，改变了题目的几何对象。",
        "error_type": "interpretation",
        "error_evidence": "题干是正三棱锥，给定底边与高；不能直接套用正四面体的棱高关系。",
    },
    "16771d40-3e91-4b2e-bff9-84cbbaa8d702": {
        "sample_id": 2,
        "error_quote": "d_4 = n",
        "error_unit": "把第四小的约数错误地设为最大约数 n。",
        "error_type": "unsupported_inference",
        "error_evidence": "题目只定义 d_k=n；当 k 大于 4 时，d_4 不等于 n。",
    },
    "1b9a6c2a-6b9d-4eb3-b655-36858d2fb3ed": {
        "sample_id": 0,
        "error_quote": "pqr + p + q + r = (p + 1)(q + 1)(r + 1) - 1",
        "error_unit": "错误展开 (p+1)(q+1)(r+1)，遗漏 pq、pr、qr 三个交叉项。",
        "error_type": "algebra",
        "error_evidence": "完整展开还包含 pq+pr+qr，因此该等式不成立。",
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
        selected.append(
            {
                **row,
                **annotation,
                "error_onset_char": char_offset,
                "error_onset_token": token_offset,
                "annotation_confidence": "high",
                "auditor": "Codex manual review",
                "annotation_status": "reviewed_for_probe_pilot",
            }
        )
    if len(selected) != len(ANNOTATIONS):
        found = {row["prompt_id"] for row in selected}
        missing = sorted(set(ANNOTATIONS) - found)
        raise ValueError(f"missing annotated rows: {missing}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in selected:
        print(f"annotated={row['prompt_id']} onset_token={row['error_onset_token']} quote={row['error_quote']}", flush=True)
    print(f"manual_probe_set={a.output} rows={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
