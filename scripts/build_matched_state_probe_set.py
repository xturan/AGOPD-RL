#!/usr/bin/env python3
"""Build wrong/correct matched trajectory rows for Appendix-F probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    output = []
    for row in rows:
        wrong = {**row, "trajectory_kind": "student_wrong"}
        output.append(wrong)
        correct_text = row.get("matched_correct_response")
        if not correct_text:
            continue
        correct_ids = tokenizer(correct_text, add_special_tokens=False)["input_ids"]
        correct = {
            **row,
            "trajectory_kind": "student_correct_matched",
            "student_response": correct_text,
            "error_onset_token": min(int(row["error_onset_token"]), len(correct_ids)),
            "error_quote": None,
            "error_unit": "matched correct trajectory at the wrong-path anchor offset",
            "error_type": "matched_control",
            "error_evidence": "同题正确轨迹，用错误轨迹的 token offset 作为状态位置对照。",
            "annotation_status": "matched_correct_control",
        }
        output.append(correct)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"matched_probe_set={a.output} rows={len(output)} wrong={len(rows)} correct_controls={len(output)-len(rows)}", flush=True)


if __name__ == "__main__":
    main()
