#!/usr/bin/env python3
"""Build the exact prompt pool used by the CRPTR transfer pilot.

The prompt column is written as Arrow list<struct>, preserving verl's chat
message shape instead of round-tripping it through a JSON string.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student-input", type=Path, required=True)
    p.add_argument("--teacher-input", type=Path, required=True)
    p.add_argument("--control-input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-prompts", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    students = {r["idx"]: r for r in read_jsonl(args.student_input)}
    teacher = defaultdict(dict)
    for r in read_jsonl(args.teacher_input):
        teacher[r["idx"]][r["condition"]] = r
    control = defaultdict(dict)
    for r in read_jsonl(args.control_input):
        control[r["idx"]][r["condition"]] = r

    candidates = []
    for idx in sorted(students):
        student = students[idx]
        root = teacher[idx].get("root")
        wrong = control[idx].get("wrong_path_matched")
        has_verified_root = root and any(s.get("correct") for s in root.get("teacher_samples", []))
        if root and wrong and has_verified_root:
            candidates.append({
                "data_source": "cptr_transfer",
                "prompt": [{"role": "user", "content": str(student["content"])}],
                "ability": "MATH",
                "reward_model": {"ground_truth": str(student["gt"]), "style": "rule-lighteval/MATH_v2"},
                "extra_info": {"index": str(idx)},
            })
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.max_prompts]
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(candidates, schema=schema), args.output)
    print(f"saved={args.output} rows={len(candidates)} unique_idx={len({r['extra_info']['index'] for r in candidates})}")


if __name__ == "__main__":
    main()
