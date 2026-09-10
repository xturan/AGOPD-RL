#!/usr/bin/env python3
"""Merge independently collected Appendix-F JSONL artifacts by prompt id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("inputs", nargs="+", type=Path)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    records = []
    seen = set()
    for path in a.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("prompt_id"), row.get("trajectory_kind"), row.get("state"), row.get("student_sample_id"))
            if key in seen:
                raise ValueError(f"duplicate record key {key} from {path}")
            seen.add(key)
            records.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged={a.output} rows={len(records)}", flush=True)


if __name__ == "__main__":
    main()
