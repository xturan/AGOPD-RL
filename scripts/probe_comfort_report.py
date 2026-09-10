#!/usr/bin/env python3
"""Report the comfort-300 probes: teacher(8B) vs student(4B base) vs trained student A.

Usage (cloud, cwd=${AGOPD_ROOT}):
  python scripts/probe_comfort_report.py [--samples 2]
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

PATHS = {
    "teacher8b": "reports/eval-dapo-v1/probe_teacher8b_comfort300.jsonl",
    "student4B-base": "reports/eval-dapo-v1/probe_student4bbase_comfort300.jsonl",
    "studentA-trained": "reports/eval-dapo-v1/probe_studentA_comfort300.jsonl",
}


def load(path: str):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def rate(rows, key):
    return sum(bool(r.get(key)) for r in rows) / len(rows) * 100


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1)) / (2 ** n)
    return b01, b10, min(1.0, round(p, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    data = {k: load(v) for k, v in PATHS.items() if Path(v).exists()}
    for name, rows in data.items():
        n = len(rows)
        print(
            "%-16s n=%d strict=%.1f%% semantic=%.1f%% answer_marker=%.1f%% boxed=%.1f%% mean_len=%.0f"
            % (name, n, rate(rows, "correct"), rate(rows, "semantic_correct"),
               rate(rows, "answer_marker"), rate(rows, "boxed_marker"),
               sum(r.get("response_length", 0) for r in rows) / n)
        )

    if {"teacher8b", "studentA-trained"} <= set(data):
        A = [bool(r.get("correct")) for r in data["studentA-trained"]]
        T = [bool(r.get("correct")) for r in data["teacher8b"]]
        only_A = sum(1 for x, y in zip(A, T) if x and not y)
        only_T = sum(1 for x, y in zip(A, T) if y and not x)
        b01, b10, p = mcnemar(A, T)
        print("\npaired teacher8b vs studentA-trained: A-only-correct=%d teacher-only-correct=%d McNemar p=%.4f"
              % (only_A, only_T, p))

    for name in ("teacher8b", "studentA-trained"):
        if name not in data:
            continue
        print("\n=== %s samples ===" % name)
        for r in data[name][: args.samples]:
            out = (r.get("output") or "").strip().replace("\n", " ")
            print("gt=%s correct=%s len=%s" % (r.get("ground_truth"), r.get("correct"), r.get("response_length")))
            print("  head:", out[:260])
            print("  tail:", out[-260:])


if __name__ == "__main__":
    main()
