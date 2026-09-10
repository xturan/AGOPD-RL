#!/usr/bin/env python3
"""Paired comparison of 4B verify arms A (pure GRPO) vs B (RL+OPD, 8B teacher).

Reads the 4-shard fixed-1024 eval jsonl of each arm, pairs by (shard, row order),
and reports strict / semantic accuracy with paired discordance, McNemar exact p,
and a normal-approx 95% CI on the difference.
"""
from __future__ import annotations
import argparse, json, glob, math, sys
from pathlib import Path

DIR = Path("reports/eval-dapo-v1")

def load(pattern):
    """returns list of dicts in (shard, row) order"""
    rows = []
    files = sorted(glob.glob(str(DIR / pattern)))
    assert files, f"no files for {pattern}"
    for f in files:
        for line in open(f, encoding="utf-8"):
            rows.append(json.loads(line))
    return rows

def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)

def paired(a, b, key):
    n = len(a)
    assert n == len(b), (n, len(b))
    ca = [bool(r.get(key)) for r in a]
    cb = [bool(r.get(key)) for r in b]
    acc_a = sum(ca) / n * 100
    acc_b = sum(cb) / n * 100
    d = acc_b - acc_a
    b01 = sum(1 for x, y in zip(ca, cb) if (not x) and y)   # A wrong -> B right
    b10 = sum(1 for x, y in zip(ca, cb) if x and (not y))   # A right -> B wrong
    # normal-approx 95% CI for paired difference of proportions
    var = (b01 + b10) / n - ((b01 - b10) / n) ** 2
    se = math.sqrt(max(var, 0.0) / n)
    ci = (d - 1.96 * se * 100, d + 1.96 * se * 100)
    return dict(n=n, acc_a=round(acc_a, 2), acc_b=round(acc_b, 2), delta_pp=round(d, 2),
                ci95=[round(ci[0], 2), round(ci[1], 2)],
                b01=b01, b10=b10, mcnemar_p=round(mcnemar_exact(b01, b10), 4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-pattern", default="scale4b_verify_a_grpo160_2048_s?.jsonl")
    ap.add_argument("--b-pattern", default="scale4b_verify_b_8bopd160_2048_s?.jsonl")
    ap.add_argument("--a-label", default="A: 4B pure GRPO")
    ap.add_argument("--b-label", default="B: 4B + 8B-teacher RL+OPD (unconditional, hard CE)")
    ap.add_argument("--out", default="scale4b_verify_ab_summary.json")
    args = ap.parse_args()

    A = load(args.a_pattern)
    B = load(args.b_pattern)
    out = {"arms": {"A": args.a_label, "B": args.b_label},
           "source": {"A": args.a_pattern, "B": args.b_pattern}}
    for key, name in [("correct", "strict"), ("semantic_correct", "semantic")]:
        out[name] = paired(A, B, key)
    # extra descriptive fields
    for tag, rows in [("A", A), ("B", B)]:
        n = len(rows)
        out.setdefault("descriptive", {})[tag] = {
            "rows": n,
            "trunc_mean_len": round(sum(r.get("response_length", 0) for r in rows) / n, 1),
            "truncation_rate": round(sum(1 for r in rows if r.get("truncated")) / n, 4),
            "answer_parse_rate": round(sum(1 for r in rows if r.get("parse_success")) / n, 4),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (DIR / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", DIR / args.out)

if __name__ == "__main__":
    main()
