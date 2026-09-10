"""Difficulty -> gain analysis: base p_hat (val census, n=8) vs GRPO-180 fixed eval.
输入: reports/figures/val_census_base_all.json + /tmp/fp_grpo_comfort180.jsonl
输出: 每桶 base_p_mean / grpo_score / gain 的终端表 + JSON
"""
import json
from collections import defaultdict

cen = {r["idx"]: r["p_hat"] for r in json.load(open("reports/figures/val_census_base_all.json"))}
ev = [json.loads(l) for l in open("/tmp/fp_grpo_comfort180.jsonl")]

# join
joined = []
for r in ev:
    p = cen.get(r["prompt_id"])
    if p is not None:
        joined.append((p, r["correct"], r.get("response_length", 0)))

print(f"joined {len(joined)}/1024")

def coarse(p):
    if p < 0.125: return "hard(p<.125)"
    if p < 0.75: return "frontier"
    return "easy(>.75)"

# fine buckets (p_hat discrete n=8 values) + coarse
fine = defaultdict(list)
coarse_b = defaultdict(list)
for p, c, ln in joined:
    fine[p].append((c, ln))
    coarse_b[coarse(p)].append((p, c, ln))

print("\n=== fine (by base p_hat value) ===")
print(f"{'p_hat':>5s} {'n':>5s} {'base_p':>7s} {'grpo':>7s} {'gain':>7s} {'rlen':>6s}")
out = []
for p in sorted(fine):
    rows = fine[p]
    n = len(rows)
    c = sum(r[0] for r in rows) / n
    ln = sum(r[1] for r in rows) / n
    gain = c - p
    out.append({"bucket": f"p_hat={p}", "n": n, "p_S_base": round(p, 3),
                "p_grpo": round(c, 3), "gain": round(gain, 3), "rlen": round(ln)})
    print(f"{p:5.3f} {n:5d} {p:7.3f} {c:7.3f} {gain:+7.3f} {ln:6.0f}")

print("\n=== coarse ===")
print(f"{'bucket':15s} {'n':>5s} {'base_p':>7s} {'grpo':>7s} {'gain':>7s}")
for b in ["hard(p<.125)", "frontier", "easy(>.75)"]:
    rows = coarse_b[b]
    if not rows: continue
    n = len(rows)
    pb = sum(r[0] for r in rows) / n
    gc = sum(r[1] for r in rows) / n
    print(f"{b:15s} {n:5d} {pb:7.3f} {gc:7.3f} {gc-pb:+7.3f}")

json.dump({"fine": out, "total": len(joined),
           "overall_base": sum(p for p, _, _ in joined) / len(joined),
           "overall_grpo": sum(c for _, c, _ in joined) / len(joined)},
          open("reports/figures/difficulty_gain.json", "w"), indent=1)
print("\nWROTE reports/figures/difficulty_gain.json")
