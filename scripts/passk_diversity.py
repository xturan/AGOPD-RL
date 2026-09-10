"""Pass@k curve + diversity diagnostic (local 4090).
Samples val subset x n=8 (temp 1.0) under /think prompt.
Outputs: pass@k curve, distinct-correct-answers per question, self-bleu-ish proxy.
"""
import argparse
import json
import math
from collections import Counter

import pandas as pd
from vllm import LLM, SamplingParams

from agopd.reward.math_reward import score_math_response

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/stage2c_grpo240")
p.add_argument("--n", type=int, default=8)
p.add_argument("--offset", type=int, default=0)
p.add_argument("--half", type=int, default=0, help="0=first half, 1=second half")
p.add_argument("--limit", type=int, default=128)
p.add_argument("--prompt-mode", default="think", choices=["think", "no_think"])
p.add_argument("--gpu", type=int, default=1)
args = p.parse_args()

val_all = pd.read_parquet("data/dapo-verl-v1/val.parquet")
if args.half == 1:
    val = val_all.iloc[args.limit//2:args.limit].reset_index(drop=True)
else:
    val = val_all.head(args.limit//2)
prompts, gts = [], []
for _, r in val.iterrows():
    c = str(r["prompt"])
    if args.prompt_mode == "think":
        c = c.replace("/no_think", "/think")
    prompts.append(c)
    gts.append(str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth", "")))

llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
          gpu_memory_utilization=0.9, max_model_len=16384, enforce_eager=False,
          max_num_batched_tokens=8192, max_num_seqs=64)
sp = SamplingParams(n=args.n, temperature=1.0, top_p=0.95, top_k=20, max_tokens=8192)
outs = llm.generate(prompts, sp)

n_correct = []      # per question: number of correct samples
distinct_correct = []  # per question: distinct predicted answers among correct
lens = []
for o, gt in zip(outs, gts):
    texts = [x.text for x in o.outputs]
    scored = [score_math_response(t, gt) for t in texts]
    correct = [s for s in scored if s.correct]
    n_correct.append(len(correct))
    distinct_correct.append(len({s.predicted_answer for s in correct}))
    lens += [len(t.split()) for t in texts]

# pass@k unbiased (compute per-question with the k-of-n formula) — use empirical: P(at least 1 correct in first k draws) approximated via complement of all-fail over observed multiset
p_hat = [c / args.n for c in n_correct]
passk = {}
for k in [1, 2, 4, 8]:
    # empirical: for each question, probability at least one of k random draws correct = 1 - comb(n-c, k)/comb(n, k)
    vals = []
    for c in n_correct:
        fail = args.n - c
        if args.n >= k:
            pk = 1 - math.comb(fail, k) / math.comb(args.n, k)
        else:
            pk = 1.0 if c > 0 else 0.0
        vals.append(pk)
    passk[k] = sum(vals) / len(vals)

res = {
    "model": args.model, "mode": args.prompt_mode, "n": args.n, "questions": len(outs),
    "p1_empirical": sum(1 for c in n_correct if c > 0) / len(n_correct),
    "passk": passk,
    "mean_correct_per_q": sum(n_correct) / len(n_correct),
    "mean_distinct_correct_ans": sum(distinct_correct) / len(n_correct),
    "frac_q_with_multiple_correct": sum(1 for c in n_correct if c >= 2) / len(n_correct),
    "mean_len_tokens": sum(lens) / len(lens),
}
print(json.dumps(res, indent=1))
