#!/usr/bin/env python3
"""Cross-scale p-hat census (matched subset), non-think chat-template protocol.

Sample --max-prompts prompts deterministically (seed 42) from DAPO train.parquet,
draw n=8 vLLM completions per prompt (T=0.6), score each with the rule verifier,
and record k = number correct per prompt. Supports --shard-idx/--n-shards.
Usage (cloud, one process per GPU):
  CUDA_VISIBLE_DEVICES=i python census_scale.py --model models/Qwen3-4B \
    --max-prompts 2000 --shard-idx i --n-shards 4 --out outputs/census_4b
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from agopd.reward.math_reward import score_math_response

def _messages(value):
    if hasattr(value, "tolist"): value = value.tolist()
    if not isinstance(value, list): return [{"role": "user", "content": str(value)}]
    return [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in value if isinstance(m, dict)]

def _render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\\n\\n{problem}\\n\\n"
    'Remember to put your answer on its own line after "Answer:".\\n\\n/no_think'
)

def _gt(value): return str(value.get("ground_truth") or "") if isinstance(value, dict) else ""
def _id(value): return str(value.get("index") or value.get("id") or "") if isinstance(value, dict) else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-file", default="data/dapo-verl-v1/train.parquet")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=2000)
    ap.add_argument("--list-seed", type=int, default=42)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--gmu", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--prompt-mode", default="chat", choices=["chat","survey"])
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_file)
    rng = np.random.RandomState(args.list_seed)
    sel = rng.permutation(len(df))[: args.max_prompts]
    my = sel[args.shard_idx::args.n_shards]
    if args.smoke: my = sel[: args.smoke]
    sub = df.iloc[my].to_dict("records")
    print(f"model={args.model} shard {args.shard_idx}/{args.n_shards} prompts={len(sub)}", flush=True)

    if args.prompt_mode == "survey":
        def _prob(v):
            if hasattr(v,"tolist"): v=v.tolist()
            return str(v)
        prompts = [PROMPT_TEMPLATE.format(problem=_prob(r["prompt"])) for r in sub]
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [_render(tok, _messages(r["prompt"])) for r in sub]
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=args.gmu,
              max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for b in range(0, len(sub), args.chunk):
        blk_p = prompts[b:b+args.chunk]; blk_r = sub[b:b+args.chunk]
        outs = llm.generate(blk_p, sp)
        for rec, res in zip(blk_r, outs):
            gts = _gt(rec["reward_model"]); idx = _id(rec.get("extra_info", {}))
            k = 0
            for o in res.outputs:
                sc = score_math_response(o.text, gts)
                k += int(bool(sc.correct))
            rows.append({"idx": idx, "k": k, "n": args.n, "gt": gts})
            with open(f"{args.out}/shard{args.shard_idx}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"idx": idx, "k": k, "n": args.n}) + "\n")
        print(f"  batch {b//args.chunk} done, cumulative rows {len(rows)}", flush=True)

    ks = np.array([r["k"] for r in rows])
    if len(ks):
        p0 = (ks == 0).mean()
        meanp = ks.mean() / args.n
        se = np.sqrt(p0 * (1 - p0) / len(ks))
        print(f"SHARD p0={p0:.4f} mean_p={meanp:.4f} n_prompts={len(ks)} "
              f"p0_ci=[{p0-1.96*se:.4f},{p0+1.96*se:.4f}]", flush=True)

if __name__ == "__main__":
    main()
