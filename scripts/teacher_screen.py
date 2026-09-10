"""Teacher 候选筛选:同一 comfort 题池 × n8 采样 → p̂_T + 输出质量。
候选:Qwen3-8B(base) / OpenMath-Nemotron-7B(与已有 4B-GRPO 数据对比)
用法: CUDA_VISIBLE_DEVICES=0 python teacher_screen.py --model models/Qwen3-8B --n 300 --out reports/teacher8b_screen.jsonl
"""
import argparse, json, random, re, statistics
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from agopd.reward.math_reward import score_math_response
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # comfort 题(内容直接用 — content 已含模板,与训练一致)
    import ast
    train = pd.read_parquet("data/dapo-verl-v1/train_comfort.parquet")
    rows = []
    for _, r in train.iterrows():
        content = str(r["prompt"])
        m = re.search(r"'content': '(.+?)', 'role'", content, re.S)
        if m:
            content = m.group(1).replace("\\n", "\n").replace("\\\\", "\\")
        gt = str(r["reward_model"].get("ground_truth") or r.get("extra_info", {}).get("ground_truth", ""))
        rows.append({"content": content, "gt": gt})
    rng = random.Random(args.seed)
    rows = rng.sample(rows, min(args.n, len(rows)))
    n_per = (len(rows) + args.n_shards - 1) // args.n_shards
    rows = rows[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(rows)} questions", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=1, gpu_memory_utilization=0.72, max_model_len=4096,
              enforce_eager=True, max_num_batched_tokens=4096, max_num_seqs=32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    prompts = [r["content"] for r in rows]
    for i0 in range(0, len(prompts), 32):
        chunk = prompts[i0:i0+32]
        outs = llm.generate(chunk, SamplingParams(n=8, temperature=0.6, top_p=0.95,
                                                  top_k=20, max_tokens=2048))
        for r, out in zip(rows[i0:i0+32], outs):
            texts = [o.text for o in out.outputs]
            n_ok = sum(1 for t in texts if score_math_response(t, r["gt"]).correct)
            lens = [len(tok(t)["input_ids"]) for t in texts]
            r["p_T"] = n_ok / len(texts)
            r["len_mean"] = statistics.mean(lens)
            r["trunc_rate"] = sum(1 for l in lens if l >= 2040) / len(lens)
            records.append(r)
        print(f"  {min(i0+32, len(prompts))}/{len(prompts)}", flush=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    print(f"\n=== Teacher screen ({args.model}, n={n}) ===")
    print(f"p_T mean: {statistics.mean(r['p_T'] for r in records):.3f}")
    print(f"teacher 会(p_T>0): {sum(1 for r in records if r['p_T'] > 0)/n:.1%}")
    print(f"teacher 强(p_T>=0.5): {sum(1 for r in records if r['p_T'] >= 0.5)/n:.1%}")
    print(f"len_mean: {statistics.mean(r['len_mean'] for r in records):.0f} | trunc: {statistics.mean(r['trunc_rate'] for r in records):.1%}")


if __name__ == "__main__":
    main()
