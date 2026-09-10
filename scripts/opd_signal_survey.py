"""OPD 信号量化:comfort 桶题上 teacher(4B-GRPO)可教题比例。
student p̂(普查,base 1.7B n8)已存在;teacher 重采样 n=8 →
矩阵:student 不会(p̂<0.125)但 teacher 会(p̂_T>0)的题 = OPD 可用信号
"""
import argparse, json, time
from pathlib import Path

import pandas as pd

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-questions", type=int, default=1472)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=2)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from agopd.reward.math_reward import score_math_response

    # comfort 题(含 student p̂)
    train = pd.read_parquet("data/dapo-verl-v1/train_comfort.parquet")
    survey = pd.concat([pd.read_parquet(f) for f in
                        Path("outputs/p_hat_survey").glob("p_hat_shard*.parquet")])
    p_by_idx = dict(zip(survey["index"].astype(str), survey["p_hat"]))
    for bf in ["outputs/coldstart-17-4000-shard0/buckets.jsonl",
               "outputs/coldstart-17-4000-shard1/buckets.jsonl"]:
        for line in open(bf):
            d = json.loads(line)
            p_by_idx[d["index"]] = d["p_hat"]
    import ast, re
    rows = []
    for _, r in train.iterrows():
        content = str(r["prompt"])
        m = re.search(r"'content': '(.+?)', 'role'", content, re.S)
        if m:
            content = m.group(1).replace("\\n", "\n").replace("\\\\", "\\")
        rows.append({"content": content,
                     "idx": str(r["extra_info"]["index"]),
                     "p_S": p_by_idx.get(str(r["extra_info"]["index"]), None),
                     "gt": str(r["reward_model"].get("ground_truth") or r.get("extra_info", {}).get("ground_truth", ""))})
    rows = rows[: args.max_questions]
    n_per = (len(rows) + args.n_shards - 1) // args.n_shards
    rows = rows[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(rows)} questions | p̂ matched: {sum(1 for r in rows if r['p_S'] is not None)}")

    llm = LLM(model="models/Qwen3-4B-grpo-50step-ckpt2", dtype="bfloat16",
              tensor_parallel_size=1, gpu_memory_utilization=0.8, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=8192, max_num_seqs=128)

    t0 = time.time()
    prompts = [PROMPT_TEMPLATE.format(problem=r["content"]) for r in rows]
    out_path = Path(args.out or f"reports/opd_signal_shard{args.shard_idx}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fo:
        for i in range(0, len(prompts), 128):
            chunk = prompts[i:i+128]
            outs = llm.generate(chunk, SamplingParams(n=args.n, temperature=0.6, top_p=0.95,
                                                      top_k=20, max_tokens=2048))
            for r, out in zip(rows[i:i+128], outs):
                texts = [o.text for o in out.outputs]
                n_ok = sum(1 for t in texts if score_math_response(t, r["gt"]).correct)
                r["p_T"] = n_ok / len(texts)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i // 128 + 1) % 4 == 0:
                print(f"  {min(i+128, len(prompts))}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"WROTE {out_path}")

    # 汇总
    ok = [r for r in rows if r["p_S"] is not None and r["p_T"] is not None]
    n = len(ok)
    s_hard = [r for r in ok if r["p_S"] < 0.125]
    s_front = [r for r in ok if 0.125 <= r["p_S"] <= 0.75]
    print(f"\n=== OPD 信号矩阵(comfort {n} 题)===")
    print(f"student hard(p̂<0.125): {len(s_hard)} | 其中 teacher 会(p̂_T>0): "
          f"{sum(1 for r in s_hard if r['p_T'] > 0)} ({sum(1 for r in s_hard if r['p_T'] > 0)/max(1,len(s_hard)):.1%})")
    print(f"student frontier: {len(s_front)} | teacher 会: {sum(1 for r in s_front if r['p_T'] > 0)}")
    t_only = [r for r in ok if r["p_S"] < 0.125 and r["p_T"] > 0]
    both = [r for r in ok if r["p_S"] >= 0.125 and r["p_T"] > 0]
    print(f"teacher-only 可教题(student 不会 & teacher 会): {len(t_only)} ({len(t_only)/n:.1%})")
    print(f"teacher p̂_T 分布: mean={sum(r['p_T'] for r in ok)/n:.3f}")


if __name__ == "__main__":
    main()
