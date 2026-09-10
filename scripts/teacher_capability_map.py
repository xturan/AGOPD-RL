"""Teacher capability map:按学生难度分桶的分层评测。
用法: python teacher_capability_map.py --student-grpo <dir> --n-per-bucket 60 [--shard-idx 0 --n-shards 2]
输入依赖: reports/figures/val_census_base_s{0,1}.json(val 1024 的 base p̂)
流程: 读 val census → 按 base p̂ 分 5 桶 → 每桶抽 N → 两模型(4B-GRPO / GRPO-28)各 n=8 → per-bucket p
产出: reports/figures/teacher_capability_map.json + 终端表
"""
import argparse, glob, json, random, statistics, sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student-grpo", default="models/Qwen3-1.7B-fp-grpo-comfort-180")
    p.add_argument("--teacher", default="models/Qwen3-4B-grpo-50step-ckpt2")
    p.add_argument("--n-per-bucket", type=int, default=60)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from agopd.reward.math_reward import score_math_response
    import re

    # val census
    census = []
    for f in sorted(glob.glob("reports/figures/val_census_base_all.json")):
        census.extend(json.load(open(f)))
    print(f"census rows: {len(census)}")

    # 从 val parquet 取 content
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files="data/dapo-verl-v1/val.parquet", split="train")
    by_idx = {}
    for r in ds:
        idx = str(r.get("extra_info", {}).get("index", ""))
        gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth"))
        content = str(r["prompt"])
        m = re.search(r"'content': '(.+?)', 'role'", content, re.S)
        if m:
            content = m.group(1).replace("\\n", "\n").replace("\\\\", "\\")
        by_idx[idx] = (content, gt)

    # 分桶
    def bucket(p):
        if p == 0: return "hard(p=0)"
        if p < 0.25: return "p<.25"
        if p < 0.5: return ".25-.5"
        if p < 0.75: return ".5-.75"
        return "easy(>.75)"
    buckets = defaultdict(list)
    for c in census:
        b = bucket(c["p_hat"])
        if c["idx"] in by_idx:
            buckets[b].append(c["idx"])
    print("bucket sizes:", {k: len(v) for k, v in buckets.items()})

    # 每桶抽 N
    rng = random.Random(42)
    sel = []
    for b, idxs in buckets.items():
        picked = rng.sample(idxs, min(args.n_per_bucket, len(idxs)))
        for i in picked:
            content, gt = by_idx[i]
            sel.append({"idx": i, "bucket": b, "content": content, "gt": gt, "p_S_base": next(c["p_hat"] for c in census if c["idx"] == i)})
    # shard 切分
    n_per = (len(sel) + args.n_shards - 1) // args.n_shards
    sel = sel[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(sel)} items")

    llm1 = LLM(model=args.teacher, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.45, max_model_len=4096, enforce_eager=False,
               max_num_batched_tokens=8192, max_num_seqs=128)
    llm2 = LLM(model=args.student_grpo, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.4, max_model_len=4096, enforce_eager=False,
               max_num_batched_tokens=8192, max_num_seqs=128)

    prompts = [r["content"] for r in sel]
    for r in sel:
        r["p_T"] = 0.0
        r["p_S"] = 0.0
    for i0 in range(0, len(prompts), 32):
        chunk = prompts[i0:i0+32]
        o1 = llm1.generate(chunk, SamplingParams(n=args.n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048))
        o2 = llm2.generate(chunk, SamplingParams(n=args.n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048))
        for rec, a, b in zip(sel[i0:i0+32], o1, o2):
            t1 = [x.text for x in a.outputs]
            t2 = [x.text for x in b.outputs]
            rec["p_T"] = sum(1 for t in t1 if score_math_response(t, rec["gt"]).correct) / len(t1)
            rec["p_S"] = sum(1 for t in t2 if score_math_response(t, rec["gt"]).correct) / len(t2)
        print(f"  {min(i0+32, len(prompts))}/{len(prompts)}", flush=True)

    # 汇总 per bucket
    agg = defaultdict(lambda: {"n": 0, "p_T": [], "p_S": []})
    for r in sel:
        agg[r["bucket"]]["n"] += 1
        agg[r["bucket"]]["p_T"].append(r["p_T"])
        agg[r["bucket"]]["p_S"].append(r["p_S"])
    order = ["hard(p=0)", "p<.25", ".25-.5", ".5-.75", "easy(>.75)"]
    print("\n=== Teacher capability map ===")
    print(f"{'bucket':12s} {'n':>4s} {'student_base':>12s} {'teacher_4B':>10s} {'student_GRPO':>12s} {'T-B':>8s} {'T-S':>8s}")
    result = []
    for b in order:
        if b not in agg: continue
        a = agg[b]
        pb = statistics.mean(r["p_S_base"] for r in sel if r["bucket"] == b) if sel else 0
        pt = statistics.mean(a["p_T"]); ps = statistics.mean(a["p_S"])
        result.append({"bucket": b, "n": a["n"], "p_S_base": round(pb,3), "p_T": round(pt,3), "p_S_grpo": round(ps,3),
                       "T_minus_Base": round(pt - pb,3), "T_minus_S": round(pt - ps,3)})
        print(f"{b:12s} {a['n']:4d} {pb:12.3f} {pt:10.3f} {ps:12.3f} {pt-pb:+8.3f} {pt-ps:+8.3f}")
    json.dump(result, open(f"reports/figures/teacher_capability_map_s{args.shard_idx}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
