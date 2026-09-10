"""阶段 2A RFT 采样:GRPO 模型全量采样(temp 1.0,n=8),轨迹全文落盘。
产出:每题 p̂' + 8 条轨迹全文(correct 标记)→ 供清洗与 hard 重分层。
用法(云端): CUDA_VISIBLE_DEVICES=$i python rft_survey.py --model <dir> --shard-idx $i --n-shards 4
"""
import argparse, json, random
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", default="${AGOPD_ROOT}/data/dapo-verl-v1/train.parquet")
    p.add_argument("--n", type=int, default=2000, help="total questions (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--out-dir", default="${AGOPD_ROOT}/outputs/rft_survey")
    p.add_argument("--exclude", default=None, help="jsonl/dir of previously sampled rows (idx) to skip")
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from agopd.reward.math_reward import score_math_response

    train = pd.read_parquet(args.train_file)
    records = []
    for _, r in train.iterrows():
        idx = str(r["extra_info"]["index"])
        content = str(r["prompt"])
        gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth", ""))
        records.append({"idx": idx, "content": content, "gt": gt})
    if args.exclude:
        import glob as _g
        excl = set()
        for ef in _g.glob(args.exclude):
            for line in open(ef):
                excl.add(json.loads(line)["idx"])
        records = [r for r in records if r["idx"] not in excl]
        print(f"excluded {len(excl)}, remaining: {len(records)}", flush=True)
    rng = random.Random(args.seed)
    if args.n and args.n < len(records):
        records = rng.sample(records, args.n)
    n_per = (len(records) + args.n_shards - 1) // args.n_shards
    records = records[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(records)} questions", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=16384, max_num_seqs=256)
    # prompt = content 直接(val/train content 已含完整模板,不二次包装)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"rft_shard{args.shard_idx}.jsonl"
    n_ok_total = 0
    with open(out_f, "w") as fo:
        for i0 in range(0, len(records), 64):
            chunk = records[i0:i0+64]
            prompts = [r["content"] for r in chunk]
            outs = llm.generate(prompts, SamplingParams(n=8, temperature=1.0, top_p=0.95,
                                                        top_k=20, max_tokens=2048))
            for r, out in zip(chunk, outs):
                texts = [o.text for o in out.outputs]
                scored = [score_math_response(t, r["gt"]).correct for t in texts]
                r["p_hat"] = sum(scored) / len(scored)
                r["correct_texts"] = [t for t, c in zip(texts, scored) if c]
                n_ok_total += sum(scored)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i0 // 64 + 1) % 5 == 0:
                print(f"  {min(i0+64, len(records))}/{len(records)}", flush=True)
    n = len(records)
    print(f"WROTE {out_f}: {n} questions, correct trajectories: {n_ok_total}", flush=True)


if __name__ == "__main__":
    main()
