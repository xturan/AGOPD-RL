"""DAPO 训练集 p̂ 普查:base 1.7B rollout n=8 → Easy/Frontier/Hard 三桶。
输出:每桶 parquet + 分布统计 + frontier 图数据。支持 --shard-idx/--n-shards。
用法: python survey_p_hat.py --model models/Qwen3-1.7B --n 8 [--shard-idx 0 --n-shards 4]
"""
import argparse, json, time
from pathlib import Path
from datasets import load_dataset

from agopd.reward.math_reward import score_math_response
from build_coldstart_data import _ground_truth, _classify



def _gen_batch(llm, prompts, n, max_tokens, temperature, top_p, top_k, seed):
    """seed=-1 时不传 seed(vLLM per-request seed + FlashInfer fallback 会把吞吐打到 ~117 tok/s)"""
    from vllm import SamplingParams
    kw = dict(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)
    if seed is not None and seed >= 0:
        kw["seed"] = seed
    return llm.generate(prompts, SamplingParams(**kw))


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="data/dapo-verl-v1/train.parquet")
    p.add_argument("--model", default="models/Qwen3-1.7B")
    p.add_argument("--n", type=int, default=8, help="rollouts per prompt for p_hat")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=-1, help="-1 = no per-request seed (fast)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-batched-tokens", type=int, default=49152)
    p.add_argument("--max-seqs", type=int, default=512)
    p.add_argument("--chunk", type=int, default=128, help="prompts per vLLM batch call")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--out-dir", default="outputs/p_hat_survey")
    p.add_argument("--smoke", type=int, default=0, help="only process N prompts (timing)")
    p.add_argument("--exclude", default=None,
                   help="jsonl with existing buckets (index field) to skip")
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM

    ds = load_dataset("parquet", data_files=args.train_file, split="train")
    records = []
    for rec in ds:
        records.append({
            "index": str(rec.get("index") or rec.get("extra_info", {}).get("index") or ""),
            "problem": str(rec["prompt"]),
            "ground_truth": _ground_truth(rec),
        })
    print(f"total records: {len(records)}")
    if args.exclude:
        seen = set()
        for line in open(args.exclude):
            seen.add(json.loads(line)["index"])
        records = [r for r in records if r["index"] not in seen]
        print(f"excluded {len(seen)} existing prompts, remaining: {len(records)}")
    if args.smoke:
        records = records[: args.smoke]
        print(f"SMOKE mode: {len(records)} records")
    else:
        n_per = (len(records) + args.n_shards - 1) // args.n_shards
        records = records[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
        print(f"shard {args.shard_idx}/{args.n_shards}: records {len(records)}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=False,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
    )

    t0 = time.time()
    for rec in records:
        rec["prompt"] = PROMPT_TEMPLATE.format(problem=rec.pop("problem"))

    scored_rows = []
    for i in range(0, len(records), args.chunk):
        chunk = records[i : i + args.chunk]
        outputs = _gen_batch(
            llm, [r["prompt"] for r in chunk], args.n,
            args.max_tokens, args.temperature, args.top_p, args.top_k, args.seed,
        )
        for rec, out in zip(chunk, outputs):
            texts = [o.text for o in out.outputs]
            correct = [score_math_response(t, rec["ground_truth"]).correct for t in texts]
            p_hat = sum(correct) / len(correct)
            scored_rows.append({
                "index": rec["index"],
                "prompt": rec["prompt"],
                "ground_truth": rec["ground_truth"],
                "p_hat": p_hat,
                "n_correct": sum(correct),
                "bucket": _classify(p_hat),
            })
        if (i // args.chunk + 1) % 10 == 0:
            el = time.time() - t0
            done = i + len(chunk)
            print(f"  chunk {(i // args.chunk + 1)}/{max(1, (len(records) + args.chunk - 1) // args.chunk)} "
                  f"done | {done}/{len(records)} | {el:.0f}s", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_idx}" if args.n_shards > 1 else "full"
    import pandas as pd
    df = pd.DataFrame(scored_rows)
    df.to_parquet(out / f"p_hat_{tag}.parquet", index=False)
    from collections import Counter
    c = Counter(r["bucket"] for r in scored_rows)
    print(f"[shard {args.shard_idx}] records={len(scored_rows)} easy={c['easy']} "
          f"frontier={c['frontier']} hard={c['hard']} | {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
