"""Stage 1:student(1.7B)采样轨迹 — 多卡分片并行(n=1)。
用法(2 卡): CUDA_VISIBLE_DEVICES=0 python sample_student_trajs.py --shard-idx 0 --n-shards 2 --out /tmp/trajs_shard0.jsonl
"""
import argparse, json, time

STUDENT = "models/Qwen3-1.7B"
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-questions", type=int, default=1024)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=2)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from agopd.reward.math_reward import score_math_response

    ds = load_dataset("parquet", data_files="data/dapo-verl-v1/val.parquet", split="train")
    rows = list(ds)[: args.n_questions]
    n_per = (len(rows) + args.n_shards - 1) // args.n_shards
    rows = rows[args.shard_idx * n_per : (args.shard_idx + 1) * n_per]
    print(f"shard {args.shard_idx}/{args.n_shards}: {len(rows)} questions", flush=True)

    llm = LLM(model=STUDENT, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=16384, max_num_seqs=256)

    problems = []
    for r in rows:
        gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth"))
        problems.append((str(r["prompt"]), gt))

    t0 = time.time()
    out_rows = []
    for i0 in range(0, len(problems), args.chunk):
        chunk = problems[i0 : i0 + args.chunk]
        prompts = [PROMPT_TEMPLATE.format(problem=p) for p, _ in chunk]
        outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.6, top_p=0.95,
                                                    top_k=20, max_tokens=args.max_tokens))
        for (content, gt), out in zip(chunk, outs):
            text = out.outputs[0].text
            correct = score_math_response(text, gt).correct
            out_rows.append({"prompt": content, "response": text, "gt": gt,
                             "correct": correct})
        print(f"  {min(i0 + args.chunk, len(problems))}/{len(problems)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    n_c = sum(1 for r in out_rows if r["correct"])
    print(f"WROTE {args.out}: {len(out_rows)} rows "
          f"({n_c} correct / {len(out_rows)-n_c} wrong) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
