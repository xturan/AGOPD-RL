"""温度对照采样:同题池 × {0.3, 0.6, 1.0, 1.5},对比正确率/长度/截断/多样性。
用法: python temp_sweep.py --model <dir> --n 60
"""
import argparse, json, random, re, statistics
from pathlib import Path

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)
TEMPS = [0.3, 0.6, 1.0, 1.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-1.7B-fp-grpo-comfort-180")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from agopd.reward.math_reward import score_math_response
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("parquet", data_files="data/dapo-verl-v1/val.parquet", split="train")
    rows = list(ds)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(rows)), args.n)
    problems = []
    for i in idxs:
        r = rows[i]
        gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth"))
        problems.append((i, str(r["prompt"]), gt))
    prompts = [PROMPT_TEMPLATE.format(problem=p) for _, p, _ in problems]

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=1, gpu_memory_utilization=0.75, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=16384, max_num_seqs=256)

    print("temp | correct | len_mean(tok) | trunc2048 | unique_first50")
    for T in TEMPS:
        outs = llm.generate(prompts, SamplingParams(n=1, temperature=T, top_p=0.95,
                                                    top_k=20, max_tokens=2048))
        n_ok = 0
        lens = []
        trunc = 0
        firsts = []
        for (_, _, gt), out in zip(problems, outs):
            text = out.outputs[0].text
            res = score_math_response(text, gt)
            if res.correct:
                n_ok += 1
            lt = len(tok(text)["input_ids"])
            lens.append(lt)
            if lt >= 2040:
                trunc += 1
            firsts.append(text[:50])
        print(f"{T:.1f} | {n_ok}/{args.n} ({n_ok/args.n:.1%}) | {statistics.mean(lens):.0f} | "
              f"{trunc} ({trunc/args.n:.1%}) | {len(set(firsts))} unique")


if __name__ == "__main__":
    main()
