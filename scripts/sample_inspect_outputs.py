"""GRPO 模型采样质检:抽题采样 → 分 correct/wrong → 打印推理摘录供人工审视。
用法: python sample_inspect_outputs.py --model <dir> --n-per-class 10
"""
import argparse, json, random, re
from pathlib import Path

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen3-1.7B-fp-grpo-comfort-180")
    p.add_argument("--n-questions", type=int, default=80, help="sample pool size")
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from agopd.reward.math_reward import score_math_response

    ds = load_dataset("parquet", data_files="data/dapo-verl-v1/val.parquet", split="train")
    rows = list(ds)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(rows)), args.n_questions)
    problems = []
    for i in idxs:
        r = rows[i]
        gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth"))
        problems.append((i, str(r["prompt"]), gt))

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.9, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=16384, max_num_seqs=256)
    prompts = [PROMPT_TEMPLATE.format(problem=p) for _, p, _ in problems]
    outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.6, top_p=0.95,
                                                top_k=20, max_tokens=2048))
    results = []
    for (idx, _, gt), out in zip(problems, outs):
        text = out.outputs[0].text
        res = score_math_response(text, gt)
        results.append({"idx": idx, "gt": gt, "correct": res.correct,
                        "reason": res.reason, "pred": res.predicted_answer, "text": text})

    corr = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]
    print(f"pool: {len(results)} | correct: {len(corr)} | wrong: {len(wrong)}")

    chosen = corr[: args.n_per_class] + wrong[: args.n_per_class]
    # 打乱展示顺序
    rng.shuffle(chosen)
    for r in chosen:
        tag = "CORRECT" if r["correct"] else "WRONG"
        print("=" * 100)
        print(f"[{tag}] val_idx={r['idx']} | GT={r['gt']} | pred={r['pred']} | reason={r['reason']}")
        # 打印题面前 150 字符 + 推理开头 500 + 结尾 300
        full_prompt = next(p for p, _, _ in zip(prompts, problems, problems) if True)
        print(f"--- prompt head ---")
        print(next(p for (i2, p, _), _2 in zip(problems, results) if i2 == r["idx"])[:150].replace("\n", " "))
        print("--- reasoning head ---")
        print(r["text"][:500])
        print("--- reasoning tail ---")
        print(r["text"][-300:])


if __name__ == "__main__":
    main()
