"""GRPO 模型输出质量审计:200 题池全文采样 → 量化问题面。
指标:模板 echo 率 / 直接答案率 / 尾部重复率 / 推理格式分布 / 长度截断。
用法: python quality_audit.py --model <dir> --n 200
"""
import argparse, json, random, re, statistics
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
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/grpo_quality_audit.jsonl")
    return p.parse_args()


def has_repetition(text, min_span=8, min_repeat=4):
    """检测尾部重复循环:某 8+ 字符片段连续重复 ≥4 次"""
    for m in re.finditer(r"(.{%d,})\1{%d,}" % (min_span, min_repeat - 1), text[-800:]):
        return True
    return False


def classify_format(text):
    """推理格式分类"""
    body = re.sub(r"^.*?Answer: \$?Answer", "", text, count=1, flags=re.S)  # 去模板 echo 前缀
    if re.search(r"Step \d+|步骤 \d+", body, re.I):
        return "step-labeled"
    if re.search(r"^\s*\d+[\.\)、]", body, re.M):
        return "numbered"
    if len(body.strip()) < 60:
        return "no-reasoning"
    if re.search(r"\$.*\$|\\\\frac|\\sum|\\int", body):
        return "prose-with-math"
    return "plain-prose"


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    from agopd.reward.math_reward import score_math_response

    ds = load_dataset("parquet", data_files="data/dapo-verl-v1/val.parquet", split="train")
    rows = list(ds)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(rows)), args.n)

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=1, gpu_memory_utilization=0.75, max_model_len=4096,
              enforce_eager=False, max_num_batched_tokens=16384, max_num_seqs=256)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i0 in range(0, args.n, 100):
        chunk = idxs[i0:i0+100]
        problems = []
        for i in chunk:
            r = rows[i]
            gt = str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth"))
            problems.append((i, str(r["prompt"]), gt))
        # 注意:val content 已含完整模板,直接作为 prompt(与 verl 训练渲染一致);
        # 二次包装(PROMPT_TEMPLATE.format)会导致双重模板 → 模型 echo/混乱
        prompts = [p for _, p, _ in problems]
        outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.6, top_p=0.95,
                                                    top_k=20, max_tokens=2048))
        for (idx, _, gt), out in zip(problems, outs):
            text = out.outputs[0].text
            res = score_math_response(text, gt)
            records.append({"idx": idx, "gt": gt, "correct": res.correct, "text": text})
        print(f"  sampled {len(records)}/{args.n}", flush=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    n = len(records)
    n_ok = sum(1 for r in records if r["correct"])
    echo = sum(1 for r in records if re.search(r"Answer:\s*\$?Answer", r["text"][:200]))
    direct = sum(1 for r in records if classify_format(r["text"]) == "no-reasoning")
    repeat = sum(1 for r in records if has_repetition(r["text"]))
    trunc = sum(1 for r in records if len(r["text"]) >= 2000)
    lens = [len(r["text"]) for r in records]
    fmts = {}
    for r in records:
        f = classify_format(r["text"])
        fmts[f] = fmts.get(f, 0) + 1
    # 正确 vs 错误的格式差异
    print(f"\n=== 质量审计(n={n}) ===")
    print(f"correct: {n_ok} ({n_ok/n:.1%})")
    print(f"模板 echo(开头复述 Answer: $Answer): {echo} ({echo/n:.1%})")
    print(f"直接答案无推理: {direct} ({direct/n:.1%})")
    print(f"尾部重复循环: {repeat} ({repeat/n:.1%})")
    print(f"近截断(len>=2000): {trunc} ({trunc/n:.1%})")
    print(f"长度: mean={statistics.mean(lens):.0f} median={statistics.median(lens):.0f}")
    print(f"格式分布: {fmts}")
    # 正确轨迹中的问题率(关键:作为 SFT 数据源的可信度)
    ok_recs = [r for r in records if r["correct"]]
    ok_echo = sum(1 for r in ok_recs if re.search(r"Answer:\s*\$?Answer", r["text"][:200]))
    ok_repeat = sum(1 for r in ok_recs if has_repetition(r["text"]))
    ok_direct = sum(1 for r in ok_recs if classify_format(r["text"]) == "no-reasoning")
    print(f"\n=== 正确轨迹内部(将作 SFT 数据源)===")
    print(f"correct n={len(ok_recs)}: echo {ok_echo} ({ok_echo/max(1,len(ok_recs)):.1%}) | "
          f"repeat {ok_repeat} ({ok_repeat/max(1,len(ok_recs)):.1%}) | direct {ok_direct} ({ok_direct/max(1,len(ok_recs)):.1%})")


if __name__ == "__main__":
    main()
