"""Teacher entropy x competence offline diagnostic (local 4090).
对 val 分桶题:teacher 4B 在 student 轨迹上逐 token 前向算 entropy,
join teacher map 的题级 competence(p_T),做 2x2 与 calibration 分析。

用法: python teacher_entropy_diag.py --student-trajs /tmp/stage2c_eval2048.jsonl --limit 60
"""
import argparse
import json

import pandas as pd
import torch

from agopd.reward.math_reward import score_math_response


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="models/Qwen3-4B-grpo-50step-ckpt2")
    p.add_argument("--student-trajs", default="/tmp/stage2c_eval2048.jsonl")
    p.add_argument("--bucket-json", default="reports/figures/teacher_capability_map_s0.json")
    p.add_argument("--limit", type=int, default=60, help="questions to process")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. bucket map: idx -> bucket (from teacher map run)
    tmap = json.load(open(args.bucket_json))
    # teacher map 里没有 idx?看字段——用 student trajs 自身分桶
    trajs = [json.loads(l) for l in open(args.student_trajs)]

    # teacher competence per question: need teacher answers. teacher map has p_T per bucket
    # 但 per-question p_T 不在 tmap 里 → 这里用已有 grpo240_eval 中 teacher 对照?简化:
    # 对每条轨迹 teacher 重新答 n=4(生成)拿 competence;再对 student 轨迹做 teacher 前向拿 entropy。
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.5, max_model_len=8192, enforce_eager=False,
              max_num_batched_tokens=8192, max_num_seqs=64)
    # 抽取题面(取 student traj 前 60 条,需 prompt 还原 → 用 val census idx join?直接用 traj 内嵌?eval jsonl 无 prompt 原文
    # 改用 val.parquet 前 N 题 + student 轨迹
    import glob
    val = pd.read_parquet("data/dapo-verl-v1/val.parquet").head(args.limit)
    gts = [str(r.get("reward_model", {}).get("ground_truth") or r.get("extra_info", {}).get("ground_truth", "")) for _, r in val.iterrows()]
    # teacher competence: n=4 生成
    prompts = [str(r["prompt"]) for _, r in val.iterrows()]
    outs = llm.generate(prompts, SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048))
    comp = []
    for o, gt in zip(outs, gts):
        c = sum(1 for x in o.outputs if score_math_response(x.text, gt).correct) / 4
        comp.append(c)
    # 2. teacher forward on a few student trajectories for token entropy
    n_traj = min(20, len(trajs))
    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    # 用 think prompt 还原:val prompt 已含模板;student 输出拼接
    results = []
    for i in range(n_traj):
        r = trajs[i]
        text = str(val.iloc[i]["prompt"]).replace("/no_think", "/think") + r["output"]
        ids = tok(text, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            lg = model(input_ids=ids[:, :-1]).logits[0]  # (seq-1, vocab)
        probs = torch.softmax(lg.float(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(-1)  # (seq-1,)
        target = ids[0, 1:]
        topk = probs.topk(5, dim=-1)
        her = (ent > 0.8).float().mean().item()
        results.append({
            "idx": i, "question_comp": comp[i],
            "HER_08": her, "mean_entropy": ent.mean().item(),
            "p_top1": topk.values[:, 0].mean().item(),
        })
        print(json.dumps(results[-1]), flush=True)
    json.dump(results, open("/tmp/teacher_entropy_diag.json", "w"), indent=1)
    print("WROTE /tmp/teacher_entropy_diag.json")


if __name__ == "__main__":
    main()
