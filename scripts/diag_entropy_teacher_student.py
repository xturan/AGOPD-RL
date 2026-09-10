"""P0 诊断:Teacher State Mismatch — 在 Student 轨迹的 state 上比较 H_T vs H_S。

流程:
1. 取 DAPO 题 N 道,student(Qwen3-4B-grpo-ckpt2)vLLM rollout n=4
2. 每题挑 1 条 correct + 1 条 wrong 轨迹(若存在)
3. 每条轨迹 HF 前向一次(student GPU0 / teacher GPU1 并行),取全位置 logits
4. 算 per-position H_S/H_T,按轨迹类型(correct/wrong)+ 输出位置分桶统计

检验假设:wrong 轨迹(≈训练中 all-zero/mixed 的错误成分)上
H_T(state) 显著高于 correct 轨迹 → Teacher 在 Student 陌生区域不确定,
把高熵分布蒸给学生 → entropy inflation。

用法: python diag_entropy_teacher_student.py [--n-questions 30] [--max-trajs-per-q 2]
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = "models/Qwen3-1.7B"
TEACHER = "models/Qwen3-4B-grpo-50step-ckpt2"
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".\n\n/no_think'
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs", required=True, help="stage1 trajectory jsonl")
    p.add_argument("--n", type=int, default=1, help="rollouts per question")
    p.add_argument("--chunk", type=int, default=256, help="questions per vLLM batch call")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--out", default="reports/diag_entropy_ts.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 读轨迹(stage1 sample_student_trajs.py 产出)----------
    trajs = []
    for line in open(args.trajs):
        d = json.loads(line)
        trajs.append((d["prompt"], d["response"], d["gt"], bool(d["correct"])))
    print(f"loaded {len(trajs)} trajectories "
          f"({sum(1 for t in trajs if t[3])} correct / {sum(1 for t in trajs if not t[3])} wrong)", flush=True)

    # ---------- 2. HF logits ----------
    print("loading HF models (student 1.7B GPU0 / teacher 4B GPU1)", flush=True)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    torch.cuda.set_device(0)
    ms = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to("cuda:0").eval()
    torch.cuda.set_device(1)
    mt = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to("cuda:1").eval()

    TOPK = 256

    def pos_entropy(model, device, text):
        """一次前向,topk 重归一化熵(≈ 全 vocab 熵,后处理快 ~300x)。"""
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()  # (T, V)
        # 数值稳定 topk 熵:减每位置 max
        m = logits.max(dim=-1, keepdim=True).values
        z = (logits - m)
        vals, _ = torch.topk(z, k=min(TOPK, z.shape[-1]), dim=-1)
        p = vals.exp()
        p = p / p.sum(dim=-1, keepdim=True)
        ent = -(p * p.log()).sum(dim=-1)  # (T,)
        del logits, z, vals, p
        return ent, ids.shape[1]

    import threading

    def run_model(model, device, texts, out, idx0):
        """线程内逐条前向(独占自己的 GPU),返回 entropy tensor 列表"""
        ents = []
        for j, text in enumerate(texts):
            ent, T = pos_entropy(model, device, text)
            ents.append(ent)
            if (j + 1) % 8 == 0:
                print(f"  [{device}] {idx0 + j + 1}/{len(trajs)} done", flush=True)
        out.extend(ents)

    results = []
    ent_s_all, ent_t_all = [], []
    t_s = threading.Thread(target=run_model, args=(ms, 0, [PROMPT_TEMPLATE.format(problem=c) + r for c, r, _, _ in trajs], ent_s_all, 0))
    t_t = threading.Thread(target=run_model, args=(mt, 1, [PROMPT_TEMPLATE.format(problem=c) + r for c, r, _, _ in trajs], ent_t_all, 0))
    t_s.start(); t_t.start()
    t_s.join(); t_t.join()
    print(f"HF forward done: {len(ent_s_all)} student / {len(ent_t_all)} teacher", flush=True)

    for (content, resp, gt, correct), ent_s, ent_t in zip(trajs, ent_s_all, ent_t_all):
        tag = "correct" if correct else "wrong"
        T_s = ent_s.shape[0]
        third = T_s // 3
        results.append({
            "type": tag,
            "gt": gt,
            "H_S_mean": ent_s.mean().item(),
            "H_T_mean": ent_t.mean().item(),
            "H_S_early": ent_s[:third].mean().item(),
            "H_S_mid": ent_s[third:2*third].mean().item(),
            "H_S_late": ent_s[2*third:].mean().item(),
            "H_T_early": ent_t[:third].mean().item(),
            "H_T_mid": ent_t[third:2*third].mean().item(),
            "H_T_late": ent_t[2*third:].mean().item(),
            "resp_len": len(resp),
        })

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"WROTE {out_path} ({len(results)} rows)")

    # ---------- 3. 汇总 ----------
    import statistics
    for typ in ("correct", "wrong"):
        rs = [r for r in results if r["type"] == typ]
        if not rs:
            continue
        hs = statistics.mean(r["H_S_mean"] for r in rs)
        ht = statistics.mean(r["H_T_mean"] for r in rs)
        print(f"[{typ}] n={len(rs)} H_S={hs:.4f} H_T={ht:.4f} Δ(H_T-H_S)={ht-hs:+.4f} "
              f"| late: H_S={statistics.mean(r['H_S_late'] for r in rs):.4f} "
              f"H_T={statistics.mean(r['H_T_late'] for r in rs):.4f}")


if __name__ == "__main__":
    main()
