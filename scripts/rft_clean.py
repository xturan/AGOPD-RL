"""RFT 轨迹清洗:correct 轨迹截断到 GT 答案出现处(去掉尾部废话/重复/临时答案后文)。
输入: rft_survey 输出 jsonl(含 correct_texts)
输出: SFT 数据 jsonl(prompt 直接使用,无模板二次包装)
用法: python rft_clean.py --input <dir or files> --output <sft.jsonl> [--max-per-question 4]
"""
import argparse, glob, json, re
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="glob pattern of rft jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--min-len", type=int, default=80)
    return p.parse_args()


def cut_at_gt(text: str, gt: str) -> str:
    """截断到 GT 答案出现处(含)。优先级:
    1. 'Answer: {gt}' 行尾边界
    2. boxed{{gt}}
    3. 末尾独立 {gt}(行/段边界)
    fallback: 最后一个 Answer 出现处
    """
    gt_esc = re.escape(gt.strip())
    patterns = [
        rf"Answer:\s*\$?{gt_esc}\$?(?=\s*$|\s*[\\`\s]|\)|\])",
        rf"boxed\{{{gt_esc}\}}",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.M)
        if m:
            return text[: m.end()]
    # fallback: 最后一个 Answer:
    ans = [m for m in re.finditer(r"Answer:", text)]
    if ans:
        return text[: ans[-1].end() + 60]
    return text


def main():
    args = parse_args()
    files = sorted(glob.glob(args.input))
    print(f"input files: {len(files)}")
    out_rows = []
    stats = {"ok": 0, "gt_cut": 0, "fallback_last_answer": 0, "too_short": 0}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if not r.get("correct_texts"):
                continue
            for t in r["correct_texts"]:
                cut = cut_at_gt(t, r["gt"])
                if len(cut) < args.min_len:
                    stats["too_short"] += 1
                    continue
                if cut != t:
                    stats["gt_cut"] += 1
                stats["ok"] += 1
                out_rows.append({"idx": r["idx"], "prompt": r["content"],
                                 "ground_truth": r["gt"], "output": cut})
    print(f"stats: {stats}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fo:
        for r in out_rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"WROTE {out}: {len(out_rows)} cleaned trajectories")


if __name__ == "__main__":
    main()
