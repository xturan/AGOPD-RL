#!/usr/bin/env python3
"""Figure 2  ·  teacher supervision: problem-root ability -> state-level recoverability.

(a) teacher vs Base accuracy over five competence bins (chat-template remap)
(b) teacher rescue vs erroneous-prefix depth (teacher_rescue_summary, 227 prompts)
(c) length-matched control: root / random-filler / wrong / correct prefix
    (random_control_summary, 157 matched prompts)

Red = erroneous-semantics state only; grey = length-only control. No grid.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()
OUT = fs.OUT
CPTR = fs.ROOT / "reports" / "cptr"

def fig2a() -> None:
    d = json.loads((OUT / "teacher_base_capability_map_corrected.json").read_text())
    labels = ["Hard", "Low", "Mid", "High", "Easy"]
    base = np.array([s["base_accuracy"] for s in d["summary"]]) * 100
    teacher = np.array([s["teacher_accuracy"] for s in d["summary"]]) * 100
    margin = teacher - base
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(x - w/2, base, w, color=fs.PRIMARY_LIGHT, label="Student: Qwen3-1.7B Base")
    ax.bar(x + w/2, teacher, w, color=fs.PRIMARY, label="Teacher: Qwen3-4B-GRPO")
    for i, m in enumerate(margin):
        ax.text(i, max(base[i], teacher[i]) + 2.6, f"{m:+.1f} pp", ha="center",
                fontsize=7.8, color=fs.INK2)
    ax.set_xticks(x, labels); ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left", handlelength=1.5, columnspacing=1.2)
    fs.save(fig, "fig2a_teacher_advantage")

def fig2b() -> None:
    s = json.loads((CPTR / "teacher_rescue_summary.json").read_text())
    keys = ["root", "path_25", "path_50", "path_75"]
    values = [100.0 * s[k]["mean"] for k in keys]
    xl = ["Root", "Erroneous 25%", "Erroneous 50%", "Erroneous 75%"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    ax.plot(x, values, color=fs.PRIMARY, marker="o", ms=5.0, lw=1.8)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=7.6, color=fs.INK2)
    ax.set_xticks(x, xl); ax.set_ylim(28, 66); ax.set_yticks(np.arange(30, 66, 5))
    ax.set_ylabel("Teacher rescue (%)")
    ax.set_xlabel("Erroneous student prefix depth")
    fs.save(fig, "fig2b_rescue_depth_raw")

def fig2c() -> None:
    s = json.loads((CPTR / "random_control_summary.json").read_text())
    keys = ["root", "random_filler_matched", "wrong_path_matched", "correct_path_matched"]
    xl = ["Root", "Random filler", "Wrong prefix", "Correct prefix"]
    values = [100.0 * s[k]["mean"] for k in keys]
    colors = [fs.PRIMARY_LIGHT, fs.NEUTRAL, fs.NEG, fs.PRIMARY]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    bars = ax.bar(x, values, width=0.56, color=colors, edgecolor="none")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.8, f"{v:.1f}%", ha="center",
                fontsize=7.6, color=fs.INK2)
    ax.set_xticks(x, xl); ax.set_ylim(0, 90)
    ax.set_ylabel("Teacher accuracy (%)")
    fs.save(fig, "fig2c_matched_control_raw")

def main() -> None:
    fig2a(); fig2b(); fig2c()
    from figcrop import align_row
    align_row([str(fs.OUT/"fig2b_rescue_depth_raw.png"), str(fs.OUT/"fig2c_matched_control_raw.png")],
              [str(fs.OUT/"fig2b_rescue_depth.png"), str(fs.OUT/"fig2c_matched_control.png")])
    for f in (fs.OUT/"fig2b_rescue_depth_raw.png", fs.OUT/"fig2c_matched_control_raw.png"):
        if f.exists(): f.unlink()

if __name__ == "__main__":
    main()
