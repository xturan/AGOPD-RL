#!/usr/bin/env python3
"""Figure 4 · difficulty-bucket profile as a six-model radar (chapter 5.2).

Axes: five Base-competence bins (Hard -> Easy). Series: Base / GRPO / OPD /
RL+OPD / AGOPD (figstyle.METHOD) + fixed Teacher (ink, dashed-diamond). Values
are strict-verifier accuracy on the fixed 1,024 prompts, merged per prompt with
the five-method CSV and the teacher per-question file.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

def main() -> None:
    df = pd.read_csv(fs.OUT / "five_methods_per_question.csv")
    with open(fs.ROOT / "reports" / "eval-dapo-v1" / "grpo-4b-50step-ckpt2.jsonl", encoding="utf-8") as _fh:
        tr = pd.DataFrame([json.loads(_l) for _l in _fh if _l.strip()])[["prompt_id", "correct"]]
    df = df.merge(tr, on="prompt_id", how="left")
    order = ["hard(p=0)", ".125-.25", ".25-.5", ".5-.75", "easy(>.75)"]
    labels = ["Hard", "Low", "Mid", "High", "Easy"]
    meth = ["base", "grpo", "opd", "rlopd", "agopd"]
    # value arrays in [0,1]
    vals = {name: np.array([df.loc[df.bucket == b, col].mean() for b in order]) for name, col in
            [("Base", "base"), ("GRPO", "grpo"), ("OPD", "opd"),
             ("RL+OPD", "rlopd"), ("AGOPD", "agopd")]}
    vals["Teacher"] = np.array([df.loc[df.bucket == b, "correct"].mean() for b in order])

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(4.6, 5.0), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles, labels, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color=fs.MUTED, fontsize=7)
    ax.grid(color="#dfe3e2", lw=0.7)
    ax.spines["polar"].set_color("#c9cfcd")

    series = [("Base", fs.METHOD["Base"]), ("GRPO", fs.METHOD["GRPO"]),
              ("OPD", fs.METHOD["OPD"]), ("RL+OPD", fs.METHOD["RL+OPD"]),
              ("AGOPD", fs.METHOD["AGOPD"])]
    series.append(("Teacher", dict(color=fs.INK, ls=(0, (3, 1)), lw=1.4, marker="D", ms=3.6)))
    for name, st in series:
        v = np.r_[vals[name] * 100, vals[name][0] * 100]
        a = angles + [angles[0]]
        ax.plot(a, v, color=st["color"], lw=st["lw"], ls=st["ls"],
                marker=st["marker"], ms=st["ms"], label=name)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.05))
    fs.save(fig, "fig4_difficulty_radar", dpi=170)

if __name__ == "__main__":
    main()
