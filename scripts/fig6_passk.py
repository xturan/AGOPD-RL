#!/usr/bin/env python3
"""Figure 6  ·  Pass@k over a fixed 128-prompt subset (chapter 5.3).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

def main() -> None:
    passk = json.loads((fs.OUT / "passk_five_methods.json").read_text())
    ks = passk["k"]
    keymap = {"Base": "Base", "GRPO-180": "GRPO", "OPD": "OPD",
              "AGOPD": "AGOPD", "RL+OPD": "RL+OPD"}
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for name, ys in passk["values"].items():
        mname = keymap.get(name, name)
        st = fs.METHOD.get(mname)
        if st is None:
            st = dict(color=fs.NEUTRAL, ls="-", lw=1.3, marker="s", ms=3.6)
        ax.plot(ks, ys, marker=st["marker"], ms=st["ms"], lw=st["lw"], ls=st["ls"],
                color=st["color"], label=mname)
    ax.set_xticks(ks)
    ax.set_ylim(0.34, 0.72); ax.set_yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ax.set_xlabel("k (samples per question)")
    ax.set_ylabel("Pass@k")
    fs.title(ax, "Pass@k across methods")
    ax.legend(frameon=False, fontsize=7.6, ncol=3, loc="upper left", columnspacing=1.0)
    fs.save(fig, "fig6_passk_redesigned")

if __name__ == "__main__":
    main()
