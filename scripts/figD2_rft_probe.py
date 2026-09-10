#!/usr/bin/env python3
"""Figure D2  ·  RFT-SFT checkpoint probe at 4,096-token budget (appendix D).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()
plt.rcParams["font.size"]=7.2

def main() -> None:
    steps = np.array([1000, 2000, 3000, 4000, 5000])
    accuracy = np.array([30.1, 31.1, 33.5, 32.7, 31.0])
    reference = 30.37
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    ax.plot(steps, accuracy, color=fs.PRIMARY, marker="o", ms=4.2, lw=1.6,
            label="RFT-SFT checkpoint")
    ax.fill_between(steps, accuracy, reference, where=accuracy >= reference,
                    color=fs.PRIMARY_LIGHT, alpha=0.25, linewidth=0)
    ax.axhline(reference, color=fs.NEUTRAL, ls=(0, (4, 3)), lw=1.2,
               label="180-step GRPO reference (Table D1)")
    for x, y in zip(steps, accuracy):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.8, color=fs.INK2)
    ax.annotate("probe peak", (3000, 33.5), textcoords="offset points", xytext=(16, 12),
                arrowprops={"arrowstyle": "-", "color": fs.MUTED, "lw": 0.7},
                fontsize=6.5, color=fs.INK2)
    ax.set_xlim(750, 5250); ax.set_ylim(28.5, 35.2)
    ax.set_xticks(steps)
    ax.set_xlabel("RFT-SFT checkpoint step")
    ax.set_ylabel("Accuracy @ 4,096 (%)")
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    fs.save(fig, "figD2_rft_probe_redesigned")

if __name__ == "__main__":
    main()
