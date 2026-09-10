#!/usr/bin/env python3
"""Render the CRPTR counterfactual teacher-state rescue figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "figF1_cptr_rescue.png"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#596466",
            "axes.labelcolor": "#334042",
            "xtick.color": "#334042",
            "ytick.color": "#334042",
        }
    )
    labels = ["Root", "Random filler", "Wrong prefix", "Correct prefix"]
    values = [65.92, 64.01, 56.85, 72.61]
    colors = ["#155b66", "#9fb9b7", "#9c4b50", "#155b66"]
    fig, ax = plt.subplots(figsize=(7.4, 3.25))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.2f}%", ha="center", fontsize=8.5)
    ax.set_ylabel("Teacher accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Counterfactual teacher rescue under matched student states", fontsize=10.5, pad=10)
    ax.text(0.99, -0.18, "157 prompts; four teacher samples per condition; matched prefix length", transform=ax.transAxes, ha="right", color="#596466", fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("created", OUT)


if __name__ == "__main__":
    main()
