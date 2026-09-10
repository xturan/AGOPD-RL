#!/usr/bin/env python3
"""Render compact validation figures for the AGOPD paper.

The script uses only committed fixed-set evaluation summaries. It intentionally
does not fabricate the missing multi-sample Pass@k result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SPECS = [
    ("Base", ROOT / "reports/eval-dapo-v1/base-2048.summary.json", ROOT / "reports/eval-dapo-v1/base-4096.summary.json"),
    ("Stage-2B\nGRPO", ROOT / "reports/figures/grpo240_eval2048.summary.json", ROOT / "reports/figures/grpo240_eval4096.summary.json"),
    ("Stage-2B\nAGOPD", ROOT / "reports/figures/agopd240_eval2048.summary.json", ROOT / "reports/figures/agopd240_eval4096.summary.json"),
    ("Stage-2C\nGRPO", ROOT / "reports/figures/stage2c_eval2048.summary.json", ROOT / "reports/figures/stage2c_eval4096.summary.json"),
]

COLORS = ["#8ca5a3", "#155b66", "#b55c5f", "#3b6d8a"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    rows = []
    for name, p2048, p4096 in SPECS:
        a = load(p2048)
        b = load(p4096)
        rows.append(
            {
                "name": name,
                "acc2048": 100 * a["accuracy"],
                "acc4096": 100 * b["accuracy"],
                "complete2048": 100 * (1 - a["truncation_rate"]),
                "complete4096": 100 * (1 - b["truncation_rate"]),
                "parse2048": 100 * a["answer_parse_rate"],
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#70787a",
            "axes.labelcolor": "#3f4749",
            "xtick.color": "#3f4749",
            "ytick.color": "#3f4749",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.15), sharey=False)
    panels = [
        ("acc2048", "Accuracy @ 2048", (20, 34)),
        ("acc4096", "Accuracy @ 4096", (24, 36)),
        ("complete2048", "Completed responses @ 2048", (65, 100)),
    ]
    y = np.arange(len(rows))
    for ax, (key, title, limits) in zip(axes, panels):
        values = [r[key] for r in rows]
        for i, (value, color) in enumerate(zip(values, COLORS)):
            ax.plot(value, i, "o", color=color, ms=7, zorder=3)
            ax.hlines(i, limits[0], value, color="#dfe3e2", lw=1.3, zorder=1)
            ax.text(value + (limits[1] - limits[0]) * 0.025, i, f"{value:.1f}", va="center", fontsize=9)
        ax.set_xlim(*limits)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(title, fontsize=10.5, pad=9)
        ax.grid(axis="x", color="#e7eae9", lw=0.8)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_yticks(y, [r["name"] for r in rows])
        else:
            ax.set_yticks(y, [])
    fig.suptitle("Fixed validation profile across four checkpoints", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_validation_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels = ["accuracy\n@2048", "accuracy\n@4096", "completion\n@2048", "completion\n@4096", "parse\n@2048"]
    metrics = ["acc2048", "acc4096", "complete2048", "complete4096", "parse2048"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.0, 4.75), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="#8b9594", fontsize=8)
    ax.grid(color="#dfe3e2", lw=0.8)
    for row, color in zip(rows, COLORS):
        values = [row[k] for k in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, color=color, lw=1.7, marker="o", ms=3.8, label=row["name"].replace("\n", " "))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "radar_validation_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
