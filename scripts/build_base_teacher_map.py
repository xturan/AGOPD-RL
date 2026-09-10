#!/usr/bin/env python3
"""Join the fixed Base and teacher remaps and render Figure 2."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
BASE = OUT / "remap_base_all.json"
TEACHER = OUT / "remap_teacher_all.json"
ORDER = ["hard(p=0)", "p<.25", ".25-.5", ".5-.75", "easy(>.75)"]
LABELS = ["Hard", "Low", "Mid", "High", "Easy"]


def main() -> None:
    base_rows = {row["idx"]: row for row in json.loads(BASE.read_text())}
    teacher_rows = {row["idx"]: row for row in json.loads(TEACHER.read_text())}
    if set(base_rows) != set(teacher_rows):
        raise RuntimeError("Base and teacher remaps are not matched by prompt id")

    rows = []
    grouped = defaultdict(list)
    for idx, base in base_rows.items():
        teacher = teacher_rows[idx]
        if base["bucket"] != teacher["bucket"]:
            raise RuntimeError(f"bucket mismatch for {idx}")
        row = {
            "idx": idx,
            "bucket": base["bucket"],
            "base_p_hat8": base["p_hat8"],
            "teacher_p_hat8": teacher["p_hat8"],
            "margin_pp": 100 * (teacher["p_hat8"] - base["p_hat8"]),
        }
        rows.append(row)
        grouped[row["bucket"]].append(row)
    rows.sort(key=lambda r: (ORDER.index(r["bucket"]), r["idx"]))
    summary = []
    for bucket in ORDER:
        items = grouped[bucket]
        summary.append(
            {
                "bucket": bucket,
                "n_prompts": len(items),
                "base_accuracy": sum(r["base_p_hat8"] for r in items) / len(items),
                "teacher_accuracy": sum(r["teacher_p_hat8"] for r in items) / len(items),
                "margin_pp": sum(r["margin_pp"] for r in items) / len(items),
            }
        )
    payload = {
        "protocol": {
            "prompts_per_bucket": 60,
            "samples_per_prompt": 8,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 42,
            "student": "Qwen3-1.7B Base",
            "teacher": "Qwen3-4B-GRPO fixed checkpoint",
        },
        "summary": summary,
        "rows": rows,
    }
    (OUT / "teacher_base_capability_map_corrected.json").write_text(json.dumps(payload, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3e4547",
            "axes.linewidth": 0.7,
            "axes.labelcolor": "#3e4547",
            "xtick.color": "#3e4547",
            "ytick.color": "#3e4547",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    base = np.array([x["base_accuracy"] for x in summary]) * 100
    teacher = np.array([x["teacher_accuracy"] for x in summary]) * 100
    margin = np.array([x["margin_pp"] for x in summary])
    x = np.arange(len(LABELS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 3.55))
    ax.bar(x - width / 2, base, width, color="#9fb9b7", label="Student: Qwen3-1.7B Base")
    ax.bar(x + width / 2, teacher, width, color="#355f54", label="Teacher: Qwen3-4B-GRPO")
    for i, m in enumerate(margin):
        ax.text(i, max(base[i], teacher[i]) + 3, f"{m:+.1f} pp", ha="center", fontsize=7.8, color="#9c4b50")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Base–teacher capability map", loc="center", fontsize=8.8, fontweight="normal")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper left")
    ax.text(0, -0.18, "60 prompts per bin; n=8 samples per prompt; margin = teacher − Base", transform=ax.transAxes, fontsize=7.1, color="#6f7779")
    fig.savefig(OUT / "fig2_teacher_base_map.png", dpi=240, facecolor="white")
    fig.savefig(OUT / "fig2_teacher_base_map.svg", facecolor="white")
    fig.savefig(OUT / "fig2_teacher_base_map.pdf", facecolor="white")
    plt.close(fig)
    print("created", OUT / "fig2_teacher_base_map.png")
    print("summary", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
