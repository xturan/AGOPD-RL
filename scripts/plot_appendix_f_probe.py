#!/usr/bin/env python3
"""Publication-style Appendix-F state-probe panels.

Figure contract:
  - claim: a verified wrong Student state reduces subsequent Teacher rescue;
  - primary evidence: prompt-level paired rescue rates;
  - secondary evidence: top-16 conditional entropy at the same state anchors;
  - panel labels, figure number, title and legend explanation live in HTML;
  - plots contain no in-image title and preserve every prompt-level point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs

fs.setup()

STATE_ORDER = ["root", "error_onset", "post_error_64"]
STATE_LABELS = ["Root", "First error", "+64 tokens"]
KIND_WRONG = "student_wrong"
KIND_CORRECT = "student_correct_matched"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    return p.parse_args()


def save_publication(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def rows(summary: dict, kind: str, state: str) -> list[dict]:
    return [
        row
        for row in summary["rows"]
        if row["trajectory_kind"] == kind and row["state"] == state
    ]


def panel(
    summary: dict,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    colors: tuple[str, str],
    output: Path,
    show_legend: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    x = list(range(len(STATE_ORDER)))
    for kind, color, linestyle, label in (
        (KIND_WRONG, colors[0], "-", "wrong trajectory"),
        (KIND_CORRECT, colors[1], "--", "matched correct trajectory"),
    ):
        means = []
        for xi, state in zip(x, STATE_ORDER):
            values = [float(row[metric]) for row in rows(summary, kind, state)]
            means.append(sum(values) / len(values))
            offsets = [-0.045, -0.022, 0.0, 0.022, 0.045]
            for offset, value in zip(offsets, values):
                ax.plot(
                    xi + offset,
                    value,
                    marker="o",
                    markersize=3.2,
                    color=color,
                    alpha=0.42,
                    markeredgewidth=0,
                    linestyle="none",
                    zorder=2,
                )
        ax.plot(
            x,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker="o",
            markersize=4.3,
            markeredgewidth=0,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, STATE_LABELS)
    ax.set_xlabel("State position", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=7.7, length=3, width=0.65)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    if show_legend:
        ax.legend(
            frameon=False,
            fontsize=7.0,
            handlelength=1.7,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
        )
    save_publication(fig, output)


def composite(summary: dict, output: Path) -> None:
    """Draw one fixed-size 2x2 canvas so all panels align in HTML."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    configs = [
        ("teacher_correct_count", "Teacher correct continuations (of 8)", (0, 8), (fs.PRIMARY, fs.PRIMARY_LIGHT), "(a)"),
        ("student_correct_count", "Student correct continuations (of 8)", (0, 8), (fs.NEG, "#d6a4a7"), "(b)"),
        ("teacher_entropy", "Teacher top-16 entropy (nat)", (0, 0.75), (fs.PRIMARY, fs.PRIMARY_LIGHT), "(c)"),
        ("student_entropy", "Student top-16 entropy (nat)", (0, 0.75), (fs.NEG, "#d6a4a7"), "(d)"),
    ]
    legend_handles = None
    legend_labels = None
    x = list(range(len(STATE_ORDER)))
    for ax, (metric, ylabel, ylim, colors, letter) in zip(axes.flat, configs):
        for kind, color, linestyle, label in (
            (KIND_WRONG, colors[0], "-", "wrong trajectory"),
            (KIND_CORRECT, colors[1], "--", "matched correct trajectory"),
        ):
            means = []
            for xi, state in zip(x, STATE_ORDER):
                values = [float(row[metric]) for row in rows(summary, kind, state)]
                means.append(sum(values) / len(values))
                offsets = [-0.045, -0.022, 0.0, 0.022, 0.045]
                for offset, value in zip(offsets, values):
                    ax.plot(xi + offset, value, marker="o", markersize=2.6, color=color, alpha=0.42, markeredgewidth=0, linestyle="none", zorder=2)
            line, = ax.plot(x, means, color=color, linestyle=linestyle, linewidth=1.55, marker="o", markersize=3.9, markeredgewidth=0, label=label, zorder=3)
            if legend_handles is None:
                legend_handles = [line]
                legend_labels = [label]
            elif len(legend_handles) == 1:
                legend_handles.append(line)
                legend_labels.append(label)
        ax.text(0.01, 0.98, letter, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=fs.INK2)
        ax.set_xticks(x, STATE_LABELS)
        ax.set_xlabel("State position", fontsize=8.0)
        ax.set_ylabel(ylabel, fontsize=8.0)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7.1, length=2.5, width=0.6)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.98, wspace=0.40, hspace=0.43)
    fig.legend(legend_handles, legend_labels, frameon=False, fontsize=7.3, handlelength=1.7, ncol=2, loc="lower center", bbox_to_anchor=(0.50, 0.025))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    prefix = a.output_prefix
    panel(
        summary,
        "teacher_correct_count",
        "Teacher correct continuations (of 8)",
        (0, 8),
        (fs.PRIMARY, fs.PRIMARY_LIGHT),
        prefix.with_name(prefix.name + "_teacher_rescue.png"),
        show_legend=True,
    )
    panel(
        summary,
        "student_correct_count",
        "Student correct continuations (of 8)",
        (0, 8),
        (fs.NEG, "#d6a4a7"),
        prefix.with_name(prefix.name + "_student_rescue.png"),
        show_legend=False,
    )
    panel(
        summary,
        "teacher_entropy",
        "Teacher top-16 entropy (nat)",
        (0, 0.75),
        (fs.PRIMARY, fs.PRIMARY_LIGHT),
        prefix.with_name(prefix.name + "_teacher_entropy.png"),
        show_legend=False,
    )
    panel(
        summary,
        "student_entropy",
        "Student top-16 entropy (nat)",
        (0, 0.75),
        (fs.NEG, "#d6a4a7"),
        prefix.with_name(prefix.name + "_student_entropy.png"),
        show_legend=False,
    )
    composite(summary, prefix.with_name(prefix.name + "_composite.png"))
    print(f"figure_prefix={prefix}", flush=True)


if __name__ == "__main__":
    main()
