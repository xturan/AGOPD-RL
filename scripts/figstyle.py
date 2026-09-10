#!/usr/bin/env python3
"""Shared visual style for every AGOPD paper figure (single source of truth).

Visual definition (locked):
  - Palette  : primary #155b66 | auxiliary #9fb9b7 | negative #9c4b50 | neutral grey #6f7779
  - Red is reserved for genuinely negative marks (degraded states, negative deltas,
    harmful region). Non-red shades encode degree light->dark within one hue family.
  - No grid. Top/right spines off. Axes labelled explicitly.
  - Element spacing: no interleaving/overlap; generous margins; titles padded.
Method identity map (shared by fig4(b) / fig6 / D2 references):
  Base / GRPO / OPD / RL+OPD / AGOPD — see METHOD below (hue + style, never color alone).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"

# ---- palette ----
PRIMARY      = "#155b66"   # deep teal (main)
PRIMARY_LIGHT= "#9fb9b7"   # light teal (aux)
PRIMARY_MID  = "#5d8f8b"   # mid teal (ramp step, same hue)
NEG          = "#9c4b50"   # red  -> bad / degraded / harmful
NEUTRAL      = "#6f7779"   # grey -> controls / reference lines / neutral arms
INK          = "#17191a"
INK2         = "#3e4547"
MUTED        = "#6f7779"
RULE         = "#d9dddc"

# ---- method identity: hue + line style (identity never color-alone) ----
METHOD = {
    "Base":   dict(color=PRIMARY_LIGHT, ls="-",  lw=1.3, marker="o", ms=4.0),
    "GRPO":   dict(color=PRIMARY,       ls="--", lw=1.4, marker="o", ms=4.0),
    "OPD":    dict(color=NEUTRAL,       ls="-",  lw=1.3, marker="s", ms=3.6),
    "RL+OPD": dict(color=NEG,           ls="-",  lw=1.5, marker="^", ms=4.2),
    "AGOPD":  dict(color=PRIMARY,       ls="-",  lw=2.2, marker="o", ms=4.6),
}
METHOD_ORDER = ["Base", "GRPO", "OPD", "RL+OPD", "AGOPD"]

def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

def save(fig, stem: str, dpi: int = 200) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print("created", OUT / f"{stem}.png")

def title(ax, text: str) -> None:
    ax.set_title(text, loc="center", fontsize=9.0, fontweight="normal", pad=9)

def panel_letter(ax, tag: str) -> None:
    ax.text(-0.09, 1.05, tag, transform=ax.transAxes, fontsize=9.0,
            color=INK2, va="bottom", ha="left")

def note(ax, text: str, y: float = -0.34) -> None:
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.0, color=MUTED, va="top")
