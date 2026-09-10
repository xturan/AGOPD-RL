#!/usr/bin/env python3
"""Figure 1 · capability census + effective group rate.

Clean images only (no in-image titles, no bottom notes, no added markers).
Panel label + figure title live in the HTML under each image.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

COUNTS = np.array([7905, 2097, 1383, 985, 928, 809, 694, 684, 679], dtype=float)
N = int(COUNTS.sum())
PCT = COUNTS / N * 100.0

def panel_a() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    k = np.arange(9)
    ax.bar(k, PCT, width=0.62, color=fs.PRIMARY, edgecolor="none")
    ax.annotate(f"{PCT[0]:.1f}%", (0, PCT[0]), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=8.2, color=fs.INK2)
    ax.annotate(f"{PCT[1]:.1f}%", (1, PCT[1]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8.2, color=fs.INK2)
    ax.set_xticks(k, [str(int(x)) for x in k])
    ax.set_xlabel("Correct samples per prompt (of n = 8)")
    ax.set_ylabel("Share of prompts (%)")
    ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20))
    fs.save(fig, "fig1a_census_raw")

def panel_b() -> None:
    Gs = [2, 6, 16]
    ramp = [fs.PRIMARY_LIGHT, fs.PRIMARY_MID, fs.PRIMARY]
    p = np.linspace(0.0, 1.0, 401)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for G, col in zip(Gs, ramp):
        egr = 1.0 - (1.0 - p) ** G - p ** G
        ax.plot(p, egr, color=col, lw=2.0, label=f"G = {G}")
    ax.set_xlabel("Single-attempt success probability $p$")
    ax.set_ylabel("Effective group rate")
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_xlim(0, 1)
    # legend outside the axes -> never overlaps the curves
    ax.legend(frameon=False, fontsize=8.0, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fs.save(fig, "fig1b_egr_raw")

if __name__ == "__main__":
    panel_a(); panel_b()
    from figcrop import align_row
    from pathlib import Path as _P
    align_row([str(fs.OUT/_P("fig1a_census_raw.png")), str(fs.OUT/_P("fig1b_egr_raw.png"))],
              [str(fs.OUT/_P("fig1a_census.png")), str(fs.OUT/_P("fig1b_egr.png"))])
    (fs.OUT/_P("fig1a_census_raw.png")).unlink()
    (fs.OUT/_P("fig1b_egr_raw.png")).unlink()
