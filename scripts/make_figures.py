"""Generate report figures (ICLR paper style, all-ASCII figure text).
Figure captions stay in HTML (Chinese). Text inside figures is English.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

ACCENT = "#3B5B7E"
ACCENT2 = "#B9C4CF"
ACCENT3 = "#D6DFE8"
GREEN = "#6a8f5a"
RED = "#C44E52"
FIGDIR = "reports/figures/"


def load(name):
    return {int(k): v for k, v in json.load(open(f"{FIGDIR}/metrics_{name}.json")).items()}


def smooth(xs, ys, w=5):
    ys = np.array(ys, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(ys, kernel, mode="same")


def series(d, key):
    xs = sorted(d)
    return [x for x in xs], [d[x][key] if d[x].get(key) is not None else np.nan for x in xs]


def line_ax(ax, d, key, label, color=ACCENT, smooth_w=5, lw=1.6, ylabel=None):
    xs, ys = series(d, key)
    ax.plot(xs, smooth(xs, ys, smooth_w), color=color, lw=lw, label=label, alpha=0.85)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("step")


# ---- Fig 1: Capability frontier (full census, 16164 q, corrected) ----
fig, ax = plt.subplots(figsize=(7, 2.8))
cats = ["hard\n(p_hat = 0)", "frontier\n(0.125-0.625)", "easy\n(>= 0.75)"]
vals = [89.2, 10.5, 0.3]
cols = [RED, ACCENT, ACCENT2]
bars = ax.bar(cats, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("share of prompts (%)")
ax.set_ylim(0, 100)
ax.set_title("1.7B base capability over 16,164 train prompts (n=8 full census)", fontsize=10.5)
ax.grid(axis="y", alpha=0.5)
fig.savefig(f"{FIGDIR}/fig1_frontier.png"); plt.close(fig)

# ---- Fig 2: GRPO full-param curves ----
g = load("grpo_fullparam")
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
line_ax(axes[0,0], g, "score", "score", ylabel="online score")
axes[0,0].axhline(0.39, color=ACCENT2, ls="--", lw=0.8)
axes[0,0].text(5, 0.40, "bucket census mean ~0.39", fontsize=8.5, color="#666")
line_ax(axes[0,1], g, "entropy", "entropy", ylabel="entropy")
line_ax(axes[0,2], g, "pg", "pg_loss", ylabel="pg_loss")
line_ax(axes[1,0], g, "grad", "grad_norm", ylabel="grad_norm")
line_ax(axes[1,1], g, "rlen", "response len", ylabel="len (tokens)")
line_ax(axes[1,2], g, "cr", "truncation", ylabel="trunc rate")
fig.suptitle("GRPO (full-param lr1e-6 + KL0.05, comfort bucket, 184 steps)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f"{FIGDIR}/fig2_grpo_curves.png"); plt.close(fig)

# ---- Fig 3: EGR chain ----
fig, ax = plt.subplots(figsize=(7, 3))
stages = ["full\n(n4)", "frontier\n(n4)", "comfort\n(n4)", "comfort\n(n8)"]
egr = [20, 52.5, 60.8, 80.0]
ax.plot(range(4), egr, "o-", color=ACCENT, lw=1.8, ms=7)
for i, v in enumerate(egr):
    ax.text(i, v + 4, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=9.5)
ax.set_ylabel("EGR (%)"); ax.set_ylim(0, 100)
ax.set_title("Effective group rate along the fix chain (measured)", fontsize=10.5)
fig.savefig(f"{FIGDIR}/fig3_egr.png"); plt.close(fig)

# ---- Fig 4: update channel ----
l1 = load("lora_lr1e4"); l2 = load("lora_lr3e4")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
line_ax(axes[0], l1, "grad", "LoRA r16 lr1e-4", color=ACCENT2)
line_ax(axes[0], l2, "grad", "LoRA r16 lr3e-4", color=ACCENT3)
line_ax(axes[0], g, "grad", "full-param lr1e-6", color=ACCENT)
axes[0].set_title("grad_norm: LoRA ~0.03 vs full-param 0.5-1.4", fontsize=10)
line_ax(axes[1], l1, "pg", "LoRA r16 lr1e-4", color=ACCENT2)
line_ax(axes[1], l2, "pg", "LoRA r16 lr3e-4", color=ACCENT3)
line_ax(axes[1], g, "pg", "full-param", color=ACCENT)
axes[1].set_title("pg_loss: 3x lr does not move LoRA", fontsize=10)
for a in axes: a.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig4_channel.png"); plt.close(fig)

# ---- Fig 5: collapse vs stable ----
c5 = load("fp_collapse_lr1e5")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
line_ax(axes[0], c5, "rlen", "lr1e-5 (collapse)", color=RED)
line_ax(axes[0], g, "rlen", "lr1e-6 + KL0.05 (stable)", color=ACCENT)
axes[0].axhline(2048, color="#999", ls=":", lw=0.8)
axes[0].text(1, 1990, "2048 cap", fontsize=8, color="#666")
axes[0].set_title("Response length: drift-verbosity-truncation loop", fontsize=10)
line_ax(axes[1], c5, "score", "lr1e-5", color=RED)
line_ax(axes[1], g, "score", "lr1e-6 + KL0.05", color=ACCENT)
axes[1].set_title("score: signal dies after collapse", fontsize=10)
for a in axes: a.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig5_collapse.png"); plt.close(fig)

# ---- Fig 6: four methods ----
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for name, label, c in [("grpo_fullparam", "GRPO", ACCENT),
                       ("agopd", "AGOPD (gate)", GREEN),
                       ("opd", "OPD (pure)", ACCENT2),
                       ("rlopd", "RL+OPD (no gate)", RED)]:
    d = load(name)
    xs, ys = series(d, "score")
    axes[0].plot(xs, smooth(xs, ys, 7), color=c, lw=1.6, label=label, alpha=0.9)
axes[0].set_title("Online score, four methods (same comfort config)", fontsize=10)
axes[0].legend(fontsize=9, frameon=False)
for name, label, c in [("grpo_fullparam", "GRPO", ACCENT),
                       ("agopd", "AGOPD (gate)", GREEN),
                       ("rlopd", "RL+OPD (no gate)", RED)]:
    d = load(name)
    xs, ys = series(d, "entropy")
    axes[1].plot(xs, smooth(xs, ys, 7), color=c, lw=1.6, label=label, alpha=0.9)
axes[1].set_title("Entropy: gate vs no-gate distribution health", fontsize=10)
axes[1].legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig6_fourmethods.png"); plt.close(fig)

# ---- Fig 7: truncation in-dist vs ood ----
fig, ax = plt.subplots(figsize=(7, 3))
cats = ["train rollout\n(comfort)", "fixed eval\n(val, OOD)"]
vals = [15.6, 94.0]
cols = [ACCENT, RED]
bars = ax.bar(cats, vals, color=cols, width=0.45)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.0f}%", ha="center")
ax.set_ylabel("share hitting 2048 cap (%)")
ax.set_ylim(0, 110)
ax.set_title("Truncation: clean in-distribution, degenerate OOD", fontsize=10.5)
ax.grid(axis="y", alpha=0.5)
fig.savefig(f"{FIGDIR}/fig7_trunc.png"); plt.close(fig)

# ---- Fig 8: teacher screen + gate ----
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
teachers = ["4B-GRPO", "Qwen3-8B", "OpenMath-7B"]
pt = [0.152, 0.078, 0.084]
bars = axes[0].bar(teachers, pt, color=[ACCENT, ACCENT2, ACCENT3], width=0.5)
for b, v in zip(bars, pt):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=9.5)
axes[0].set_ylabel("mean p_T (comfort bucket)")
axes[0].set_title("Teacher screening: distribution specialization > size", fontsize=10)
axes[0].grid(axis="y", alpha=0.5)
cats = ["all comfort", "gate-passed"]
vals = [32, 72]
bars = axes[1].bar(cats, vals, color=[ACCENT2, ACCENT], width=0.45)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x()+b.get_width()/2, v+2, f"{v}%", ha="center")
axes[1].set_ylim(0, 100)
axes[1].set_ylabel("teacher competence")
axes[1].set_title("Gate enrichment (3x)", fontsize=10)
axes[1].grid(axis="y", alpha=0.5)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig8_teacher.png"); plt.close(fig)

# ---- Fig 9: learning signal ----
fig, ax = plt.subplots(figsize=(7, 3.2))
xs_w = [8, 23, 38, 53, 68, 83]
sp = [-0.037, -0.013, -0.005, -0.022, 0.035, 0.058]
ax.plot(xs_w, sp, "o-", color=ACCENT, lw=1.8, ms=6)
ax.axhline(0, color="#999", ls="--", lw=0.8)
for x, y in zip(xs_w, sp):
    ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
ax.set_xlabel("step"); ax.set_ylabel("score - p_hat (difficulty-calibrated)")
ax.set_title("Real learning signal: net gain after removing bucket difficulty", fontsize=10.5)
ax.grid(alpha=0.4)
fig.savefig(f"{FIGDIR}/fig9_signal.png"); plt.close(fig)

print("done")
