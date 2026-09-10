#!/usr/bin/env python3
"""Scheme A schematic (non-data): layer-on-layer probability sieves from the full
prompt set to the training data.
Top: p-hat axis (difficulty). A broad mass shows all prompts; two nested
translucent bands mark (1) group-signal region (per-prompt success such that
within-group contrast is possible) and (2) learnable / comfortable region;
their intersection (darker) is kept. Below: funnel to the training pool.
Illustrative only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

def main():
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.linspace(0, 1, 500)
    mass = 0.16 + 0.9*np.exp(-((x-0.5)/0.34)**2)
    ax.plot(x, mass*0.75, color=fs.PRIMARY_LIGHT, lw=1.6)
    ax.fill_between(x, 0, mass*0.75, color=fs.PRIMARY_LIGHT, alpha=0.22)
    # sieve 1: group-signal region  (wide)
    ax.axvspan(0.18, 0.85, color=fs.PRIMARY, alpha=0.10)
    # sieve 2: learnable/comfort (nested, narrower)
    ax.axvspan(0.32, 0.72, color=fs.PRIMARY, alpha=0.18)
    # intersection highlight
    ax.axvspan(0.32, 0.72, color=fs.PRIMARY, alpha=0.0)
    # labels of thresholds / sieves
    ax.annotate('Sieve 1: group-signal\n(p̂ allows within-group contrast)', xy=(0.28,0.62),
                xytext=(0.02,0.86), fontsize=7.4, color=fs.INK2)
    ax.annotate('Sieve 2: learnable region\n(comfort)', xy=(0.5,0.66),
                xytext=(0.55,0.90), fontsize=7.4, color=fs.INK2)
    # arrows narrowing to kept data
    ax.annotate('', xy=(0.5,0.30), xytext=(0.5,0.52),
                arrowprops=dict(arrowstyle='-|>', color=fs.PRIMARY, lw=1.6))
    ax.text(0.52,0.40,'keep intersection',fontsize=7.2,color=fs.PRIMARY)
    # funnel boxes at bottom
    for yy,w in [(0.18,0.72),(0.06,0.40)]:
        ax.add_patch(plt.Rectangle((0.5-w/2,yy),w,0.09,facecolor=fs.PRIMARY_LIGHT if w>0.5 else fs.PRIMARY,
                                   edgecolor=fs.INK2,lw=0.9))
    ax.text(0.5,0.14,'training pool (comfort)',ha='center',fontsize=8,color=fs.INK)
    ax.text(0.5,0.005,'difficulty  p̂  →',ha='center',fontsize=9,color=fs.INK2)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    fs.save(fig,'fig7_data_sieve_schematic')

if __name__=='__main__':
    main()
