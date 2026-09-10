#!/usr/bin/env python3
"""Figure 6 · schematic (not data). Two operating regimes across difficulty.
x: difficulty increases rightward; y: relative emphasis. Red lobe = hard /
low-success boundary region; green lobe = easier / comfort region. Illustrative
only; no scatter marks, margins kept so nothing is clipped."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

RED='#9c4b50'; GREEN='#6aa84f'
def main():
    x=np.linspace(0,1,400)
    def lobe(xc,sd): return 0.85*np.exp(-((x-xc)/sd)**2)
    red=lobe(0.78,0.17); green=lobe(0.34,0.19)
    fig,ax=plt.subplots(figsize=(6.2,3.3))
    ax.fill_between(x,red,color=RED,alpha=0.30)
    ax.plot(x,red,color=RED,lw=2.0)
    ax.fill_between(x,green,color=GREEN,alpha=0.30)
    ax.plot(x,green,color=GREEN,lw=2.0)
    ax.set_xlim(-0.02,1.10); ax.set_ylim(-0.05,1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('Difficulty  →', fontsize=9.5)
    ax.set_ylabel('Relative emphasis', fontsize=9.5)
    fs.save(fig,'fig6_difficulty_schematic')
if __name__=='__main__':
    main()
