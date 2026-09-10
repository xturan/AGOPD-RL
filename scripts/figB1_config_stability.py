#!/usr/bin/env python3
"""Appendix B real-data figure: full-parameter LR stability.
Three runs (all full-param) differing in lr: 1e-6 (+KL0.05, frozen main),
1e-5 (late collapse), 3e-4 (early collapse). Two panels: online score and
policy entropy vs training step. Data from metrics_*.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()

RUNS=[('metrics_grpo_fullparam.json','lr $10^{-6}$ + KL 0.05 (frozen)',fs.PRIMARY,'-',2.0),
      ('metrics_fp_collapse_lr1e5.json','lr $10^{-5}$',fs.NEG,'--',1.5),
      ('metrics_fp_collapse_lr3e4.json','lr $3\\times10^{-4}$','#b07a30','--',1.5)]
def load(fn):
    d=json.load(open(fs.OUT/fn)); return {int(k):v for k,v in d.items()}
def main():
    data=[(load(fn),lab,col,ls,lw) for fn,lab,col,ls,lw in RUNS]
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),gridspec_kw={'wspace':0.42})
    for ax,key,yl in [(axes[0],'score',(0,0.6)),(axes[1],'entropy',(0,0.35))]:
        for d,lab,col,ls,lw in data:
            steps=sorted(d); y=[d[s][key] for s in steps]
            ax.plot(steps,y,color=col,ls=ls,lw=lw,label=lab)
        ax.set_xlabel('Training step'); ax.set_xlim(0,190)
        ax.set_ylim(*yl)
    axes[0].set_ylabel('Online score')
    axes[1].set_ylabel('Policy entropy')
    axes[0].legend(frameon=False,fontsize=6.8,loc='upper left')
    fs.save(fig,'figB1_config_stability')
if __name__=='__main__':
    main()
