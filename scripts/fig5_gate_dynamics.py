#!/usr/bin/env python3
"""Figure 5 · gating dynamics & distillation scope (AGOPD vs RL+OPD).
Two separate single-panel images, panel id + short title rendered UNDER each
image in the HTML (.fig-panel-label). No in-image letters/titles/notes.
(a) candidate trajectories (negative-advantage) and teacher-competence retention
    conditional on candidates;
(b) masked distillation loss, 7-step moving average.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()
OUT=fs.OUT

def load(name):
    return {int(k):v for k,v in json.loads((OUT/f'metrics_{name}.json').read_text()).items()}
def sma(v,w=7):
    a=np.asarray(v,float)
    return a if len(a)<w else np.convolve(a,np.ones(w)/w,mode='same')

def fig5a():
    fig,ax=plt.subplots(figsize=(4.6,3.1))
    bars=ax.bar(np.arange(2),[40.22,61.96],width=0.5,color=[fs.PRIMARY_LIGHT,fs.PRIMARY])
    for b,v in zip(bars,[40.22,61.96]):
        ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}%',ha='center',fontsize=8.4,color=fs.INK2)
    ax.set_xticks(np.arange(2),['Candidate\n(neg. advantage)','Teacher-competence\nretained'])
    ax.set_ylim(0,100); ax.set_ylabel('Share of trajectories (%)')
    fs.save(fig,'fig5a_selectivity')

def fig5b():
    ag=load('agopd'); rl=load('rlopd'); steps=np.array(sorted(ag))
    fig,ax=plt.subplots(figsize=(4.6,3.1))
    ax.plot(steps,sma([rl[s]['distill'] for s in steps]),color=fs.NEG,lw=1.6,label='RL+OPD')
    ax.plot(steps,sma([ag[s]['distill'] for s in steps]),color=fs.PRIMARY,lw=1.6,label='AGOPD')
    ax.set_xlabel('Training step'); ax.set_ylabel('Masked distillation loss')
    ax.set_xlim(1,max(steps)); ax.set_ylim(bottom=0)
    ax.legend(frameon=False,fontsize=7.6,loc='upper right')
    fs.save(fig,'fig5b_loss')

if __name__=='__main__':
    fig5a(); fig5b()
