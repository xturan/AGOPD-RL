#!/usr/bin/env python3
"""Faithful reconstruction (pixel-informed) of the ORIGINAL Appendix B figures.
Detected from the embedded originals: 1630x490 canvas, old palette main green
#54906c, red/pink #cc7890, light grey grid, two side-by-side panels.
Layout/data match make_figures fig4_channel (grad/pg) and fig5_collapse
(rlen/score). Approximate at composition level; tune on visual diff."""
from __future__ import annotations
import json, sys, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT=Path(__file__).resolve().parents[1]/'reports/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#d8d8d8','grid.linewidth':0.7,'grid.alpha':0.8})
GREEN='#54906c'; PINK='#cc7890'; GRAYBLUE='#8ea7b8'; GRAY2='#b4babd'
def load(name): return {int(k):v for k,v in json.loads((OUT/f'metrics_{name}.json').read_text()).items()}
def fig_common():
    fig,axes=plt.subplots(1,2,figsize=(13.0,3.9))
    fig.patch.set_facecolor('white')
    return fig,axes
def sma(v,w=7):
    a=np.asarray(v,float)
    return a if len(a)<w else np.convolve(a,np.ones(w)/w,mode='same')
def draw(ax,d,key,color,label,ls='-',lw=2.2):
    st=sorted(d); ax.plot(st,sma([d[s][key] for s in st]),color=color,ls=ls,lw=lw,label=label)
def finalize(fig,stem,ax,capy=1.0):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{stem}.png',dpi=200,facecolor='white')
    plt.close(fig)

def _one(ax): pass
def save4(name_ax):
    ax=name_ax
    ax.set_xlabel('step')
    ax.grid(True,color='#d8d8d8',linewidth=0.7,alpha=0.8)
    ax.margins(x=0.01)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(stem+'.png'),dpi=200,facecolor='white')
    plt.close(fig)
def B1():
    l1=load('lora_lr1e4'); l2=load('lora_lr3e4'); g=load('grpo_fullparam')
    out=[]
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    draw(ax,l1,'grad',GRAYBLUE,'LoRA r16 lr1e-4',ls='--'); draw(ax,l2,'grad',GRAY2,'LoRA r16 lr3e-4',ls='--'); draw(ax,g,'grad',GREEN,'full-param lr1e-6')
    ax.set_ylim(0,1.25); ax.legend(frameon=False,fontsize=9); ax.set_xlabel('step')
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/'figB1a_grad.png',dpi=200,facecolor='white'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    draw(ax,l1,'pg',GRAYBLUE,'LoRA r16 lr1e-4',ls='--'); draw(ax,l2,'pg',GRAY2,'LoRA r16 lr3e-4',ls='--'); draw(ax,g,'pg',GREEN,'full-param')
    ax.set_ylim(0,0.11); ax.legend(frameon=False,fontsize=9); ax.set_xlabel('step')
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/'figB1b_pg.png',dpi=200,facecolor='white'); plt.close(fig)
def B2():
    c=load('fp_collapse_lr1e5'); g=load('grpo_fullparam')
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    draw(ax,c,'rlen',PINK,'lr1e-5 (collapse)',ls='--'); draw(ax,g,'rlen',GREEN,'lr1e-6 + KL0.05 (stable)')
    ax.axhline(2048,color='#999',ls=':',lw=1); ax.set_ylim(0,2200); ax.legend(frameon=False,fontsize=9); ax.set_xlabel('step')
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/'figB2a_rlen.png',dpi=200,facecolor='white'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.9))
    draw(ax,c,'score',PINK,'lr1e-5',ls='--'); draw(ax,g,'score',GREEN,'lr1e-6 + KL0.05')
    ax.set_ylim(0,0.5); ax.legend(frameon=False,fontsize=9); ax.set_xlabel('step')
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/'figB2b_score.png',dpi=200,facecolor='white'); plt.close(fig)
if __name__=='__main__':
    B1(); B2()
