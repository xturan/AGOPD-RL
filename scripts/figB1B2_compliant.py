#!/usr/bin/env python3
"""Appendix B figures (single-panel per image, uniform thin lines lw=0.7,
palette-fixed, grid-free). HTML places (a)/(b) short titles under each image.
B1: update channel (grad norm / pg loss). B2: LR stability (response length /
online score)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup()
OUT=fs.OUT
def load(name): return {int(k):v for k,v in json.loads((OUT/f'metrics_{name}.json').read_text()).items()}
def save_one(ax, stem):
    ax.set_xlabel('Training step')
    fs.save(fig, stem)
    plt.close(fig)
def line(ax,d,key,color,label,ls='-'):
    st=sorted(d); ax.plot(st,[d[s][key] for s in st],color=color,ls=ls,lw=1.0,label=label)
def leg(ax): ax.legend(frameon=False,fontsize=6.4)
def figB1a():
    global fig
    l1=load('lora_lr1e4'); l2=load('lora_lr3e4'); g=load('grpo_fullparam')
    fig,ax=plt.subplots(figsize=(4.6,2.9))
    line(ax,l1,'grad',fs.PRIMARY_LIGHT,'LoRA lr1e-4'); line(ax,l2,'grad',fs.NEUTRAL,'LoRA lr3e-4'); line(ax,g,'grad',fs.PRIMARY,'full-param lr1e-6')
    ax.set_ylabel('Gradient norm')
    leg(ax); save_one(ax,'figB1a_grad')
def figB1b():
    global fig
    l1=load('lora_lr1e4'); l2=load('lora_lr3e4'); g=load('grpo_fullparam')
    fig,ax=plt.subplots(figsize=(4.6,2.9))
    line(ax,l1,'pg',fs.PRIMARY_LIGHT,'LoRA lr1e-4'); line(ax,l2,'pg',fs.NEUTRAL,'LoRA lr3e-4'); line(ax,g,'pg',fs.PRIMARY,'full-param')
    ax.set_ylabel('Policy-gradient loss')
    leg(ax); save_one(ax,'figB1b_pg')
def figB2a():
    global fig
    c=load('fp_collapse_lr1e5'); g=load('grpo_fullparam')
    fig,ax=plt.subplots(figsize=(4.6,2.9))
    line(ax,c,'rlen',fs.NEG,'lr1e-5 (collapse)','--'); line(ax,g,'rlen',fs.PRIMARY,'lr1e-6 + KL0.05 (stable)')
    ax.axhline(2048,color=fs.NEUTRAL,ls=':',lw=0.6)
    ax.set_ylabel('Response length')
    leg(ax); save_one(ax,'figB2a_rlen')
def figB2b():
    global fig
    c=load('fp_collapse_lr1e5'); g=load('grpo_fullparam')
    fig,ax=plt.subplots(figsize=(4.6,2.9))
    line(ax,c,'score',fs.NEG,'lr1e-5','--'); line(ax,g,'score',fs.PRIMARY,'lr1e-6 + KL0.05')
    ax.set_ylabel('Online score')
    leg(ax); save_one(ax,'figB2b_score')
if __name__=='__main__':
    figB1a(); figB1b(); figB2a(); figB2b()
