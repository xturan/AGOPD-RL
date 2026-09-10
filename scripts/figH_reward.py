#!/usr/bin/env python3
"""figH_reward.png : GRPO training reward vs base scale.
1.7B = four-arm main GRPO (full-param, 184 steps) from reports/figures/metrics_grpo_fullparam.json.
4B   = scale_grpo_4b_50step_ckpt2 run (full-param lr1e-6, 50 steps) from /tmp log critic/score/mean.
5-step moving average; raw faint underneath."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MAIN='#155b66'; AUX='#9fb9b7'; INK='#333333'
R4=[0.406,0.219,0.375,0.312,0.469,0.406,0.312,0.281,0.344,0.438,0.250,0.469,0.219,0.344,0.375,
    0.344,0.406,0.469,0.531,0.438,0.562,0.219,0.344,0.375,0.594,0.562,0.531,0.469,0.438,0.219,
    0.219,0.438,0.375,0.438,0.344,0.500,0.562,0.406,0.562,0.406,0.375,0.469,0.281,0.469,0.469,
    0.500,0.344,0.250,0.531,0.656]
def sma(a,w=5):
    a=np.asarray(a,float); h=w//2; out=[]
    for i in range(len(a)):
        lo=max(0,i-h); hi=min(len(a),i+h+1)
        out.append(a[lo:hi].mean())
    return np.asarray(out)
def main():
    g=json.load(open('reports/figures/metrics_grpo_fullparam.json'))
    s17=sorted((int(k) for k in g))
    r17=[g[str(s)]['score'] for s in s17]
    fig,ax=plt.subplots(figsize=(6.4,3.4))
    s17r=sma(r17); s4r=sma(R4)
    ax.plot(s17,s17r,color=AUX,lw=2.2,zorder=3,label='GRPO 1.7B (main)')
    steps4=list(range(1,51))
    ax.plot(steps4,s4r,color=MAIN,lw=2.2,zorder=3,label='GRPO 4B (50-step)')
    ax.set_xlim(0,190); ax.set_ylim(0,0.8); ax.set_yticks([0,.2,.4,.6,.8])
    ax.set_xlabel('GRPO training step',fontsize=10); ax.set_ylabel('online mean result score',fontsize=9.5)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#bbbbbb')
    ax.tick_params(colors='#444444',labelsize=9)
    ax.legend(frameon=False,fontsize=8.6,loc='lower right')
    # end annotations placed above the smooth lines (no raw points underneath)
    ax.annotate(f'end≈{float(s17r[-1]):.2f}',(s17[-1],s17r[-1]),xytext=(176,0.63),
                fontsize=8.5,color=AUX,ha='center',
                arrowprops=dict(arrowstyle='-',color=AUX,lw=0.8))
    ax.annotate(f'end≈{float(s4r[-1]):.2f}',(50,s4r[-1]),xytext=(30,0.70),
                fontsize=8.5,color=MAIN,ha='center',
                arrowprops=dict(arrowstyle='-',color=MAIN,lw=0.8))
    fig.tight_layout(); fig.savefig('reports/figures/figH_reward.png',dpi=200); plt.close(fig)
    print('figH_reward.png written; sma17end',round(float(sma(r17)[-1]),3),'sma4end',round(float(sma(R4)[-1]),3))
if __name__=='__main__': main()
