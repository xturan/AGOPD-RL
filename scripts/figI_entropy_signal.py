#!/usr/bin/env python3
"""figI_entropy.png / figI_score.png : overnight 'entropy-signal dilemma' variants.
Real per-step logs -> /tmp/variant_metrics.json. Reference arms from metrics_*.json.
9-step centered moving average (count-normalized) to remove step-level noise."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MAIN='#155b66'; AUX='#9fb9b7'; RED='#9c4b50'; NEU='#6f7779'
def load_agg(name):
    d=json.load(open(f'reports/figures/metrics_{name}.json'))
    ss=sorted(int(k) for k in d)
    return ss,[d[str(s)]['entropy'] for s in ss],[d[str(s)]['score'] for s in ss]
def sma(a,w=9):
    a=np.asarray(a,float); h=w//2; out=[]
    for i in range(len(a)):
        lo=max(0,i-h); hi=min(len(a),i+h+1)
        out.append(a[lo:hi].mean())
    return out
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#bbbbbb')
    ax.tick_params(colors='#444444',labelsize=9)
def main():
    V=json.load(open('/tmp/variant_metrics.json'))
    gss,gent,gsc=load_agg('grpo_fullparam'); ass,aent,asc=load_agg('agopd')
    refs=[('AGOPD main (overlap anchor + gate)',ass,aent,asc,MAIN,'-',1.8),
          ('GRPO main',gss,gent,gsc,NEU,'--',1.2)]
    varset=[('disagreement pure OPD (no RL)',V['disagreement_opd'],RED,'-',2.0),
            ('teachable AGOPD (KL 0.05)',V['teachable'],NEU,'-',1.7),
            ('teachable + entropy penalty β0.02',V['teachable_ep'],AUX,'-',1.7)]
    def do(key,ylab,ylim,extract):
        fig,ax=plt.subplots(figsize=(6.8,3.5))
        for lab,ss,en,sc,col,ls,lw in refs:
            ys=extract(en,sc); ax.plot(ss,sma(ys),color=col,ls=ls,lw=lw)
        for lab,d,col,ls,lw in varset:
            ys=[extract(d['entropy'],d['score'])[i] for i in range(len(d['steps']))]
            ax.plot(d['steps'],sma(ys),color=col,ls=ls,lw=lw)
        ax.set_xlim(0,190); ax.set_ylim(*ylim)
        ax.set_xlabel('training step',fontsize=10); ax.set_ylabel(ylab,fontsize=10)
        style(ax)
        handles=[plt.Line2D([],[],color=c,ls=l,lw=w,label=lab) for lab,ss,en,sc,c,l,w in refs]+[plt.Line2D([],[],color=c,ls=l,lw=w,label=lab) for lab,d,c,l,w in varset]
        ax.legend(handles=handles,frameon=False,fontsize=7.2,loc='upper left')
        fig.tight_layout(); fig.savefig(f'reports/figures/figI_{key}.png',dpi=200); plt.close(fig)
        print('wrote figI_%s.png'%key)
    do('entropy','policy entropy',(0,0.7),lambda en,sc:en)
    do('score','online mean result score',(0,0.7),lambda en,sc:sc)
if __name__=='__main__': main()
