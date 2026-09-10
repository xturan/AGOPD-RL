#!/usr/bin/env python3
"""figI_soft_vs_hard.png : AGOPD distill target soft(softT07) vs hard(argmax) entropy.
9-step centered moving average; palette; single panel."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MAIN='#155b66'; AUX='#9fb9b7'; NEU='#6f7779'
def sma(a,w=9):
    a=np.asarray(a,float); h=w//2; out=[]
    for i in range(len(a)):
        lo=max(0,i-h); hi=min(len(a),i+h+1)
        out.append(a[lo:hi].mean())
    return out
def main():
    d=json.load(open('/tmp/soft_hard_metrics.json'))
    fig,ax=plt.subplots(figsize=(6.0,3.3))
    for key,col,lab in [('hard',MAIN,'AGOPD hard (teacher argmax)'),
                        ('softT07',AUX,'AGOPD soft (softT07)')]:
        ss=d[key]['steps']; ax.plot(ss,sma(d[key]['entropy']),color=col,lw=2.0,label=lab)
    ax.set_xlim(0,190); ax.set_ylim(0,0.4); ax.set_yticks([0,.1,.2,.3,.4])
    ax.set_xlabel('training step',fontsize=10); ax.set_ylabel('policy entropy',fontsize=10)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#bbbbbb')
    ax.tick_params(colors='#444444',labelsize=9)
    ax.legend(frameon=False,fontsize=8.6,loc='lower right')
    fig.tight_layout(); fig.savefig('reports/figures/figI_soft_vs_hard.png',dpi=200); plt.close(fig)
    print('figI_soft_vs_hard.png written')
if __name__=='__main__': main()
