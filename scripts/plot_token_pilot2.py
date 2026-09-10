#!/usr/bin/env python3
"""Pilot figures for Appendix F (state-level counterfactual, token-distribution evidence).
figF1: rescue-rate vs prefix depth (teacher vs student, 3 examples pooled, n=24/pt).
figF2: four-combo (student_correct x teacher_correct) mean top-16 token entropy.
Single-panel PNGs, palette per conventions, no grid, text in panel kept to axes only."""
import json, math, statistics, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
MAIN='#155b66'; AUX='#9fb9b7'; NEG='#9c4b50'; NEU='#6f7779'
def load(p):
    return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def meanent(r):
    es=[]
    for t in r['tokens']:
        top=[x for x in t.get('top',[]) if x.get('lp') is not None]
        if not top: continue
        ps=[math.exp(x['lp']) for x in top]; z=sum(ps)
        if z<=0: continue
        ps=[p/z for p in ps]; es.append(-sum(p*math.log(p) for p in ps))
    return statistics.mean(es) if es else None
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#bbbbbb')
    ax.tick_params(colors='#444444',labelsize=9)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
def main():
    T=load('outputs/token_pilot/teacher2.jsonl'); S=load('outputs/token_pilot/student2.jsonl')
    def rate(rows,cond):
        sub=[r for r in rows if r['cond']==cond]
        return sum(int(r['correct']) for r in sub)/len(sub) if sub else None
    order=['correct','root','wrong_25','wrong_50','wrong_75','wrong_full']
    xl=['correct\nprefix','root','wrong\n25%','wrong\n50%','wrong\n75%','wrong\nfull']
    tr=[rate(T,c) for c in order]; sr=[rate(S,c) for c in order]
    # --- figF1 depth ---
    fig,ax=plt.subplots(figsize=(5.4,3.3))
    xs=range(len(order))
    ax.plot(xs,tr,'-o',color=MAIN,lw=2.2,ms=6,label='teacher')
    ax.plot(xs,sr,'-o',color=AUX,lw=2.2,ms=6,label='student')
    ax.set_xticks(list(xs)); ax.set_xticklabels(xl,fontsize=9)
    ax.set_ylim(-0.04,1.04); ax.set_yticks([0,.25,.5,.75,1])
    ax.set_ylabel('P(correct answer)',fontsize=10)
    style(ax); ax.legend(frameon=False,fontsize=9,loc='lower left',ncol=2)
    for x in xs: ax.annotate(f"{tr[x]:.0%}",(x,tr[x]),textcoords='offset points',xytext=(0,6),ha='center',fontsize=8,color=MAIN)
    for x in xs: ax.annotate(f"{sr[x]:.0%}",(x,sr[x]),textcoords='offset points',xytext=(0,-13),ha='center',fontsize=8,color=AUX)
    fig.tight_layout(); fig.savefig('reports/figures/figF1_rescue_depth.png',dpi=200); plt.close(fig)
    # --- figF2 four-combo entropy ---
    def combos(Trows,Srows):
        gT={}; gS={}
        for r in Trows: gT.setdefault((r['idx'],r['cond']),[]).append(r)
        for r in Srows: gS.setdefault((r['idx'],r['cond']),[]).append(r)
        out={}
        for k in set(gT)&set(gS):
            for a,b in zip(gT[k],gS[k]):
                es=[e for e in (meanent(a),meanent(b)) if e is not None]
                if es: out.setdefault((bool(b['correct']),bool(a['correct'])),[]).append(statistics.mean(es))
        return out
    co=combos(T,S)
    order2=[(1,1),(0,1),(1,0),(0,0)]
    xl2=['both\ncorrect','teacher ok\nstudent wrong','teacher wrong\nstudent ok','both\nwrong']
    labs={}
    vals=[statistics.mean(co[k]) if k in co else None for k in order2]
    ns=[len(co[k]) if k in co else 0 for k in order2]
    fig,ax=plt.subplots(figsize=(5.6,3.3))
    cols=[MAIN,AUX,NEG,NEU]
    ax.bar(range(len(order2)),vals,width=0.6,color=cols,alpha=0.9)
    ax.set_xticks(range(len(order2))); ax.set_xticklabels(xl2,fontsize=8.6,rotation=12,ha='right')
    ax.set_ylabel('mean top-16 entropy',fontsize=10)
    ax.set_ylim(0,0.17)
    style(ax)
    for i,(v,n) in enumerate(zip(vals,ns)):
        if v is not None: ax.annotate(f'{v:.3f}',(i,v),textcoords='offset points',xytext=(0,3),ha='center',fontsize=9,color=cols[i])
    fig.tight_layout(); fig.savefig('reports/figures/figF2_fourcombo_ent.png',dpi=200); plt.close(fig)
    print('figF1_rescue_depth.png figF2_fourcombo_ent.png written')
if __name__=='__main__': main()
