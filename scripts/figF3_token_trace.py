#!/usr/bin/env python3
"""figF3_token_trace.png : token-stream illustration for Appendix F.
db3e2e94, wrong_50 prefix: teacher (rescues, correct) vs student (stuck, wrong).
Each row = a 9-token window selected at the sample's highest top-16 entropy region
(a reasoning decision point). Each token chip carries a mini horizontal top-16 bar:
sampled token's probability mass colored (teacher teal / student red); remaining
top-16 mass gray -- peaked => mostly colored; diffuse => large gray mass.
Real data; single panel; palette per conventions."""
import json, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TEAL='#155b66'; RED='#9c4b50'; GRAY='#c9cccd'; INK='#333333'
TEALT='#e4efed'; REDT='#f6e6e5'
def load(p):
    return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def rowent(tokens):
    es=[]
    for x in tokens:
        ps=[math.exp(q['lp']) for q in x.get('top',[]) if q.get('lp') is not None]
        z=sum(ps)
        if z>0:
            ps=[p/z for p in ps]; es.append(-sum(p*math.log(p) for p in ps))
    return statistics.mean(es) if es else None
def best_window(tokens,n=9):
    idxs=[i for i in range(len(tokens)) if '<|im_end|>' not in (tokens[i].get('t') or '')]
    if len(idxs)<n: return tokens[max(0,len(tokens)-n):]
    best=None; be=-1.0
    for s in range(len(idxs)-n+1):
        win=[tokens[idxs[s+i]] for i in range(n)]
        e=rowent(win) or 0.0
        if e>be: be=e; best=win
    return best
def barset(tok):
    items=[(math.exp(x['lp']) if x.get('lp') is not None else 0.0) for x in tok.get('top',[])]
    z=sum(items)
    if z<=0: return [(1.0,False)]
    return [(p/z, i==0) for i,p in enumerate(items)]
def main():
    T=load('outputs/token_pilot/teacher2.jsonl'); S=load('outputs/token_pilot/student2.jsonl')
    def pick(rows,correct):
        for r in rows:
            if r['idx'].startswith('db3e2e94') and r['cond']=='wrong_50' and r['correct']==correct:
                return r
    tr=pick(T,True); st=pick(S,False)
    assert tr and st, 'samples missing'
    clean=lambda t:(t or '').replace('\n',' ').replace('\r',' ').strip()
    fig,ax=plt.subplots(figsize=(13.2,3.7))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    rows=[(f'teacher · rescues (8/8) · mean H={rowent(tr["tokens"]):.3f}',tr,TEAL,TEALT),
          (f'student · stuck (0/8) · mean H={rowent(st["tokens"]):.3f}',st,RED,REDT)]
    band=[0.88,0.34]
    n=9
    for (lab,r,col,tint),top in zip(rows,band):
        win=best_window(r['tokens'],n)
        ax.text(0.004, top+0.045, lab, transform=ax.transAxes, fontsize=10.5, color=col,
                fontweight='bold', va='center')
        ax.text(0.004, top-0.035, '9-token window\n(highest-entropy span)', transform=ax.transAxes,
                fontsize=7.2, color='#666666', va='center')
        x0=0.175; dx=0.0825
        for j,tk in enumerate(win):
            x=x0+j*dx; cx=x+dx/2
            ax.text(cx, top, clean(tk.get('t'))[:12], transform=ax.transAxes, fontsize=7.4,
                    color=INK, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.16', fc=tint, ec=col, lw=1.3))
            # mini top-16 horizontal bar
            bx=x+dx*0.10; bw=dx*0.80; by=top-0.135; xw=bx
            for p,chosen in barset(tk):
                w=p*bw
                ax.barh(by, w, left=xw, height=0.030, color=col if chosen else GRAY, edgecolor='none')
                xw+=w
    ax.text(0.004,0.115,'Mini bars = top-16 mass at each position (colored = sampled token; gray = alternatives). '
            'Window is the 9-token highest-entropy span; rows are not time-aligned.',fontsize=7.8,color=INK,transform=ax.transAxes,va='center')
    ax.text(0.004,0.035,'Same wrong_50 prefix (db3e2e94); each row is one real continuation out of n=8.',fontsize=7.8,
            color=INK,transform=ax.transAxes,va='center')
    fig.savefig('reports/figures/figF3_token_trace.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print('figF3_token_trace.png written')
if __name__=='__main__': main()
