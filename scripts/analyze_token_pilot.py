#!/usr/bin/env python3
"""Analyze token pilot: merge teacher/student per (idx,cond), four-combo top-16
entropy, and dump a small summary + csv for plotting."""
import json, math, glob, sys
from collections import defaultdict
import numpy as np
def load(p):
    rows=[]
    for ln in open(p,encoding='utf-8'):
        ln=ln.strip()
        if ln: rows.append(json.loads(ln))
    return rows
def tok_entropy(tk):
    top=[x for x in tk.get('top',[]) if x.get('lp') is not None]
    if not top: return None
    ps=[math.exp(x['lp']) for x in top]
    z=sum(ps)
    if z<=0: return None
    ps=[p/z for p in ps]
    return -sum(p*math.log(p) for p in ps)
def main():
    T=load(sys.argv[1]); S=load(sys.argv[2])
    def ent_by_correct(rows):
        d={}
        for r in rows:
            ent=[tok_entropy(t) for t in r.get('tokens',[]) if tok_entropy(t) is not None]
            if ent: d.setdefault(bool(r['correct']),[]).append(float(np.mean(ent)))
        return d
    te=ent_by_correct(T); se=ent_by_correct(S)
    # four combos approximated by pairing order within (idx,cond)
    def key(r): return (r['idx'],r['cond'])
    gT=defaultdict(list); gS=defaultdict(list)
    for r in T: gT[key(r)].append(r)
    for r in S: gS[key(r)].append(r)
    combo=defaultdict(list)
    for k in set(gT)&set(gS):
        for a,b in zip(gT[k],gS[k]):
            ent=[]
            for t in (a['tokens']+b['tokens']):
                e=tok_entropy(t)
                if e is not None: ent.append(e)
            if ent:
                keyname=(bool(b['correct']),bool(a['correct']))  # (student,teacher)
                combo[keyname].append(float(np.mean(ent)))
    names={(1,1):'both_correct',(1,0):'student_correct_teacher_wrong',(0,1):'student_wrong_teacher_correct',(0,0):'both_wrong'}
    import statistics as _st
    print('TEACHER correct-mean top16 entropy:', {k:round(_st.mean(v),3) for k,v in te.items()})
    print('STUDENT correct-mean top16 entropy:', {k:round(_st.mean(v),3) for k,v in se.items()})
    print('FOUR-COMBO mean entropy:')
    for k in names:
        v=combo.get(k)
        print(' ',names[k], round(float(np.mean(v)),3) if v else None, 'n', len(v) if v else 0)
    out={'combo':{names[k]: (round(float(np.mean(v)),3), len(v)) if v else None for k in names}}
    json.dump(out,open('reports/figures/token_pilot_summary.json','w'),indent=2)
if __name__=='__main__': main()
