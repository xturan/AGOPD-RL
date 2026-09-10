#!/usr/bin/env python3
"""Analyze token pilot v2: per (idx,cond,model) correctness & top-16 token entropy;
four-combo (student_correct x teacher_correct) entropy. Dump summary json + ascii table."""
import json, math, sys, collections, statistics
def load(p):
    return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def tok_entropy(tk):
    top=[x for x in tk.get('top',[]) if x.get('lp') is not None]
    if not top: return None
    ps=[math.exp(x['lp']) for x in top]; z=sum(ps)
    if z<=0 or not math.isfinite(z): return None
    ps=[p/z for p in ps]
    return -sum(p*math.log(p) for p in ps)
def row_ent(tokens):
    es=[tok_entropy(t) for t in tokens]; es=[e for e in es if e is not None]
    return statistics.mean(es) if es else None
def main():
    T=load(sys.argv[1]); S=load(sys.argv[2])
    conds=['root','correct','wrong_25','wrong_50','wrong_75','wrong_full','filler']
    def bymodel(rows):
        d=collections.defaultdict(dict)  # (idx,cond)->{'corr':k,'tot':n,'ent':[...]}
        for r in rows:
            key=(r['idx'],r['cond'])
            rec=d[key]
            rec['corr']=rec.get('corr',0)+int(r['correct']); rec['tot']=rec.get('tot',0)+1
            e=row_ent(r['tokens'])
            if e is not None: rec.setdefault('ents',[]).append(e)
        return d
    dT=bymodel(T); dS=bymodel(S)
    print(f'{"idx":9s} {"cond":10s} {"t_corr":>14s} {"s_corr":>14s} {"t_ent":>9s} {"s_ent":>9s}')
    for idx in sorted(set([k[0] for k in dT])|set([k[0] for k in dS])):
        for c in conds:
            t=dT.get((idx,c)); s=dS.get((idx,c))
            if t is None and s is None: continue
            tc=f"{t['corr']}/{t['tot']}" if t else '-'
            sc=f"{s['corr']}/{s['tot']}" if s else '-'
            te=f"{statistics.mean(t['ents']):.3f}" if t and t['ents'] else '-'
            se=f"{statistics.mean(s['ents']):.3f}" if s and s['ents'] else '-'
            print(f'{idx[:8]:9s} {c:10s} {tc:>14s} {sc:>14s} {te:>9s} {se:>9s}')
    # four-combo: pair by (idx,cond, sample ordinal) teacher x student
    def g(rows):
        d=collections.defaultdict(list)
        for r in rows: d[(r['idx'],r['cond'])].append(r)
        return d
    gT=g(T); gS=g(S)
    combo=collections.defaultdict(list)
    for k in set(gT)&set(gS):
        for a,b in zip(gT[k],gS[k]):   # a teacher, b student
            es=[e for e in (row_ent(a['tokens']),row_ent(b['tokens'])) if e is not None]
            if es:
                combo[(bool(b['correct']),bool(a['correct']))].append(statistics.mean(es))
    names={(1,1):'both_correct',(0,1):'s_wrong_t_correct',(1,0):'s_correct_t_wrong',(0,0):'both_wrong'}
    print('\nFOUR-COMBO (student,teacher) top-16 mean entropy / token:')
    out={}
    for k in [(1,1),(0,1),(1,0),(0,0)]:
        v=combo.get(k)
        if v:
            print(f"  {names[k]:18s} mean {statistics.mean(v):.4f}  n={len(v)}")
            out[names[k]]={'mean':round(statistics.mean(v),4),'n':len(v)}
        else:
            print(f"  {names[k]:18s} EMPTY")
            out[names[k]]=None
    summary={'conds':{},'combo':out}
    for (idx,c) in sorted(set(dT)|set(dS)):
        t=dT.get((idx,c)); s=dS.get((idx,c))
        summary['conds'][f'{idx[:8]}|{c}']={
            't_corr':(t['corr'],t['tot']) if t else None,
            's_corr':(s['corr'],s['tot']) if s else None,
            't_ent':(round(statistics.mean(t['ents']),4),len(t['ents'])) if t and t['ents'] else None,
            's_ent':(round(statistics.mean(s['ents']),4),len(s['ents'])) if s and s['ents'] else None,
        }
    json.dump(summary,open(sys.argv[3],'w'),indent=2)
    print('\nwrote',sys.argv[3])
if __name__=='__main__': main()
