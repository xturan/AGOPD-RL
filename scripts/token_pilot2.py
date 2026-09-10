#!/usr/bin/env python3
import argparse,json,os
import pandas as pd
from transformers import AutoTokenizer
from agopd.reward.math_reward import score_math_response
def gv(o,k): return (o.get(k) if isinstance(o,dict) else getattr(o,k))
def content_of(row):
    p=row['prompt']
    if isinstance(p,str): return p
    if hasattr(p,'tolist'): p=p.tolist()
    if isinstance(p,list):
        for m in p:
            if isinstance(m,dict) and m.get('role')=='user': return str(m.get('content',''))
    return str(p)
def stn(s): return s.replace('\n\n/no_think','').replace('\n/no_think','')
def cond(content,prefix,mode):
    base=stn(content)
    if mode=='root': return base+'\n\n/no_think'
    inst=('The draft below may be wrong. Ignore it completely and solve the original problem independently.'
          if mode=='ignore' else 'The draft below may be wrong. Continue the solution, correct any mistakes you detect, and give the final answer.')
    return base+'\n\n'+inst+'\n\n<student_draft>\n'+(prefix or '')+'\n</student_draft>\n\n/no_think'
def pre_n(tok,text,n):
    ids=tok(text,add_special_tokens=False)['input_ids']
    return tok.decode(ids[:max(1,n)],skip_special_tokens=True)
def chat(tok,c):
    try: return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--n',type=int,default=8)
    ap.add_argument('--out',required=True); ap.add_argument('--max',type=int,default=2048)
    ap.add_argument('--tag',default='')
    ap.add_argument('--gpu',type=int,default=0); a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    data=json.load(open('reports/cptr/control5_examples.json',encoding='utf-8'))
    refs=json.load(open('reports/figures/reference_solutions.json',encoding='utf-8'))
    sel=[e for e in data if any(str(e.get('idx','')).startswith(k) for k in refs)]
    df=pd.read_parquet('data/dapo-verl-v1/train_comfort.parquet')
    by={str((r.get('extra_info') or {}).get('index','')):r for r in df.to_dict('records')}
    tok=AutoTokenizer.from_pretrained(a.model)
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype='bfloat16',gpu_memory_utilization=0.9,max_model_len=4096)
    sp=SamplingParams(n=a.n,max_tokens=a.max,temperature=0.6,top_p=0.95,top_k=20,logprobs=16,seed=42)
    import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True); out=open(a.out,'w',encoding='utf-8')
    for e in sel:
        idx=e['idx']; row=by.get(idx)
        if row is None: continue
        rm=row.get('reward_model') or {}; gt=str(rm.get('ground_truth','')); content=content_of(row)
        wrong=str(e.get('student_wrong_prefix') or ''); pt=e.get('prefix_tokens',512)
        wids=tok(wrong,add_special_tokens=False)['input_ids']
        k=next((k for k in refs if str(idx).startswith(k)),''); correct=refs.get(k,'')
        conds=[('root',None,'root')]
        conds.append(('correct', pre_n(tok,correct,min(pt,len(wids))),'path'))
        for fr in (0.25,0.5,0.75): conds.append(('wrong_'+str(int(fr*100)),pre_n(tok,wrong,int(len(wids)*fr)),'path'))
        conds.append(('wrong_full',wrong,'path'))
        filler='neutral draft placeholder text '*max(16,pt//4)
        conds.append(('filler',pre_n(tok,filler,pt),'ignore'))
        for name,prefix,mode in conds:
            text=chat(tok,cond(content,prefix,mode))
            req=llm.generate([text],sp)[0]
            for o in req.outputs:
                out_t=o.text; lps=getattr(o,'logprobs',None) or []
                toks=[]
                for lp in lps:
                    items=sorted(lp.items(), key=lambda kv:(gv(kv[1],'logprob') if gv(kv[1],'logprob') is not None else -1e9), reverse=True)
                    top=[{'t':gv(v,'decoded_token'),'lp':gv(v,'logprob')} for _t,v in items]
                    toks.append({'t':top[0]['t'] if top else None,'logp':top[0]['lp'] if top else None,'top':top})
                out.write(json.dumps({'idx':idx,'model':a.tag,'text':out_t,'gt':gt,'cond':name,'correct':bool(score_math_response(out_t,gt).correct),'tokens':toks})+'\n')
        print('done',idx,'lines',sum(1 for _ in open(a.out)),flush=True)
    out.close()
if __name__=='__main__': main()
