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
def pre_at(tok,text,frac):
    ids=tok(text,add_special_tokens=False)['input_ids']
    return tok.decode(ids[:max(1,int(len(ids)*frac))],skip_special_tokens=True)
def chat(tok,c):
    try: return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{'role':'user','content':c}],tokenize=False,add_generation_prompt=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--examples',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=2)
    ap.add_argument('--max',type=int,default=256); a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    data=json.load(open(a.examples,encoding='utf-8'))
    ex=[e for e in data if str(e.get('idx','')).startswith(('e8ae9ae6','e408476f'))]
    df=pd.read_parquet('data/dapo-verl-v1/train_comfort.parquet')
    by={str((r.get('extra_info') or {}).get('index','')):r for r in df.to_dict('records')}
    tok=AutoTokenizer.from_pretrained(a.model)
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=3072,enforce_eager=True)
    sp=SamplingParams(n=a.n,max_tokens=a.max,temperature=0.6,top_p=0.95,top_k=20,logprobs=16)
    import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    out=open(a.out,'w',encoding='utf-8')
    for e in ex:
        idx=e['idx']; row=by.get(idx)
        if row is None: print('skip',idx); continue
        rm=row.get('reward_model') or {}; gt=str(rm.get('ground_truth',''))
        content=content_of(row); wrong=str(e.get('student_wrong_prefix') or '')
        pt=e.get('prefix_tokens',512)
        conds=[('root',None,'root')]
        for fr in (0.25,0.5,0.75): conds.append(('wrong_'+str(int(fr*100)),pre_at(tok,wrong,fr),'path'))
        conds.append(('wrong_full',wrong,'path'))
        filler='neutral draft placeholder text '*max(16,pt//4)
        conds.append(('filler',pre_at(tok,filler,pt),'ignore'))
        for name,prefix,mode in conds:
            texts=[chat(tok,cond(content,prefix,mode)) for _ in range(a.n)]
            for r in llm.generate(texts,sp):
                o=r.outputs[0]; out_t=o.text
                lps=getattr(o,'logprobs',None) or []
                def _tok(x):
                    if isinstance(x,dict): return x.get('token',x.get('decoded_token',x.get('token_id')))
                    return getattr(x,'token',getattr(x,'decoded_token',getattr(x,'token_id',None)))
                def _logp(x):
                    if isinstance(x,dict): return x.get('logprob')
                    return getattr(x,'logprob',None)
                def _top(x):
                    if isinstance(x,dict): return x.get('top_logprobs') or []
                    return getattr(x,'top_logprobs',None) or []
                if os.environ.get('TOK_DEBUG') and lps:
                    import sys as _s
                    _s.stderr.write('LP0 type='+str(type(lps[0]))+' repr='+repr(lps[0])[:400]+'\n')
                toks=[]
                for lp in lps:
                    items=sorted(lp.items(), key=lambda kv: (gv(kv[1],'logprob') if gv(kv[1],'logprob') is not None else -1e9), reverse=True)
                    top=[{'t':gv(v,'decoded_token'),'lp':gv(v,'logprob')} for _t,v in items]
                    toks.append({'t': (top[0]['t'] if top else None), 'logp': (top[0]['lp'] if top else None), 'top': top})
                rec={'idx':idx,'gt':gt,'cond':name,'correct':bool(score_math_response(out_t,gt).correct),'tokens':toks}
                out.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print('done',idx,flush=True)
    out.close()
if __name__=='__main__': main()
