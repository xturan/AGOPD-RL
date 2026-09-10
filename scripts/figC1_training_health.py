import json, sys, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
fs.setup(); OUT=fs.OUT
def load(n): return {int(k):v for k,v in json.loads((OUT/f'metrics_{n}.json').read_text()).items()}
def sma(v,w=7):
    a=np.asarray(v,float); return a if len(a)<w else np.convolve(a,np.ones(w)/w,mode='same')
def panel(name,key,ylab,stem,ylim):
    runs=[('grpo_fullparam','GRPO'),('opd','OPD'),('rlopd','RL+OPD'),('agopd','AGOPD')]
    fig,ax=plt.subplots(figsize=(4.9,3.0))
    for n,lab in runs:
        d=load(n); st=sorted(d)
        stc=fs.METHOD[lab]
        ax.plot(st,sma([d[s][key] for s in st]),color=stc['color'],ls=stc['ls'],lw=1.0,label=lab)
    ax.set_xlabel('Training step'); ax.set_ylabel(ylab); ax.set_xlim(0,190); ax.set_ylim(*ylim)
    ax.legend(frameon=False,fontsize=6.6)
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f'{stem}.png',dpi=200,facecolor='white'); plt.close(fig)
panel('score','score','Online score','figC1a_score',(0,0.6))
panel('entropy','entropy','Policy entropy','figC1b_entropy',(0,0.4))
