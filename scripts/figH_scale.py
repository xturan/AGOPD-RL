#!/usr/bin/env python3
"""Appendix H proof figures:
figH1_signal_density.png  - signal density S(6) (census) by scale.
figH2_grpo_outcome.png    - actual outcome: same-budget (50-step) GRPO, fixed-1024 semantic.
Single-panel, palette, no grid."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MAIN='#155b66'; AUX='#9fb9b7'; NEG='#9c4b50'; NEU='#6f7779'; INK='#333333'
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#bbbbbb')
    ax.tick_params(colors='#444444',labelsize=10)
# fig H1
fig,ax=plt.subplots(figsize=(3.6,3.2))
labels=['1.7B\nfull census','4B\n2k sample']; vals=[36.0,41.7]
bars=ax.bar([0,1],vals,width=0.55,color=[MAIN,MAIN],alpha=0.9)
ax.set_xticks([0,1]); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylim(0,55); ax.set_yticks([0,10,20,30,40,50])
ax.set_ylabel('S(6)  (signal density, %)',fontsize=9)
style(ax)
for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),textcoords='offset points',xytext=(0,4),ha='center',fontsize=10,color=MAIN)
ax.text(-0.42,-0.5,'p̂=0:  48.9% → 38.9%\nmean p̂:  0.232 → 0.301',transform=ax.transAxes,fontsize=8,color=INK,va='top')
fig.tight_layout(); fig.savefig('reports/figures/figH1_signal_density.png',dpi=200); plt.close(fig)
# fig H2
fig,ax=plt.subplots(figsize=(4.3,3.2))
labels2=['1.7B','4B']; v2=[27.4,40.2]
ax.bar([0,1],v2,width=0.55,color=[MAIN,MAIN],alpha=0.9)
ax.set_xticks([0,1]); ax.set_xticklabels(labels2,fontsize=11)
ax.set_ylim(0,55); ax.set_yticks([0,10,20,30,40,50])
ax.set_ylabel('semantic acc. (%)',fontsize=9.5)
style(ax)
for i,v in enumerate(v2): ax.annotate(f'{v:.1f}%',(i,v),textcoords='offset points',xytext=(0,4),ha='center',fontsize=10,color=MAIN)
ax.annotate('Δ +12.8 pp',(0.5,40.2),xytext=(0.5,45),ha='center',fontsize=9,color=NEG,fontweight='bold')
ax.text(-0.42,-0.5,'fixed 1,024 eval · same budget: 50-step GRPO\nnon-think @2,048 · seed 42',transform=ax.transAxes,fontsize=8,color=INK,va='top')
fig.tight_layout(); fig.savefig('reports/figures/figH2_grpo_outcome.png',dpi=200); plt.close(fig)
print('figH1_signal_density.png figH2_grpo_outcome.png written')
