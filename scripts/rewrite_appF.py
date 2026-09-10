#!/usr/bin/env python3
# Replace old Appendix F (逐条轨迹展示, lines 454..604) with the new
# 状态级反事实 appendix: sampling protocol + rescue-depth table + four-combo
# entropy + token-trace figure. Preserves line 605 "</section>".
import base64, io
P='reports/AGOPD_paper_v6_0_restructured_20260908.html'
def b64(path):
    return base64.b64encode(open(path,'rb').read()).decode()
F1=b64('reports/figures/figF1_rescue_depth.png')
F2=b64('reports/figures/figF2_fourcombo_ent.png')
F3=b64('reports/figures/figF3_token_trace.png')
text=open(P,encoding='utf-8').read()
lines=text.split('\n')
si=None
for i,l in enumerate(lines):
    if '<h2' in l and 'APPENDIX F' in l:
        si=i; break
assert si is not None, 'heading not found'
ei=None
for j in range(si+1,len(lines)):
    if lines[j].strip()=='</section>':
        ei=j; break
assert ei is not None, 'closing </section> not found'
print('replacing 1-based lines',si+1,'..',ei,'(closing kept at',ei+1,')')
new=[]
A=new.append
A('<h2><span class="secno">APPENDIX F</span>状态级反事实:错误前缀深度上的救援率与词元分布</h2>')
A('<p>§3.3 与图 2 用大规模匹配题集合确立了状态级结论:教师救援率随学生错误前缀加深而下降。为使该结论可在词元粒度复核,并补充正文不具备的分布级证据,本附录将状态干预做成一个显式采样协议,在 3 道代表性训练题上以每题 8 次采样记录两条模型链(教师 Qwen3-4B-grpo-50step-ckpt2、学生 Qwen3-1.7B Base)在同一前缀上的完整续写与其逐词元 top-16 分布。与正文的显著性统计不同,本附录定位为机制示意与可复核诊断工件,不单独承担统计结论。</p>')
A('<h3>F.1 采样协议与判定口径</h3>')
A('<p>采样与判定协议如下:两条模型链均使用非思考聊天模板;解码参数为 temperature 0.6、top-p 0.95、top-k 20、随机种子 42;每个(题目,条件)对采样 n=8。前缀共七档:题目根状态(root);完整可验证正确解与全错草稿等长的正确前缀;学生错误前缀按词元长度的 25%、50%、75% 截取;完整学生错误草稿(全错);以及长度匹配的中性随机填充(并要求模型忽略草稿、独立求解)。每条续写均交由与正文相同的语义结果验证器按最终答案判定正确性,并记录完整文本与逐词元 top-16 对数概率。三题取自 DAPO-Math 训练集,错误前缀源自 Base 学生在该题的真实 rollout;原始前缀与工件见 reports/cptr/control5_examples.json。本附录合计 \(3 \times 7 \times 8 \times 2=336\) 条续写。</p>')
A('<h3>F.2 深度救援率与对照</h3>')
A('<p>图 F1(a) 与表 F1 报告两模型在相同前缀上的答对比例(每档 3 题×8 次,n=24)。方向性观察有三:其一,教师在 25%–75% 深度保持对学生的正向优势,而学生答对率随深度单调下降并在 75% 归零;其二,完整错误草稿(全错)同时压制两模型,教师在该档仍落入几乎全错,说明浅层可纠与深度被锁是不同机制;其三,随机填充档要求忽略草稿独立求解,教师的答对率高于同长的错误草稿档,提示错误草稿的损伤主要来自其错误推理语义,而非上下文长度或草稿存在本身。</p>')
A('<p>词元锐度提供互补视角。表 F1 后两列给出每条续写逐词元 top-16 熵的均值:在同一前缀下,教师分布的熵在各档均低于学生,即教师比学生更“锐”;但全错档教师同样几乎全错,表明锐度并不等于正确性——教师在深度错误草稿上会以较锐的分布自信地延续错误路径。数值一律见表 F1,正文不重复罗列。</p>')
A('<div class="table-wrap"><table><thead><tr><th>前缀条件</th><th class="num">教师答对 (n=24)</th><th class="num">学生答对 (n=24)</th><th class="num">教师 top-16 熵</th><th class="num">学生 top-16 熵</th></tr></thead><tbody>')
A('<tr><td>正确前缀</td><td class="num">13 / 24</td><td class="num">17 / 24</td><td class="num">0.110</td><td class="num">0.124</td></tr>')
A('<tr><td>Root(题目根状态)</td><td class="num">16 / 24</td><td class="num">12 / 24</td><td class="num">0.124</td><td class="num">0.152</td></tr>')
A('<tr><td>错误前缀 25%</td><td class="num">19 / 24</td><td class="num">8 / 24</td><td class="num">0.110</td><td class="num">0.148</td></tr>')
A('<tr><td>错误前缀 50%</td><td class="num">11 / 24</td><td class="num">7 / 24</td><td class="num">0.098</td><td class="num">0.128</td></tr>')
A('<tr><td>错误前缀 75%</td><td class="num">10 / 24</td><td class="num">0 / 24</td><td class="num">0.063</td><td class="num">0.125</td></tr>')
A('<tr><td>全错草稿</td><td class="num">1 / 24</td><td class="num">1 / 24</td><td class="num">0.086</td><td class="num">0.140</td></tr>')
A('<tr><td>随机填充(忽略草稿)</td><td class="num">14 / 24</td><td class="num">8 / 24</td><td class="num">0.134</td><td class="num">0.147</td></tr>')
A('</tbody><caption>表 F1 · 状态级救援率与词元锐度(3 题合并,每档每题 8 次)。熵为逐词元 top-16 条件分布熵的续写内均值,再取档内均值,单位 nat。</caption></table></div>')
A('<figure>')
A('<div class="fig-pair">')
A('<div class="fig-col"><img alt="Rescue rate vs wrong-prefix depth" src="data:image/png;base64,'+F1+'">')
A('<div class="fig-panel-label">(a) 答对率随错误前缀深度</div></div>')
A('<div class="fig-col"><img alt="Four-combo top-16 entropy" src="data:image/png;base64,'+F2+'">')
A('<div class="fig-panel-label">(b) 四组合下的 top-16 熵</div></div>')
A('</div>')
A('<figcaption><b>图 F1 · 状态级救援率与词元分布锐度。</b>(a) 教师与学生面对同一错误前缀的答对比例(n=24/档):教师优势保持至 75% 深度,全错草稿处两者同时趋零;(b) 按(学生判定, 教师判定)四组合分桶的逐词元 top-16 熵均值:都错桶最弥散。配对按(题目, 条件)内采样序号对齐,桶内样本量与条件构成不同,该对比仅作描述性。</figcaption>')
A('</figure>')
A('<h3>F.3 判定组合与词元分布</h3>')
A('<p>图 F1(b) 与表 F2 将 336 条续写按“学生是否答对 × 教师是否答对”四组合分桶,报告桶内逐词元 top-16 熵均值。需要说明两处口径:配对在(题目, 条件)内按采样序号对齐,而两模型的 8 次采样相互独立,故配对具有分布意义而非一一对应;教师分布整体比学生锐(表 F1),且各桶的条件构成不同,因此四组合间的熵差是描述性对照而非因果证据。图 F2 给出一个具体实例:db3e2e94 在错误 50% 前缀上教师 8/8 纠对、学生 0/8,逐词元 top-16 迷你条显示推理决策点处的分布质量;窗口取各自续写中熵最高的 9 个连续词元,不代表两行时间对齐。</p>')
A('<div class="table-wrap"><table><thead><tr><th>判定组合(学生, 教师)</th><th class="num">配对样本数</th><th class="num">top-16 熵均值 (nat)</th></tr></thead><tbody>')
A('<tr><td>都对</td><td class="num">37</td><td class="num">0.111</td></tr>')
A('<tr><td>学生错 · 教师对</td><td class="num">47</td><td class="num">0.123</td></tr>')
A('<tr><td>学生对 · 教师错</td><td class="num">16</td><td class="num">0.085</td></tr>')
A('<tr><td>都错</td><td class="num">68</td><td class="num">0.133</td></tr>')
A('</tbody><caption>表 F2 · 四组合下的 top-16 熵(全部 336 条续写配对)。桶内样本量不均等,且教师整体分布更锐,该表仅作方向性参照。</caption></table></div>')
A('<figure class="fig-compact" style="max-width:880px;margin-left:auto;margin-right:auto"><img alt="Token trace with per-token top-16 bars" src="data:image/png;base64,'+F3+'">')
A('<figcaption><b>图 F2 · 词元轨迹与逐词元 top-16 分布(db3e2e94,错误 50% 前缀)。</b>同一错误前缀下,教师续写被判定为正确(上)、学生续写被判错(下);细条为对应位置的 top-16 概率质量,彩色段为实际采样词元、灰段为其余备选质量。两行各为 8 次采样中的一条真实续写,窗口取各自最高熵区段。</figcaption>')
A('</figure>')
A('<h3>F.4 与正文的关系及局限</h3>')
A('<p>本附录的采样协议使正文 图 2(b) 的状态级结论可在 3 题上复现其方向,并首次公开逐词元 top-16 工件;但它不替代正文基于更大匹配题集合的显著性统计。局限有四:其一,每档仅 3 题×8 次,样本量小,一切档间差均为方向性观察;其二,正确前缀对照档中教师答对率低于其题目根状态,疑为正确解截断点不自然所致,故不据该档作能力主张;其三,配对按采样序号而非时间对齐;其四,熵为 top-16 条件化熵,不等于全词表熵。</p>')
block='\n'.join(new)
lines[si:ei]=block.split('\n')
open(P,'w',encoding='utf-8').write('\n'.join(lines))
print('written; new Appendix F lines',si+1,'..',si+len(block.split('\n')))
