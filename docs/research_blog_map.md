# Research Blog Map:小模型数学推理 RL 的系统性研究

**Working Title**: 从梯度饥饿到数据条件化训练 — 1.7B 数学推理的 RL/蒸馏全链路研究
(备选:On the Difficulty of Training Small Models for Math: Gradient Starvation, Teacher Mismatch, and Data-Conditioned RL)

**核心主张(One-line Thesis)**:
小模型(1.7B)在数学推理 RL 中的失败不是"容量不足",而是三个可修复的系统性错配:
① 数据分布与能力分布错配(79.6% 题对模型零梯度潜力 → 梯度饥饿)
② 更新通道与任务复杂度错配(LoRA 低秩通道在弱底座上无法承载学习)
③ Teacher 强区与蒸馏作用区错配(在线蒸馏永远作用在 teacher 的弱区)
修复三者后,1.7B 首次实现 +5.08pp 的可靠提升(23.34% → 28.42%)。

---

## 叙事结构(ICLR blog 风格:问题 → 机制 → 修复 → 验证)

### 0. 设定与动机
- 小模型 RL 的现实困境:LoRA 各配置 reward 全平、440 步无增益、full-param 部分有效但不可靠
- 与 4B 的行为差异(4B 能学 1.7B 不能)→ 引出"不是容量,是结构"
- 关键方法学:固定验证集(1024)+ data.seed=42 可比性纪律

### 1. 现象一:梯度饥饿(数据条件化的动机)
- **观测**:EGR(有效组率)仅 ~20%,80% rollout 组零学习信号
- **机制测量**:全量 16164 题 p̂ 普查(n=8)→ 能力分布断层:
  - hard(p̂<0.125):79.6%;frontier:17%;easy:3.4%
  - 题重复率 0%(seed42 顺序采样每 200 步只覆盖 15% 题)
- **概念化**:GRPO 只在 p̂∈[0.2,0.7] 区间有有效梯度;两极分布(0 或 1)都是梯度荒漠
- 图表:能力直方图(Capability Frontier)、EGR vs 步数、score−p̂ 校准方法学(扣除题难度)

### 2. 现象二:蒸馏的熵膨胀与 Teacher State Mismatch
- **观测**:forward KL 蒸馏熵 0.13→0.32(分布摊平);hard CE 熵稳定但质疑其损能
- **机制测量**(本地双模型熵诊断,1024 轨迹):
  - wrong 轨迹上 student 熵 0.35 vs correct 0.26(学生自知迷茫)
  - teacher 在 student 陌生状态熵 ≥ student(teacher 无信息可教)
- **关键纠正**:teacher"分布宽"是陌生状态的表现,不是固有校准差(避免错误解读的教训)
- **方法学贡献**:topk 熵近似、H_S/H_T gap 作为 gate 设计依据

### 3. 基础设施(支撑方法,单独章节或附录)
- verl v0.8 深度改造:Teacher-after-Advantage 流水线(异步 teacher scoring 重叠)
- 拓扑:3+1 hybrid(FSDP3+3 replica+1 teacher)、CUDA Graph、2+2/3+1 对比(-27% step time)
- 工程坑与修复(可作"reproducibility lessons"):
  - entropy 分块 patch(full-param 3+1 OOM 墙)
  - DTensor checkpoint 合并(FSDP2 → HF)
  - rollout/TB 目录隔离、resume 污染(auto resume 陷阱)
  - vLLM per-request seed + FlashInfer 117 tok/s 陷阱
  - prompt 双重模板 artifact(审计结论反转的教训)
  - 多卡 GPU 利用率纪律(NCCL 217、本地/云端分工)

### 4. 方法一:数据条件化(Dataset Conditioning)
- 普查 → 三桶 → **comfort 桶(p̂≥0.25)** 作为 GRPO 训练集
- n↑(4→8,EGR 60→80%)与 batch↓(同预算)
- 为什么 0.125 边缘档有害(初训舒适区)
- 结果:EGR 20%→80%、grad 信号 1.6-2.5×

### 5. 方法二:更新通道(Update Channel)
- LoRA(r8/r16 × lr 1e-6/1e-4/3e-4)全扫:lr 3× 无效的强证据(clip 恒 0.002)
- full-param 切换:grad 30×、首次真实学习信号(score−p̂ +5.8pp)
- **稳定化问题**:full-param lr3e-4 步 2 崩、lr1e-5 步 20 崩(冗长化漂移 → 全组 allzero → 信号消失)
- 解:lr1e-6 + KL0.05(锚定 base 防漂移)— 184 步稳定
- 概念:策略漂移 → 输出冗长化 → 截断 → 信号死亡 的退化环

### 6. 方法三:AGOPD(Advantage-Gated On-Policy Distillation)
- 设计:advantage gate(zero-group 才蒸)+ teacher competence gate + taa 流水线
- hard CE vs 温度锐化软蒸馏(softT07)对照
- **gate 价值直接证明**:AGOPD 27.05% vs 无 gate RL+OPD 24.12%(+2.9pp)
- gate 富集验证:competence 72%(gate 通过区)vs 32%(全体)

### 7. 方法四:Teacher 选择与错配分析
- 三候选筛选(4B-GRPO/8B base/OpenMath-7B × comfort 300 题):
  - 8B/7B base 未在 DAPO 分布训练 → p_T 仅 0.078/0.084(< 4B-GRPO 0.152)
- **分布互补性发现**:p_S vs p_T 弱正相关(+0.14)、"学生弱 teacher 强"题 = 0
  → teacher 强区(medium-hard)与学生采样区(comfort)不重叠
- 推论:在线蒸馏作用区恒为 teacher 弱区 → 无收益的结构性解释

### 8. 结果与消融矩阵(核心表格)
- 1.7B 固定 1024 全对照表(base/LoRA 各配置/full-param/四方法/AGOPD/SFT 冷启动)
- 关键数字:GRPO 28.42%(+5.08pp)、AGOPD 27.05%、OPD 25.68%、RL+OPD 24.12%
- 四方法严格可比(同桶/同预算/同 seed/同配置)

### 9. 输出质量审计与多阶段(阶段 2)
- 质量审计方法论(200 题池、分层 correct/wrong、prompt artifact 教训)
- 发现:能力圈内格式规范、圈外退化(截断/绕圈)
- **阶段 2 设计**:RFT(全量 temp1.0 采样 + 清洗 + LoRA r32 SFT)→ hard 重分层 → AGOPD on new-frontier
- (结果待出 — 地图预留)

### 10. 讨论与展望
- 统一框架:三个错配(数据/通道/teacher)与三个修复
- 通用性:对更大模型/其他推理任务的预测
- 局限:1.7B 单点、DAPO 单数据集、LoRA vs full-param 的算力权衡

---

## 需要的补充素材(待办)
1. [ ] 全量 RFT 采样 + hard 重分层结果(阶段 2 数字)
2. [ ] 4B 时代的对应表(Run C 42.48%、AGOPD-400 32.42% 等)— 跨规模一致性
3. [ ] temp 对照(0.3/0.6/1.0/1.5)完整数据
4. [ ] 拓扑对比图(2+2 vs 3+1 step time)
5. [ ] entropy 诊断的正式图(wrong/correct × H_S/H_T 分层)

## 写作策略
- 主叙事:三个错配(可检验机制)→ 三个修复(消融验证)→ 组合结果
- 每个机制配"测量方法学"小节(score−p̂、EGR、熵 gap)— 强调可复现
- Lessons/坑单列(工程可复现性价值高)

---

## 追加维度:跨模型尺寸对比(4/8B × 1.7/4B 双 block 研究)

**设计**:两个尺寸 block 用同一协议并行研究 —
- Block A:4B student + 8B teacher(云端 4×A100)
- Block B:1.7B student + 4B-GRPO teacher(本地 2×4090)
→ 模型尺寸成为受控变量,回答"启动差异与 teacher 差异随尺寸如何变化"

**关键对比表(地图核心补充)**:

| 维度 | 4B block | 1.7B block | 差异含义 |
|---|---|---|---|
| base 能力 | 33.01% | 23.34% | 底座 → p̂ 分布形状 → 饥饿程度 |
| **hard 题占比(普查)** | **~40%**(R89 1000 题,多 pilot 33-40% 一致) | **79.6%**(全量 16164) | 饥饿程度随底座能力剧变 |
| E1-E3 60 步(LoRA r8 lr1e-6) | 30.37/30.37/30.37%(负) | 24.41/平(基本无效) | 旧通道两尺寸均无效 |
| **Run A/B/C 强度扫描** | **r16+lr1e-4 60 步 = 42.48%(+9.47pp)** | 同配置(r16 lr1e-4/3e-4)仍无信号 | **4B 的 LoRA 通道可被 rank/lr 修复;1.7B 不可 → 通道-底座耦合** |
| LoRA 长跑 | AGOPD-400(r8 lr1e-6) 32.42% | **440 步(r8 lr1e-6) 23.93%**(R87) | 弱通道长跑双尺寸均无增益 |
| full-param 50 步 | 40.23%(+7.22pp) | 25.39%(+2.05pp) | full-param 两尺寸都有效 |
| full-param+数据条件化 | (4B 未做分桶) | **28.42%(+5.08pp)** | 1.7B 的完整修复链;4B 无需分桶(其 LoRA 已够) |
| AGOPD/hardCE | 42.29%(100 步) | 27.05%(184 步) | gate 蒸馏两尺寸均接近纯 GRPO |
| SFT 冷启动 | 39.65%(+6.64pp) | 25.20%(+1.86pp) | 冷启动增益随尺寸差衰减 |
| teacher 熵差 | 8B vs 4B:Δ≈0 | 4B-GRPO vs 1.7B:Δ+0.03 | teacher 相对宽度随尺寸差增大 |
| teacher 分布匹配 | 8B(36.43%)与 4B 接近 | 4B-GRPO 强区在 1.7B 的 hard 区 | teacher 错配在小尺寸对更严重 |

**支撑实验资产(已完备,可直接复用)**:
1. Teacher bake-off(9/2):候选筛选流程(Qwen3-8B / lllyx 4B-GRPO / OpenMath 系列 / own 4B-GRPO)→ 方法学示范
2. E1-E3 × 60 步双 block 矩阵(同协议跨机器)
3. Run A/B/C LoRA 强度扫描(4B:lr 1e-5/1e-4 判定 + Run C 100 步)
4. R86 诊断链(reward 平 → 通道强度 → full-param 验证)
5. 冷启动 v1/v2 × 4B/1.7B(4B +6.64pp vs 1.7B +1.37pp;4B RL 轨迹 vs rejection)
6. AGOPD-400(4B+8B)与 1.7B AGOPD-184 对照
7. 熵诊断双模型对(8B/4B 对 vs 4B/1.7B 对)Δ 差异

**叙事价值**:
- 回答"小模型为什么难训"的**尺寸连续谱**:LoRA 通道有效性、teacher 错配严重度、冷启动增益都随尺寸缩放
- 1.7B 的修复链(full-param/comfort/AGOPD)在 4B 上的一致性验证(42.48% Run C = 4B 的 comfort 类信号?需补 4B 分桶验证)
- teacher 差异:大对(8B→4B)teacher 中性;小对(4B→1.7B)teacher 错配 → **蒸馏对小模型的 teacher 选择更关键**

**待补素材**(修正后):
- [ ] 4B 全量 p̂ 普查(可选强化;R89 1000 题 + pilot 已支撑 ~40%)
- [ ] 阶段 2 结果(RFT SFT + hard AGOPD)
- [ ] softT07 补完(92/184 中断)
