# AGOPD-RL 训练规划与指标体系（2026-09-02 冻结）

本规划基于已闭环的诊断结论制定：
- R86/R87：LoRA rank 8 + lr 1e-6 更新通道过窄是此前 E1-E3/AGOPD 全部无效的根因；
- R88：Run C（Pure GRPO, lr=1e-4, rank=16, 60 步）strict 42.48% 大幅超越 Base 33.01%；
- 因此所有正式实验统一采用 **lr=1e-4 + rank=16 + 3+1 hybrid 拓扑** 配置。

## 1. 统一配置（所有长训练）

| 项 | 值 |
| --- | --- |
| Student | Qwen3-4B (LoRA rank 16, alpha 16) |
| Teacher (OPD/蒸馏) | Qwen3-8B TP1×1 |
| 拓扑 | 3+1 hybrid（FSDP3 + 3 colocated rollout replica + 1 Teacher） |
| batch / n / mini | 12 / 4 / 6（48 seq/step, dynamic bsz） |
| 采样 | 0.6 / 0.95 / 20, 2048 response cap |
| lr | **1e-4** |
| vLLM | CUDA Graph + 32768 batched tokens |
| 步数 | 400（固定预算） |
| 保存 | 每 10 步 checkpoint |
| 数据 | dapo-verl-v1 train / 固定 1024 val |

## 2. 训练矩阵（400 步固定预算, 云端 4×A100）

阶段一 —— 方法基线（不叠加冷启动）：

| 实验 | 方法 | 说明 |
| --- | --- | --- |
| E1-400 | Pure GRPO | Run C 配置的正式延长（历史 60 步 E1 作废） |
| E2-400 | Vanilla OPD-only (use_task_rewards=False) | 新配置下重跑 |
| E3-400 | GRPO + Vanilla OPD | 新配置下重跑 |
| AGOPD-400 | E3 + Advantage Gate (teacher_after_advantage) | 新配置下重跑（现有 400 步是 lr1e-6 旧配置,不可比） |

阶段二 —— 冷启动对照（只叠加在 AGOPD 上）：

| 实验 | 冷启动 SFT 数据 | 训练 |
| --- | --- | --- |
| AGOPD-400-base | 无（Base 直接 RL） | 阶段一已有,作对照 |
| CS-A → AGOPD-400 | 纯 4B rejection 轨迹（offline-RFT 性质,182 条级） | SFT 后 AGOPD |
| CS-B → AGOPD-400 | 4B+8B 混合（4B rejection + 8B verifier-confirmed,50 条级） | SFT 后 AGOPD |
| （对照）CS-A/B → 固定验证 | SFT 后不 RL,直接评估 | 评估 SFT 单独效果 |

两两对比：CS-A vs CS-B 的 **SFT 阶段曲线** + 各自 **AGOPD 后训练曲线**。
数据规模：第一批 coldstart-1000/5000（~1000 条/600 题）足够起步;按迭代扩量。

## 3. 10 步信号门（每个长训练启动规则）

任何长训练开始前,先跑 10 步并检查：

| 判据 | 通过阈值 | 失败动作 |
| --- | --- | --- |
| reward mean (步 6-10) | > 0.36 或末 5 步高于前 5 步 | 停止,检查配置/信号 |
| EGR（有效组率） | > 0.35 | 检查采样/数据 |
| entropy | 未快速塌缩（> 0.05） | 调 KL/采样 |
| distillation loss（E2/E3/AGOPD） | 非零 | 检查 teacher 链路 |
| gate activated_ratio（AGOPD） | > 0.05（有轨迹被激活） | 检查 gate 配置 |

参考：lr=1e-4 配置 10 步 reward mean ≈ 0.39 通过;lr=1e-6 配置 ≈ 0.33 拒绝。

## 4. 指标体系

### 4.1 通用层（已有,逐 step 记录）
- RL：critic/rewards/mean, advantages, actor/entropy, kl_loss, pg_loss, grad_norm
- OPD：actor/distillation/loss, overlap_ratio, student_mass, teacher_mass, overlap_token_advantage
- 输出：response_length/mean, clip_ratio, aborted_ratio, 格式 marker 率
- 性能：timing_s/*, throughput, memory

### 4.2 优越性层（体现 AGOPD / 冷启动方法价值的专用指标）

| 指标 | 定义 | 体现什么 |
| --- | --- | --- |
| **ERSR** | mixed groups / all groups（有效组率） | RL 信号密度;SFT 后 40%→65%+ 是目标 |
| **Teacher Token Ratio** | 实际调 Teacher 的轨迹 token / 全量轨迹 token | AGOPD 的 teacher 算力节省（vs E2/E3 100%） |
| **Learning Signal per GPU-hour** | (有效序列数 × 数据密度) / 训练卡时 | 方法的经济性——核心对比轴 |
| **PTR / NTR** | 固定 1024 val 四象限：B 类答对率 / C 类保留率 | AGOPD gate 是否降低无条件 OPD 的负迁移 |
| **Gate 激活统计** | activated_ratio, 激活轨迹平均 \|adv\|, gate 开销 ms/step | gate 行为健康度 |
| **GRPO-Readiness（SFT 后）** | all-zero↓ mixed↑ all-one 不爆 + p̂ 分布位移 | SFT 是否把 hard 题推入 learnable frontier |
| **Steps-to-signal** | reward 达到 0.40 的首步 | 收敛速度 |
| **SFT 效率** | 固定验证提升 / SFT 数据条数 | cold-start 数据质量（vs 旧 OpenR1 8000 条无效） |

### 4.3 评估协议（固定,每个 checkpoint 统一）
- 1024 条 dapo-verl-v1/val,2048 cap,0.6/0.95/20,seed 42,no-think
- direct-LoRA（--lora-adapter --lora-rank 16）——合并目录不评估（R79/R85 教训）
- 输出：strict / semantic / contract / parse / truncation / length + 四象限 PTR/NTR
- 评估时机：step 10（信号门）+ step 60（Run C 对照）+ step 400（最终）

## 5. 执行顺序（云端 4 卡串行,每实验 ~7h;2026-09-02 用户确认）

1. **AGOPD-400（base,新配置）** —— 先跑主对比臂,根据其结果判断后续方法是否需要调整
2. **E2-400, E3-400** —— 方法与 AGOPD 同预算
3. **E1-400** —— 从 Run C 的 global_step_60 checkpoint **续跑**到 400（配置相同,数据等价）
4. 冷启动数据构建（5000 题全量收齐后）→ SFT（CS-A / CS-B）→ GRPO-Readiness 评估
5. CS-A → AGOPD-400, CS-B → AGOPD-400（放最后）

每个实验启动前确认 10 步信号门;失败即停,不盲跑。
