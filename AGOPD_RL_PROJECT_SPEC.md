# AGOPD-RL：Advantage-Guided On-Policy Distillation for Compute-Efficient Reasoning RL

> 本文档用于交给 Codex / Coding Agent 作为项目级实现规范。目标不是快速拼出一个可运行 Demo，而是在 **4×A100** 约束下，构建一个可复现、可评测、可做 Ablation、具备明确研究问题和工程闭环的 **RL + OPD 后训练项目**。

---

## 0. Codex 执行原则

Codex 在实现本项目时必须遵循以下原则：

1. **先跑通 baseline，再改算法。** 不允许一开始同时实现 GRPO、OPD、Gate、Budget、异步 Teacher serving。
2. **优先复用成熟框架。** 默认使用 `verl` 作为 RL/Post-training 主框架，rollout 优先使用 `vLLM` 或 `SGLang`；不要自研 PPO/GRPO/FSDP 调度器。
3. **核心创新必须独立封装。** `Advantage Gate`、`Disagreement Gate`、`Teacher Budget`、`OPD Loss` 必须是独立模块，可通过 config 开关启用/关闭。
4. **所有实验都必须可复现。** 随机种子、模型版本、数据版本、训练超参、commit hash、环境依赖都要落盘。
5. **所有关键算法行为必须可观测。** 不能只记录 train reward；必须记录 KL、entropy、reward std、advantage、gate activation ratio、teacher token ratio、teacher/student disagreement、GPU utilization 等。
6. **不为“架构完整”而过度设计。** 两天内优先完成 V1/V2；V3 只在前两层稳定后实现。
7. **严禁隐藏失败。** NaN、reward collapse、entropy collapse、zero-variance group、OOM、rollout parser error 都必须显式记录并进入实验报告。
8. **不要把 4×A100 描述为 frontier-scale training。** 项目定位是：在有限算力下复现工业后训练范式，并验证一种 compute-efficient selective OPD-RL 方法。

---

# 1. 项目目标

## 1.1 项目名称

**AGOPD-RL**

全称：

**Advantage-Guided On-Policy Distillation for Compute-Efficient Reasoning RL**

中文：

**基于优势引导的计算高效 OPD-RL 推理模型后训练框架**

---

## 1.2 要解决的问题

标准 Outcome-RL / GRPO 的主要问题：

- Reward 通常只在完整 trajectory 结束后给出；
- long reasoning trajectory 上 credit assignment 粗；
- 同一 group 中大量失败 rollout 只能得到 scalar reward，无法知道具体哪里出错；
- 当 group 全错或 reward variance 很低时，GRPO 学习信号变弱。

标准 On-Policy Distillation（OPD）的主要问题：

- Teacher supervision 很密集，但计算成本高；
- 对已经表现良好的 trajectory 继续蒸馏可能是冗余监督；
- Teacher 与 Student 分布差异过大时可能产生 harmful supervision；
- Token 级无差别蒸馏可能把 Student 拉回 Teacher policy，削弱 Student 自身探索出的有效策略。

本项目的核心问题定义为：

> **能否利用 GRPO 自己产生的 group-relative advantage，动态决定哪些 on-policy trajectories / states 真正需要 Teacher，并在有限 Teacher Compute Budget 下，仅对高价值状态进行 OPD，从而提高 Performance / Teacher Compute？**

---

# 2. 核心研究假设

至少验证以下 4 个假设。

## H1：失败程度与 Teacher 需求相关

不是所有 rollout 都需要 Teacher。

对于 group 中表现优于平均水平的 trajectory：

```text
A_i >= 0
```

优先让 RL 自己强化，不进行 OPD。

对于表现低于 group 平均的 trajectory：

```text
A_i < 0
```

才进入 Teacher 候选集合。

---

## H2：Binary Failure Gate 太粗，Advantage Weight 更合理

Binary Gate：

```text
reward == 0 -> distill
reward > 0  -> skip
```

升级为：

```math
w_i^{fail} = ReLU(-A_i)
```

其中 `A_i` 是 GRPO group-relative advantage。

意义：

- 轻微失败 -> 小权重 OPD；
- 严重失败 -> 大权重 OPD；
- 成功 trajectory -> 不蒸馏。

---

## H3：Teacher–Student disagreement 不是越大越好

定义 Student / Teacher 在 Top-K token support 上的分布差异：

```math
d_{i,t} = JS(\pi_S^K(\cdot|s_{i,t}), \pi_T^K(\cdot|s_{i,t}))
```

或者第一版先使用 Top-K overlap：

```math
Overlap_{i,t} = \frac{|S_{i,t}^K \cap T_{i,t}^K|}{|S_{i,t}^K \cup T_{i,t}^K|}
```

假设：

- disagreement 太低：Teacher 没提供新信息；
- disagreement 适中：最值得蒸馏；
- disagreement 太高：可能已经进入 Teacher–Student policy mismatch 区域。

因此定义 Learnable Disagreement Region：

```math
g(d) = 1[\tau_{low} < d < \tau_{high}]
```

第一版允许先只实现单阈值：

```math
g(d) = 1[d > \tau]
```

但最终实验必须说明采用双阈值还是单阈值，以及原因。

---

## H4：Selective OPD 可以提高 Teacher Compute Efficiency

最终不仅比较 Accuracy，还比较：

```math
TeacherEfficiency = \frac{\Delta TaskPerformance}{TeacherGPUHours}
```

以及：

```math
TeacherTokenRatio = \frac{TeacherScoredTokens}{AllStudentGeneratedTokens}
```

目标不是只追求最高 Accuracy，而是证明：

> 在显著减少 Teacher token / GPU hour 的情况下，Selective OPD 能达到或超过 Vanilla OPD 的效果。

---

# 3. 方法定义

## 3.1 GRPO 部分

对于同一个问题 `q`，Student policy 产生 `G` 条 rollout：

```text
τ_1, τ_2, ..., τ_G
```

每条 trajectory 获得 reward：

```text
R_1, R_2, ..., R_G
```

Group-relative advantage：

```math
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}
```

其中：

```math
\mu_R = \frac{1}{G}\sum_i R_i
```

```math
\sigma_R = std(R_1, ..., R_G)
```

若 group reward variance 过低：

```text
σ_R < eps
```

则必须进入 zero-variance handling：

- GRPO advantage 置零或按框架默认稳定策略处理；
- 记录 `zero_variance_group_rate`；
- 这类 group 仍可进入 OPD candidate 流程，但不能伪造 advantage。

---

## 3.2 Advantage-Gated OPD

trajectory-level failure weight：

```math
w_i^{fail} = clamp(ReLU(-A_i), 0, w_{max})
```

默认：

```text
w_max = 2.0
```

允许 config 覆盖。

如果：

```text
A_i >= 0
```

则：

```text
teacher_candidate = false
```

如果：

```text
A_i < 0
```

则进入 state/token-level selection。

---

## 3.3 Disagreement Gate

候选 token/state 上，Teacher 返回 Top-K logprobs。

推荐第一版：

```text
K = 16
```

计算：

- `js_divergence_topk`
- `topk_overlap`
- `student_entropy`

至少保存其中两个用于分析。

最终 token weight：

```math
w_{i,t} = w_i^{fail} \cdot g(d_{i,t})
```

如果实现 Teacher Budget，则再乘：

```math
b_{i,t} \in \{0,1\}
```

最终：

```math
w_{i,t} = w_i^{fail} \cdot g(d_{i,t}) \cdot b_{i,t}
```

---

## 3.4 OPD Loss

默认只在 Student 自己生成的 token 上计算，不在 environment / prompt token 上计算。

第一版推荐支持两种 KL：

```text
forward_kl
reverse_kl
```

默认：

```text
reverse_kl
```

但必须通过 config 切换。

示意：

```math
L_{OPD}
=
\frac{1}{N}
\sum_{i,t}
w_{i,t}
D_{KL}
(\pi_S^K(\cdot|s_{i,t}) || \pi_T^K(\cdot|s_{i,t}))
```

---

## 3.5 总 Loss

```math
L_{total}
=
L_{GRPO}
+
\lambda_{opd} L_{OPD}
+
\beta KL(\pi_\theta || \pi_{ref})
```

默认第一轮实验：

```yaml
lambda_opd: 0.1
beta_kl: framework_default
```

不要在第一天同时扫大量超参。

---

# 4. V1 / V2 / V3 范围

## V1：必须完成

**GRPO + Advantage-Gated OPD**

只做：

```text
A_i < 0 -> Teacher candidate
OPD weight = ReLU(-A_i)
```

不做 disagreement，不做 budget。

V1 成功标准：

- 能完整训练；
- OPD loss 非零；
- positive/negative advantage 分流正确；
- 与 Vanilla GRPO / Vanilla OPD 都能比较；
- 无明显 NaN / entropy collapse。

---

## V2：必须争取完成

**Advantage Gate + Disagreement Gate**

增加：

```text
Teacher–Student Top-K disagreement
```

至少实现：

- top-k overlap；
- JS divergence 或近似 divergence；
- low / mid / high disagreement histogram；
- gate activation ratio。

---

## V3：有时间再做

**Budgeted Selective OPD**

定义 Teaching Value：

```math
S_{i,t}
=
ReLU(-A_i)
\cdot
f(d_{i,t})
\cdot
H(\pi_S)
```

在一个 batch / rollout group 内，只选择 Top-B% token 请求或应用 Teacher supervision。

默认预算：

```text
B = 30%
```

需要支持：

```yaml
teacher_budget_ratio: 1.0   # Vanilla OPD
teacher_budget_ratio: 0.5
teacher_budget_ratio: 0.3
```

---

# 5. 训练数据与 Benchmark

## 5.1 推荐训练数据

默认：

```text
DAPO-Math-17K
```

原因：

- 标准公开 RL reasoning 数据；
- Rule-based verifier 容易；
- 避免 LLM Judge 干扰 Reward；
- 适合做 GRPO / OPD ablation；
- 可和已有 reasoning RL 项目横向比较。

第一阶段不要全量训练。

建议：

```text
smoke_test: 64 samples
quick_train: 1k~2k
main_train: 3k~8k
```

具体规模根据 rollout throughput 动态决定。

---

## 5.2 Evaluation

至少：

```text
MATH-500
AIME 2024
AIME 2025
AMC 2023
```

如果时间不够，最小评测集：

```text
MATH-500 + AIME24
```

必须区分：

```text
train
validation
held-out evaluation
```

禁止训练数据泄露到正式 benchmark。

---

# 6. 模型配置

## 6.1 推荐主配置

默认假设：A100 80GB。

### Student

```text
Qwen3-1.7B
```

### Teacher

优先：

```text
Qwen3-4B
```

如果显存与吞吐允许，再尝试：

```text
Qwen3-8B
```

不要第一天直接上 14B+ Teacher。

---

## 6.2 如果是 A100 40GB

优先：

```text
Student: Qwen3-1.7B
Teacher: Qwen3-4B
```

并降低：

- rollout batch；
- max response length；
- teacher scoring batch；
- group size。

---

# 7. 4×A100 GPU 拓扑

目标：先稳定，再追 throughput。

## Phase A：最稳方案

```text
GPU 0-2
  Student Actor / GRPO Training
  FSDP

GPU 3
  Teacher Inference Server
```

优点：

- Teacher 和 Actor 物理隔离；
- 实现简单；
- Debug 清晰。

缺点：

- Student 只有 3 卡训练；
- rollout 和 update 的利用率未必最佳。

---

## Phase B：稳定后优化

如果 verl 支持并且显存足够：

```text
GPU 0-3
  Student Actor / Rollout / Update

Teacher
  通过阶段式加载 / colocate / offload 方案运行
```

但不要在 V1 阶段追求复杂 colocate。

---

# 8. 系统架构

```text
                           ┌─────────────────┐
                           │ Dataset Loader  │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Student Policy  │
                           │    Qwen3-*      │
                           └────────┬────────┘
                                    │
                             G rollouts / q
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Rollout Engine │
                           │ vLLM / SGLang   │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Rule Verifier   │
                           │  Reward R_i     │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ GRPO Advantage  │
                           │      A_i        │
                           └────────┬────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                 A_i >= 0                         A_i < 0
                    │                                │
                    ▼                                ▼
                RL only                       Teacher Candidate
                                                     │
                                                     ▼
                                           ┌─────────────────┐
                                           │ Teacher Scorer  │
                                           │  Top-K logits   │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ Disagreement    │
                                           │ / Budget Gate   │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │    OPD Loss     │
                                           └────────┬────────┘
                                                    │
                         ┌──────────────────────────┘
                         ▼
                 ┌─────────────────┐
                 │ Combined Loss   │
                 │ GRPO + OPD + KL │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Optimizer Step  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Metrics / W&B   │
                 └─────────────────┘
```

---

# 9. 推荐仓库目录

```text
agopd-rl/
├── README.md
├── PROJECT_SPEC.md
├── pyproject.toml
├── requirements.lock
├── .env.example
├── Makefile
│
├── configs/
│   ├── base.yaml
│   ├── grpo.yaml
│   ├── opd.yaml
│   ├── hybrid.yaml
│   ├── agopd_v1.yaml
│   ├── agopd_v2.yaml
│   └── agopd_v3_budget.yaml
│
├── src/
│   └── agopd/
│       ├── __init__.py
│       ├── data/
│       │   ├── loader.py
│       │   ├── preprocess.py
│       │   └── splits.py
│       │
│       ├── rollout/
│       │   ├── engine.py
│       │   ├── trajectory.py
│       │   └── masks.py
│       │
│       ├── reward/
│       │   ├── verifier.py
│       │   ├── math_reward.py
│       │   └── normalization.py
│       │
│       ├── grpo/
│       │   ├── advantage.py
│       │   └── diagnostics.py
│       │
│       ├── teacher/
│       │   ├── client.py
│       │   ├── scorer.py
│       │   ├── topk.py
│       │   └── cache.py
│       │
│       ├── gating/
│       │   ├── advantage_gate.py
│       │   ├── disagreement_gate.py
│       │   ├── budget_gate.py
│       │   └── teaching_value.py
│       │
│       ├── opd/
│       │   ├── loss.py
│       │   ├── kl.py
│       │   └── token_alignment.py
│       │
│       ├── trainer/
│       │   ├── verl_adapter.py
│       │   ├── hybrid_trainer.py
│       │   └── callbacks.py
│       │
│       ├── eval/
│       │   ├── benchmarks.py
│       │   ├── pass_at_k.py
│       │   └── efficiency.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── seed.py
│           ├── config.py
│           └── distributed.py
│
├── scripts/
│   ├── prepare_data.sh
│   ├── eval_base.sh
│   ├── train_grpo.sh
│   ├── train_opd.sh
│   ├── train_agopd_v1.sh
│   ├── train_agopd_v2.sh
│   ├── train_agopd_v3.sh
│   └── eval_all.sh
│
├── tests/
│   ├── test_advantage_gate.py
│   ├── test_disagreement_gate.py
│   ├── test_opd_loss.py
│   ├── test_token_mask.py
│   ├── test_zero_variance_group.py
│   └── test_teacher_budget.py
│
├── experiments/
│   ├── manifests/
│   ├── results/
│   └── plots/
│
├── docs/
│   ├── method.md
│   ├── infra.md
│   ├── experiments.md
│   └── failure_cases.md
│
└── docker/
    ├── Dockerfile
    └── compose.yaml
```

---

# 10. 关键数据结构

## 10.1 Trajectory

```python
@dataclass
class Trajectory:
    prompt_id: str
    prompt: str
    response_tokens: list[int]
    response_text: str

    reward: float
    advantage: float | None

    policy_logprobs: list[float]
    ref_logprobs: list[float] | None

    loss_mask: list[int]

    teacher_candidate: bool = False
    teacher_topk_ids: list[list[int]] | None = None
    teacher_topk_logprobs: list[list[float]] | None = None

    disagreement: list[float] | None = None
    opd_weights: list[float] | None = None
```

注意：真实 verl tensor/batch 表示可不同，但逻辑字段必须可追踪。

---

## 10.2 GateDecision

```python
@dataclass
class GateDecision:
    trajectory_weight: float
    token_mask: list[bool]
    token_weights: list[float]

    reason: str
    budget_used: int = 0
```

---

# 11. Config 设计

建议：

```yaml
project:
  name: agopd-rl
  seed: 42

model:
  student: Qwen/Qwen3-1.7B
  teacher: Qwen/Qwen3-4B
  reference: null

rollout:
  group_size: 8
  temperature: 1.0
  top_p: 0.95
  max_response_tokens: 4096

reward:
  type: rule_math

rl:
  algorithm: grpo
  kl_coef: 0.0

opd:
  enabled: true
  lambda_opd: 0.1
  kl_type: reverse
  top_k: 16

advantage_gate:
  enabled: true
  threshold: 0.0
  max_weight: 2.0

disagreement_gate:
  enabled: false
  metric: js
  tau_low: 0.05
  tau_high: 0.50

teacher_budget:
  enabled: false
  ratio: 0.30
  score: advantage_x_disagreement_x_entropy

logging:
  wandb: true
  save_trajectory_samples: true
```

所有 threshold 必须来自 config，不允许硬编码。

---

# 12. Teacher Scoring 设计

## 12.1 不传 full-vocab logits

只返回：

```text
top-k token ids
+
top-k logprobs
```

默认：

```text
K = 16
```

目的：

- 降低显存；
- 降低通信量；
- 降低 cache 体积；
- 让 OPD 适合有限 GPU。

---

## 12.2 Teacher Cache

cache key 至少包含：

```text
teacher_model_id
student_prefix_hash
position
K
```

避免重复评分。

V1 可以只做内存 cache；不要第一天上 Redis。

---

# 13. Reward / Verifier

第一版必须使用 deterministic / rule-based verifier。

不要第一版引入：

```text
LLM Judge
Reward Model
Process Reward Model
```

原因：

- 会让实验因变量不清晰；
- 增加噪声；
- 两天内无法判断是 Gate 有效还是 Judge 有效。

Reward 第一版：

```text
final_answer_correct = 1
wrong = 0
```

如果答案解析支持 partial score，可额外保留但不要影响第一版主实验。

---

# 14. 必须记录的 Metrics

## 14.1 RL Health

```text
train/reward_mean
train/reward_std
train/positive_rollout_rate
train/zero_reward_group_rate
train/zero_variance_group_rate
train/kl_to_ref
train/entropy
train/response_length_mean
train/response_length_p95
train/grad_norm
```

---

## 14.2 OPD / Gate

```text
opd/loss
opd/activated_trajectory_ratio
opd/activated_token_ratio
opd/teacher_candidate_ratio
opd/teacher_token_ratio
opd/teacher_gpu_seconds
opd/topk_overlap_mean
opd/js_divergence_mean
opd/student_entropy_selected
opd/student_entropy_skipped
opd/opd_weight_mean
```

---

## 14.3 Evaluation

```text
eval/accuracy
eval/pass@1
eval/pass@8
eval/avg_response_tokens
eval/teacher_gpu_hours
eval/performance_per_teacher_gpu_hour
```

如果 benchmark 支持：

```text
pass@16
```

也记录，用于判断 distribution sharpening。

---

# 15. 必做实验矩阵

最低必须完成以下 5 组：

| ID | Method | GRPO | OPD | Advantage Gate | Disagreement Gate |
|---|---|---:|---:|---:|---:|
| E0 | Base / SFT | 0 | 0 | 0 | 0 |
| E1 | GRPO | 1 | 0 | 0 | 0 |
| E2 | Vanilla OPD | 0 or fixed-policy setup | 1 | 0 | 0 |
| E3 | GRPO + OPD | 1 | 1 | 0 | 0 |
| E4 | **AGOPD-V1** | 1 | 1 | 1 | 0 |
| E5 | **AGOPD-V2** | 1 | 1 | 1 | 1 |

如果时间允许：

| ID | Method | Teacher Budget |
|---|---|---:|
| E6 | AGOPD-V3 | 50% |
| E7 | AGOPD-V3 | 30% |

---

# 16. Ablation 必须回答的问题

项目报告必须直接回答：

### RQ1

```text
GRPO 相比 Base 是否真正提高任务表现？
```

### RQ2

```text
Vanilla OPD 是否比纯 GRPO 提供增益？
```

### RQ3

```text
简单 GRPO + OPD 是否比单独两者更好？
```

### RQ4

```text
Advantage Gate 是否优于无差别 OPD？
```

### RQ5

```text
Disagreement Gate 是否减少 harmful / redundant teacher supervision？
```

### RQ6

```text
能否在 30%-50% Teacher token budget 下保持大部分性能？
```

---

# 17. 两天执行顺序

## Day 1 上午：环境与 baseline

1. 建 repo / config / logging；
2. 跑 base eval；
3. verl GRPO smoke test；
4. 64 sample rollout；
5. 验证 reward / advantage / masks 正确；
6. 小规模 GRPO train。

**Gate：如果 GRPO baseline 没跑通，不进入 OPD。**

---

## Day 1 下午：Vanilla OPD

1. Teacher scorer；
2. Top-K logprob；
3. OPD loss；
4. response token mask；
5. 64 sample unit/integration test；
6. 跑 Vanilla OPD baseline。

**Gate：必须看到 OPD loss、Teacher top-k 和 Student top-k 合理。**

---

## Day 1 晚上：AGOPD V1

实现：

```text
negative advantage -> teacher candidate
```

并记录：

```text
activated trajectory ratio
opd weight distribution
teacher token ratio
```

先跑 100~300 step 小实验。

---

## Day 2 上午：AGOPD V2

增加：

```text
Top-K overlap / JS disagreement
```

完成：

```text
Vanilla OPD
vs
Advantage Gate
vs
Advantage + Disagreement Gate
```

---

## Day 2 下午：统一评测

在固定：

```text
same model
same dataset
same verifier
same rollout budget
same eval decoding
```

下统一跑：

```text
MATH-500
AIME24
```

如果时间足够加：

```text
AIME25
AMC23
```

---

## Day 2 晚上：报告和仓库整理

生成：

```text
README
method.md
experiments.md
failure_cases.md
```

README 首页必须包含：

1. 一句话 Problem；
2. 一张架构图；
3. 一个主结果表；
4. 一个 Accuracy / Teacher Compute 图；
5. 复现命令。

---

# 18. 测试要求

必须有单元测试：

## Advantage Gate

```text
A > 0 -> weight == 0
A == 0 -> weight == 0
A < 0 -> weight > 0
```

## Weight Clamp

```text
large negative A -> <= max_weight
```

## Disagreement Gate

测试：

```text
identical distribution
moderate disagreement
extreme disagreement
```

## Token Mask

Prompt / padding / environment token 不进入 OPD loss。

## Zero Variance Group

```text
[0,0,0,0]
[1,1,1,1]
```

不得产生 NaN。

## Teacher Budget

Teacher token 数不得超过 config budget。

---

# 19. Failure Modes

Codex 必须对以下失败进行检测与日志：

## Reward Collapse

```text
reward_mean 长期下降
```

## Entropy Collapse

```text
entropy 快速趋近低值
```

## Response Length Explosion

```text
response tokens 持续膨胀
```

## Teacher Domination

```text
OPD loss 远大于 GRPO loss
Student policy 被 Teacher 完全拉走
```

## Gate Collapse

```text
activated_token_ratio -> 0
```

或者：

```text
activated_token_ratio -> 1
```

都说明 Gate 失去选择性。

## Zero Reward Groups

如果大量 group：

```text
all reward = 0
```

必须报告；不要假装 GRPO 正常。

---

# 20. 代码实现优先级

按照以下顺序：

```text
P0
- verl baseline
- deterministic verifier
- trajectory logging
- advantage extraction

P1
- Teacher scorer
- Top-K logits
- Vanilla OPD
- token mask

P2
- Advantage Gate
- AGOPD-V1

P3
- disagreement metric
- AGOPD-V2

P4
- Teacher Budget
- AGOPD-V3

P5
- throughput optimization
- cache
- async scoring
```

不要跨级同时开发。

---

# 21. 不做的事情

两天版本明确不做：

```text
Real-Web Agent RL
Search Agent Environment
PRM
Reward Model Training
RLAIF
Multi-Agent RL
14B+ Student
全量数据长周期训练
自研 CUDA Kernel
自研 RL Framework
复杂 Kubernetes / Slurm 平台
```

原因：这些会显著提高 integration risk，但不提高当前研究问题的可信度。

---

# 22. Codex 第一次执行时应该做什么

Codex 不要直接开始写所有代码。

第一阶段必须先完成：

1. 检查当前机器：GPU 型号、显存、CUDA、Driver、Python；
2. 检查 `verl / vLLM / SGLang` 版本兼容；
3. 初始化 repo；
4. 生成 `ENVIRONMENT_REPORT.md`；
5. 实现最小 dataset loader；
6. 跑 Base 模型 8~32 sample inference；
7. 跑 GRPO smoke test；
8. 只有 smoke test 通过后，再开始 OPD。

如果任何框架 API 与本规范不同：

> 不允许为了“严格按文档”重写框架。应封装 adapter，并在 `docs/implementation_notes.md` 说明当前版本差异。

---

# 23. 最终验收标准

## 工程验收

- [ ] 4×A100 可启动训练；
- [ ] Base / GRPO / OPD / AGOPD 使用同一套 eval；
- [ ] 训练 config 全部落盘；
- [ ] 可以从单一 shell 命令复现实验；
- [ ] 无 NaN；
- [ ] Teacher top-k scoring 可正确返回；
- [ ] OPD mask 不覆盖 prompt / padding；
- [ ] Gate 逻辑有 unit test；
- [ ] W&B / TensorBoard 指标完整；
- [ ] checkpoint 可恢复。

## 研究验收

至少证明以下之一：

### 成功条件 A

```text
AGOPD-V1/V2 > GRPO
且 Teacher token < Vanilla OPD
```

### 成功条件 B

```text
AGOPD 与 Vanilla OPD performance 接近
但 Teacher token / GPU hour 明显下降
```

### 成功条件 C

即使最终效果未超过 baseline，也必须形成清楚的 negative result：

```text
在哪些 advantage/disagreement 区间 Teacher supervision 有帮助？
在哪些区间有害？
```

这也可以成为有价值的分析结果。

---

# 24. 最终项目对外表述

不要写：

> 使用 4×A100 完成大规模 RL 训练。

推荐写：

> **基于 verl 构建分布式 On-Policy 后训练闭环，在 4×A100 上完成 Qwen 系列模型 GRPO / OPD 对照实验；针对 Outcome-RL 信号稀疏与 Vanilla OPD Teacher supervision 冗余问题，设计 Advantage-Guided Selective OPD，以 GRPO group-relative advantage 动态筛选失败轨迹，并结合 Teacher–Student disagreement 约束高价值蒸馏状态，在固定 Teacher Compute Budget 下优化 reasoning performance / teacher cost。**

---

# 25. 最重要的实现判断

Codex 在所有设计决策中优先遵循：

```text
可复现 > 花哨
baseline 完整 > 模块数量
Ablation 清晰 > 单一最好结果
训练稳定 > 模型规模
Teacher Compute Efficiency > 单纯 Accuracy
```

项目成功的标准不是“训练过一个大模型”，而是：

> **能够明确证明：为什么要引入 OPD，什么时候应该使用 Teacher，什么时候不应该使用，以及这种选择机制对效果、训练稳定性和 Teacher Compute 有什么量化影响。**

