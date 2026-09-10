# AGOPD-RL

面向小模型数学推理的**在线策略蒸馏(OPD)** 研究代码库:用受控实验说明"教师词元级监督什么时候有用、什么时候有害",并给出可复现的门控方案。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![verl](https://img.shields.io/badge/Trainer-verl%20v0.8-1f6feb)](https://github.com/volcengine/verl)
[![vLLM](https://img.shields.io/badge/Rollout-vLLM-30a2ff)](https://github.com/vllm-project/vllm)
[![LoRA](https://img.shields.io/badge/Adapter-LoRA%20r16-8a5cf6)](https://arxiv.org/abs/2106.09685)
[![License](https://img.shields.io/badge/license-Apache--2.0-2ea44f)](LICENSE)

> 本仓库只包含代码、研究与论文材料,**不含模型权重与训练数据**;所有发布文件已去除云端主机、凭据与个人路径。

## 这个项目解决什么问题

在线策略蒸馏让学生在自己的轨迹上接受教师逐词元监督,常被当作"免费的密集信号"。但在小模型 + 结果奖励 RL 的组合里,它经常**不是零收益,而是负收益**:

- 学生答对的轨迹上也被施加蒸馏,会把正确行为改回教师分布;
- 教师本身可能**弱于**训后学生:是否占优要按**状态**判断,而不是看平均分;
- 无条件、固定系数的蒸馏项会主导梯度,把 RL 变成"对教师做 SFT";
- 后果通常表现为:策略熵下降、回答变长、答案解析率下降、固定评估变差。

本仓库提供**同数据、同通道、同评估**的受控实验脚手架,用于回答:负迁移从何而来,什么样的门控/目标/调度能把它收回来。

## 方法流水线

```
题目 x
  -> 学生策略 πθ 采样 n 条轨迹                        (vLLM, no-think, 2048)
  -> 规则验证器给出结果奖励                          (strict / semantic)
  -> GRPO 组内相对优势 A_i
  -> [ 优势门 ]  仅 A_i < 0(负优势)轨迹进入候选       (AGOPD)
  -> [ 教师胜任门 ] 教师先解题,仅"教师答对"的题可蒸    (AGOPD)
  -> 词元级监督:hard-CE / top-k 正向 KL / 反向 KL     (可切换)
  -> 系数调度:warmup -> 平台 -> anneal                (SAF-OPD 式)
  -> 合并损失 L = L_GRPO + β·L_KL + c(step)·L_distill
```

## 主要结果

### 1.7B：四臂受控对照（固定 1,024 题，strict）

| 方法 | Correct / 1024 | Accuracy | Δ vs GRPO |
|---|---:|---:|---:|
| Base | 239 | 23.34% | −5.08pp |
| GRPO | 291 | 28.42% | — |
| OPD（纯蒸馏） | 263 | 25.68% | −2.73pp |
| **RL + OPD（无条件）** | 247 | **24.12%** | **−4.30pp** |
| AGOPD（优势门 + 教师胜任门） | 277 | 27.05% | −1.37pp |

结论:无条件 RL+OPD 显著拖累;门控收回大部分损失,但未超越纯 GRPO。

## 何时该用 OPD（工程结论）

| 门槛 | 判据 |
|---|---|
| 教师**逐状态**更强 | 同状态下教师把概率推向成功 token;仅"学生答错"的轨迹上与理想梯度对齐 |
| 教师有**新能力** | 同族、同配方放大规模通常无新能力;教师间能力差 16.7 分,学生只差 0.1 分 |
| 题目按**教师对齐**选 | 按教师困惑度/对齐度选题,而非学生舒适区 |
| 熵/多样性有余量 | 硬 argmax 会把熵压没;高熵 token 需要保留 |

**任一条不满足 → 不要无条件开 OPD**;应改为门控(优势门 + 教师胜任门)、软目标(反向 KL / tail-aware top-k)、系数 warmup→anneal,或干脆先不用。

## 训练配置与硬件配置（摘要）

### 硬件与拓扑

| 环境 | 配置 | 约束 |
|---|---|---|
| 云端（主实验） | **4 × A100 80GB**；3+1 混合：学生 FSDP 训练 + 同卡 rollout 占 3 卡，教师独占 1 卡 | 教师用 **TP1 × 数据并行**（长生成吞吐受并发约束 > 单请求延迟） |
| 本地（探索/调试） | 2 × RTX 4090 24GB | 8B 级模型需 `gpu_memory_utilization ≤ 0.85`、`max_num_batched_tokens ≤ 8192`；该卡型多卡 FSDP 存在 NCCL P2P 限制 → 默认单卡 |

### 训练配方

| 项 | 1.7B 全参数 | 4B LoRA |
|---|---|---|
| 优化通道 | full-param | **LoRA `r=16` + `lr=1e-4`** |
| KL 系数 | `0.05` | `0.05` |
| batch / mini-batch | `8 / 4` | `12 / 6` |
| rollout 组大小 | `6` | `4` |
| 序列预算 | prompt 1024 + response 2048 | 同左（`max_model_len 3073`） |
| 采样可比性 | `data.seed=42` | `data.seed=42` |

引擎/拓扑开关:`rollout.nnodes=0`（actor/rollout 同卡混合）、`enable_sleep_mode=True`、`free_cache_engine=True`、`max_num_batched_tokens=16384`、`rollout.agent.num_workers=16`、`enforce_eager=False`。

OPD 相关开关:`hard_ce` / `reverse_kl` / `temperature`（目标形式）、`coef_schedule=warmup_anneal`（系数调度）、`teacher_after_advantage` + `advantage_gate` + `teacher_gate`（作用域门控）。

完整版本钉扎、数据划分、评估协议与运行纪律见 **[`docs/training-configuration.md`](docs/training-configuration.md)**。

## 仓库结构

```
agopd-rl/
├── src/agopd/            # 核心包:gating(优势门/教师胜任门)、reward、data
├── scripts/              # 训练 wrapper、普查、评估、分析与探针脚本
│   ├── run_4b8b_verify.sh          # 参数化多卡 RL / OPD 启动器
│   ├── run_grpo_vanilla_opd.sh     # verl OPD 训练入口(参数化)
│   ├── evaluate_dapo_vllm.py       # 固定 1,024 评估(direct-LoRA)
│   ├── census_scale.py             # 跨规模能力普查(n=8)
│   └── analyze_4b8b_verify.py      # 两臂逐题配对 Δ / 95% CI / McNemar
├── configs/              # 环境与运行配置
├── patches/              # 针对 verl 的补丁(standalone rollout、padded logprob 等)
├── docs/                 # 方法与设计说明(实验运行日志不在开源范围)
├── reports/              # 论文 HTML、图表、评估汇总 JSON
├── pyproject.toml
└── LICENSE
```

## 快速开始

```bash
# 1) 环境(示例:conda + 官方 vLLM/verl 轮子)
pip install -e ".[dev]"
pip install -e ".[train]"        # ray / tensordict / peft / tensorboard 等

# 2) 准备 verl 并应用补丁(训练必需)
git clone https://github.com/volcengine/verl && git checkout v0.8.0
for p in patches/verl-v0.8.0-*.patch; do git -C verl apply "../$p"; done

# 3) 跨规模能力普查(每模型一张卡)
python scripts/census_scale.py --model ${AGOPD_ROOT}/models/Qwen3-4B \
    --train-file data/dapo-verl-v1/train.parquet --n 8 --max-prompts 16164 \
    --shard-idx 0 --n-shards 4 --out outputs/census_4b_full

# 4) 训练一臂(纯 GRPO 示例;门控/软目标/系数调度由 ARM 与超参控制)
ARM=A TRAIN_STEPS=160 bash scripts/run_4b8b_verify.sh

# 5) 合并 FSDP 检查点并做固定 1,024 评估(direct-LoRA)
python -m verl.model_merger merge --backend fsdp \
    --local_dir ${AGOPD_ROOT}/checkpoints/agopd-rl/<run>/global_step_160/actor \
    --target_dir ${AGOPD_ROOT}/models/<run>160
python scripts/evaluate_dapo_vllm.py --model ${AGOPD_ROOT}/models/Qwen3-4B \
    --lora-adapter ${AGOPD_ROOT}/models/<run>160/lora_adapter --lora-rank 16 \
    --data data/dapo-verl-v1/val_shards/val_s0.parquet \
    --output reports/eval-dapo-v1/<run>_s0.jsonl --max-new-tokens 2048 --seed 42
python scripts/analyze_4b8b_verify.py --a-pattern <A>_s?.jsonl --b-pattern <B>_s?.jsonl
```

## 复现要点

- **数据可比性**:所有需要比较的实验显式传 `data.seed=42`(verl 默认不播种采样序列)。
- **固定评估**:`data/dapo-verl-v1/val.parquet`(1,024 题),非思考模式、`max_new_tokens=2048`、`temperature 0.6 / top_p 0.95 / top_k 20`、`seed 42`,strict 与 semantic 双口径。
- **训练通道**:LoRA `r16 + lr=1e-4`(过窄的 r8 + lr1e-6 通道不适用于该底座)。
- **拓扑**:3+1 混合(学生 FSDP + 同卡 rollout 复用 3 卡,教师 1 卡),`rollout.nnodes=0`。
- **评估口径**:LoRA 检查点必须经 `--lora-adapter` 直接评估;直接评合并目录会退化为 Base。

## 许可

Apache-2.0,见 [LICENSE](LICENSE)。
