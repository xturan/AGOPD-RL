# 训练配置与硬件配置

本文件给出复现所需的**训练配方**与**硬件/环境配置**。所有数值均为配置项(超参与版本钉扎),不含任何实验运行记录或结果。

## 1. 软件栈与版本钉扎

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.10(验证于 3.12) | 仓库 CI 额外在 3.11 上做字节编译检查 |
| PyTorch | `2.8.0`(+cu128) | 见 `configs/local-vllm-constraints.txt` |
| torchvision / torchaudio | `0.23.0` / `2.8.0` | 与 torch 对齐 |
| vLLM | `0.10.2` | rollout 与教师推理引擎 |
| transformers | `4.55.4` | 分词与 chat template |
| tensordict | `0.9.1` | verl 数据传输 |
| numpy / scipy | `1.26.4` / `1.15.2` | 数值与统计 |
| cupy-cuda12x | `13.3.0` | CUDA 12 运行时 |
| verl | `v0.8.0` + [`patches/`](../patches) | standalone rollout、padded logprob、FSDP+sdpa 等补丁 |

安装:

```bash
pip install -e ".[dev]"
pip install -e ".[train]"
pip install -r configs/local-vllm-requirements.txt -c configs/local-vllm-constraints.txt
git clone https://github.com/volcengine/verl && git -C verl checkout v0.8.0
for p in patches/verl-v0.8.0-*.patch; do git -C verl apply "../$p"; done
```

## 2. 硬件配置与拓扑

### 2.1 云端(主实验):4 × A100 80GB

```text
GPU0 ┐
GPU1 ├─ 学生 FSDP 训练 + 同卡混合 rollout(vLLM)      # rollout.nnodes=0
GPU2 ┘
GPU3 ──── 教师推理(vLLM,TP=1;多 replica 用数据并行而非 TP2)
```

- **3+1 混合拓扑**:学生训练与采样复用 3 卡(vLLM hybrid,`enable_sleep_mode=True`、`free_cache_engine=True`),教师独占 1 卡;
- 教师侧选择 **TP1 × 数据并行** 而非 TP2:长文本生成的吞吐受并发请求数约束大于单请求延迟;
- 引擎参数:`max_num_batched_tokens=16384`、`rollout.agent.num_workers=16`、`max_model_len=3073`、`enforce_eager=False`(CUDA Graph 需端到端验证后再启用);
- 显存策略:配额只在出现 KV 不足告警时提高,**不把空余显存当作必须填满的目标**。

### 2.2 本地(探索/调试):2 × RTX 4090 24GB

- 驱动 560 / CUDA 12.6 级别的消费级栈;
- 8B 级模型需 `gpu_memory_utilization ≤ 0.85` 且 `max_num_batched_tokens ≤ 8192`;
- 已知约束:该卡型的多卡 FSDP(尤其是 cu128 栈)存在 NCCL P2P 问题 → **本地默认单卡训练**,多卡训练放到云端;
- 本地配置模板见 [`configs/local-4090.env`](../configs/local-4090.env)。

## 3. 数据与划分

| 项 | 配置 |
|---|---|
| 训练集 | 数学推理题集(带规则可验证答案);入口 `--train-file` |
| 子集策略 | 按学生能力 `p̂ = k/n` 分桶;可选"舒适桶"(中等成功率,组内信号密度高)或全量 |
| 固定验证集 | 1,024 题,`--data data/dapo-verl-v1/val.parquet` |
| 分片评估 | `val_shards/val_s{0..3}.parquet`,每片 256 题,4 卡并行 |
| 采样可比性 | 所有需要互相比较的运行显式传 `data.seed=42`(verl 默认不播种采样序列) |

## 4. 训练配方

### 4.1 两套已验证配方

| 项 | 1.7B 全参数 | 4B LoRA(推荐用于大底座) |
|---|---|---|
| 优化通道 | full-param | **LoRA `r=16` + `lr=1e-4`**(过窄的 r8 + lr1e-6 不适用) |
| 学习率 | `1e-6` | `1e-4` |
| KL 系数 | `0.05` | `0.05` |
| batch / mini-batch | `8 / 4` | `12 / 6` |
| rollout 组大小 `n` | `6` | `4` |
| 上下文 | `max_prompt 1024` + `max_response 2048` | 同左(`max_model_len 3073`) |
| 检查点 | `save_freq 10` | `save_freq 10` |

启动入口:`scripts/run_grpo_vanilla_opd.sh`(参数化),或按臂使用 `scripts/run_4b8b_verify.sh`。

### 4.2 OPD / AGOPD 开关(损失与门控)

| 开关 | 取值 | 作用 |
|---|---|---|
| `distillation.enabled` | `True/False` | 是否叠加词元级蒸馏 |
| `distillation.distillation_loss.hard_ce` | `True/False` | 教师 argmax 的 hard-CE(激进)或 soft 目标 |
| `distillation.distillation_loss.reverse_kl` | `True/False` | top-k 内**反向 KL**(mode-seeking),替代正向 KL |
| `distillation.distillation_loss.temperature` | `1.0` | 软目标温度(仅 soft 生效) |
| `distillation.distillation_loss.coef_schedule` | `constant` / `warmup_anneal` | 系数调度:`coef_warmup_steps` 内线性升起,`coef_anneal_end` 前线性退火到 `coef_floor` |
| `distillation.teacher_after_advantage` | `True/False` | 教师打分放在优势计算之后,才能按结果路由 |
| `distillation.advantage_gate.enabled` | `True/False` | 仅**负优势**轨迹进入蒸馏(阈值 `threshold`,权重上限 `max_weight`) |
| `distillation.teacher_gate.enabled` | `True/False` | 教师先解题并用验证器判定,**只蒸馏教师答对的题** |

### 4.3 结果奖励与验证器

- 规则数学验证器:`src/agopd/reward/math_reward.py`(严格契约 + 语义等价双口径);
- verl 适配层:`src/agopd/reward/verl_adapter.py`(自定义 reward function 入口);
- 非思考协议:prompt 已含答题模板,采样时直接使用,不二次包装。

## 5. 评估协议(固定口径)

```bash
# 1) FSDP 检查点 → HF/LoRA 产物
python -m verl.model_merger merge --backend fsdp \
  --local_dir ${AGOPD_ROOT}/checkpoints/agopd-rl/<run>/global_step_<N>/actor \
  --target_dir ${AGOPD_ROOT}/models/<run><N>

# 2) 固定 1,024 题评估(LoRA 必须走 --lora-adapter)
python scripts/evaluate_dapo_vllm.py \
  --model ${AGOPD_ROOT}/models/Qwen3-4B \
  --lora-adapter ${AGOPD_ROOT}/models/<run><N>/lora_adapter --lora-rank 16 \
  --data data/dapo-verl-v1/val_shards/val_s0.parquet \
  --output reports/eval-dapo-v1/<run>_s0.jsonl \
  --max-new-tokens 2048 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42
```

- 解码:非思考、`2048` 上限、`temperature 0.6 / top_p 0.95 / top_k 20`、`seed 42`;
- 指标:strict(契约)与 semantic(等价)双口径,附截断率、平均长度、解析率;
- 逐题配对:同一验证集上两两比较,输出 Δ、95% CI 与 McNemar 精确检验(`scripts/analyze_4b8b_verify.py`)。

## 6. 运行纪律(可复用)

1. 启动前扫描并清理残留推理引擎,确认目标卡显存占用接近零再启动;
2. 多卡是默认:评估/普查/采样按数据分片并行,不排队串行;
3. 集群任务用 `cd <workdir>` 起头,输出目录先建好,长任务落中间结果;
4. 每个任务只挂一条完成监控(判活用进程存活 + 步数推进,避免字符串误触发);
5. 任何引擎参数改动(批处理上限、CUDA Graph、并发 worker)先做端到端冒烟,再进主实验。
