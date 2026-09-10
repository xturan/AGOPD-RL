# 4B 学生 × 8B 教师 OPD 验证实验方案(8 小时,2026-09-10 晚)

## 1. 背景与问题

1.7B 主线结论:无条件 RL+OPD(1.7B student, 4B teacher, comfort 1472 全桶, full-param)
在固定 1,024 题上相对纯 GRPO **−4.30 pp**([−6.93, −1.66], 表 4)——状态错位型负迁移。
附录 H 的机制假说:更强的底座更容易访问正确探索状态,教师的同状态监督更有价值,
因此**更强者应在同预算 OPD 下学出正的或至少不损伤的增益**。

本实验在 4B 底座 + 8B 教师上真实检验该假说(替代此前"matched 2,000 抽样"的间接证据)。

## 2. 目标与假设

- **零假设 H0**:4B 学生同样存在无条件 OPD 负迁移(Δ = RL+OPD − GRPO ≤ 0)。
- **备择 H1**:4B 学生能从 8B 教师学到正/不损的 OPD(Δ > 0,或 CI 下界 ≥ 0)。
- 判定以**固定 1,024 题 strict 正确率的配对差 Δ 及 95% CI** 为准;对照 1.7B 的
  −4.30 pp 仅作外部参照,不作同口径显著性。

## 3. 实验臂(今晚全跑,均为全新运行;通道 = 4B 已验证 LoRA 配方)

| 臂 | 学生 | 教师 | 训练 | 说明 |
|---|---|---|---|---|
| **A** | Qwen3-4B(Base,从零) | — | 纯 GRPO | 内部受控基线 |
| **B** | Qwen3-4B(Base,从零) | Qwen3-8B(固定) | GRPO+OPD(hard-CE 无条件) | 待检验臂 |

同子集、同预算、同通道、同配置,B 与 A 仅差"是否叠加 8B 教师蒸馏"。
外部锚:Run C(同 LoRA r16+lr1e-4,4B Pure GRPO 60 步)fixed-1024 strict **42.48%**——
若今晚 A 在其附近则说明配方/子集口径一致,可作跨批次的第二个数据点。

## 4. 共享协议(4B 已验证通道,不套 1.7B full-param)

数据: `data/dapo-verl-v1/train_comfort.parquet`(1472 题); `data.seed=42`
Eval: `data/dapo-verl-v1/val.parquet`(固定 1024), 非思考 @2048, seed 42
优化通道(**Run C / R88 已验证**): **LoRA `rank=16` + `lr=1e-4`**,
  direct-LoRA 评估(`--lora-rank 16`), `batch=12 / mini=6 / n=4`,
  `kl_loss_coef=0.05`(RL 内 KL), full-param 与 r8/lr1e-6 通道一律不用
蒸馏(B): `hard_ce=True`(与 Run C/comfort 系列同), `use_task_rewards=True`(RL+OPD), 无 gate
rollout: T=0.6 / top_p 0.95 / top_k 20, max_prompt 1024, max_response 2048(max_model_len 3073)
拓扑: 3+1 hybrid(`rollout.nnodes=0`, 学生 3 卡 + 同卡 rollout, 教师 1 卡 TP1)
引擎: `enforce_eager=False`, `max_num_batched_tokens=16384`, agent workers 16,
      sleep_mode + free_cache, dynamic_bsz, `save_freq=10`

> **教师先决(必须先测)**:Qwen3-8B Base 在 strict 上强于 4B Base(36.43 vs 33.01),
> 但 semantic 弱(38.48 vs 41.02)。OPD 用 hard 目标/结果奖励,教师"可教性"以
> **规则验证器 strict 同题能力**判定。预检阶段对 comfort 抽样测 8B 教师 vs 4B 学生的
> 题目级优势;若 8B 在抽样上不显著强于学生,则在汇报中注明"教师强度受限"再解读 Δ。
> 教师为 8B:单卡 TP1,`TEACHER_GPU_MEMORY_UTILIZATION≈0.85`,
> `TEACHER_MAX_NUM_BATCHED_TOKENS=16384`;engine init OOM 则清显存重试或降到 0.80。

## 5. 时间与步数(先 smoke 后定 N,总 ≤ 7h 训练+评估)

1. preflight(~30 min):
   - 清残留 `VLLM::EngineCore`、`nvidia-smi` 确认 4 卡 <500MiB;
   - 校验 `models/Qwen3-8B`、`train_comfort.parquet` 在位;
   - **(P2,推荐)** 8B 教师 vs 4B 学生同题能力快测:comfort 抽样 ~200 题 × n=8,
     strict 对比(4 卡分片,约 25 min)→ 报告"教师题目级优势是否成立"。
2. smoke: A 3 步 + B 3 步,实测每步 `A_s`、`B_s`(LoRA 通道预计明显快于 full-param)。
3. 取对称步数 N:
   `N = ⌊ (7×3600 − T_eval − 1800 buffer) / (A_s+B_s) ⌋`
   - 目标 N ∈ [60, 200];Run C 60 步已足够让 Pure GRPO 出信号,OPD 臂 ≥60 步起步。
4. 顺序跑 A(N) → eval A → B(N) → eval B(全 4 卡分片,每臂 ≈ 30–45 min)。

**步数决策表(smoke 实测后用;LoRA 通道参考行)**

| A_s (s) | B_s (s) | 建议 N | 预计 A+B 训练时长 |
|---|---|---|---|
| 40 | 80 | ~200 | ~6.7 h |
| 60 | 120 | ~140 | ~7.0 h |
| 80 | 160 | ~105 | ~7.0 h |
| 120 | 240 | ~75 | ~7.5 h(超预算则优先保 B) |

## 6. 评估与统计

- 每臂最终 checkpoint(`global_step_N`)合并后,统一评估固定 1024:
  **strict 正确率 / 语义正确率 / Δ vs 纯 GRPO / 配对 95% CI / McNemar p / 长度与截断**。
- 配对:两模型在同一 1024 题、同一 prompt、单次采样 seed 42 → 逐题配对,算 McNemar/Wilson CI。
- 判读:
  - Δ>0 且 CI 不含 0 → 强证据(4B 学出更好 OPD,推翻 1.7B 负迁移的外推);
  - Δ≥0 且 CI 跨 0 → 不损伤(相对 1.7B 符号反转,弱支持);
  - Δ<0 且 CI 不含 0 → 假说不成立(更强的学生仍负迁移),论文按此改写。

## 7. 产物与命名

- 启动 wrapper:`scripts/run_4b8b_verify_{a,b}.sh`(由 `run_cloud_opd_series.sh` 拷改)
- 输出:`outputs/scale4b_verify_a_grpo_<N>step` / `outputs/scale4b_verify_b_8bopd_<N>step`
- checkpoint:`checkpoints/agopd-rl/…`; 日志:`/tmp/scale4b_verify_*.log`
- 结果:`reports/eval-dapo-v1/scale4b_verify_{a,b}_2048.jsonl` → 汇总 JSON + 对比小图
- 汇报:更新正文/附录 H 一处(4B×8B 同子集验证),或单独一短段。

## 8. 纪律(CLAUDE.md)

启动前杀残留引擎、4 卡确认 <500MiB;`cd ${AGOPD_ROOT};` 起头;
`data.seed=42`;每个任务只挂一条完成监控;不在云端主线空闲前占本地跑。

## 9. 风险与回退

- 8B teacher 显存不足 → 降 gmu/关 teacher graph(`enforce_eager=True`)只影响速度不影响结论。
- 4B full-param FSDP3 + hybrid 显存不够 → 开 `param/optimizer offload=True`(变慢,步数按实测重算)。
- 时间不够跑完 B → 优先保 B + 复用既有 4B-GRPO-50 固定评估做对照(降级口径,汇报注明)。
- engine init 偶发失败 → 清理后整批重试 ≤2 次,仍失败再降 2+2。

## 10. 执行前确认

- [ ] 云端 4×A100 空闲(已确认:0% / 14MiB)
- [ ] 是否按上述 A+B、comfort1472、Qwen3-4B Base、Qwen3-8B 启动
- [ ] 期望开始时刻(建议 22:30 前后)与 N 目标
