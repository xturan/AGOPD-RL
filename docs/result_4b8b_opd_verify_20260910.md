# 结果:4B 学生 × 8B 教师 无条件 RL+OPD 验证(2026-09-10 夜)

## 配置

- 通道:4B 已验证 LoRA 配方 **r16 + lr=1e-4**,batch12/mini6/n4,**KL=0.05**,seed42(与 1.7B full-param 那套明确分离)
- 数据:`train_comfort.parquet`(1,472 题);评估:固定 1,024 题,非思考 @2048,seed42,direct-LoRA r16
- 学生:Qwen3-4B Base(从零);教师:Qwen3-8B(固定,hard-CE 无条件蒸馏)
- 拓扑:3+1 hybrid(student FSDP3+同卡 rollout,教师 1 卡);160 步;单步 ≈54–58 s(两臂相当,8B 教师几乎零额外单步成本)

## 主结果(固定 1,024,配对)

| 指标 | A 纯 GRPO | B RL+OPD(8B) | Δ = B − A | 95% CI | McNemar p |
|---|---:|---:|---:|---|---:|
| strict | 42.09% | 34.96% | **−7.13 pp** | [−9.95, −4.30] | ≈0 |
| semantic | 41.70% | 36.13% | **−5.57 pp** | [−8.39, −2.74] | 0.0002 |

逐题不一致:A 错 B 对 75 题;B 错 A 对 148 题(方向压倒性)。
描述量:B 回答更长(1400.7 vs 1239.0),**答案解析率下降**(0.708 vs 0.822)。

外部锚:A(42.09%)与 Run C(42.48%,同为 r16+lr1e-4 的 4B Pure GRPO)一致 → 配方/评估口径可靠。

## 判读

- 预注册判据:若 Δ 显著为负 → **"学生探索/信号密度决定 OPD 健康性"不足以解释负迁移**。
  本结果正是该分支:即便换成 4B 学生,无条件词元级 OPD 仍显著拖累,且**幅度(−7.13pp)大于 1.7B 的 −4.30pp**。
- 因此支持更强的一种解释:**词元级无条件 OPD 目标本身与结果奖励 RL 存在冲突**(而非单纯"学生探索不足")。
  同时 B 的解析率下降、长度上升,与历史上"学到的是格式/风格而非解题能力"一致。
- 重要边界:本轮教师 Qwen3-8B Base 在 **semantic 上并不强于 4B Base**(38.48 vs 41.02),仅 strict 高 3.4pp,
  即"更强的教师"这一前提只弱成立。故结论限于:教师仅在规则验证器上略强时,4B 学生仍被无条件 OPD 拖累。

## 与论文的关系

- 直接补强附录 H 的规模论证:把"更易学习的探索空间是解决方向"修正为**必要但不充分**;
  无条件 OPD 的负迁移与底座规模不是单调缓解关系。
- 与主表 4(1.7B RL+OPD −4.30pp)形成同方向、跨规模的两个受控点(不同训练通道,需注明)。

## 工件

- 训练:`outputs/scale4b_verify_{a_grpo,b_8bopd}`;ckpt `${AGOPD_ROOT}/checkpoints/agopd-rl/scale4b_verify_*`(global_step_160)
- 合并模型:`models/scale4b_verify_{a_grpo160,b_8bopd160}`(含 `lora_adapter`)
- 评估:`reports/eval-dapo-v1/scale4b_verify_{a,b}_*_2048_s{0..3}.jsonl`;汇总 `compare` → `scale4b_verify_ab_summary.json`
- 日志:`/tmp/scale4b_verify_{a_grpo,b_8bopd}.log`, `/tmp/merge_{a,b}.log`, `/tmp/eval_{a,b}_s*.log`

## 建议的下一步(可选)

1. **AGOPD@4B(门控)对照**:同通道再跑一臂 4B+8B+两级门控,检验门控是否同样在 4B 上收回损失(把方法结论外推到 4B)。
2. **真"更强教师"条件**:先训 8B 后训练教师(GRPO 50–120 步)再跑 B,检验教师质量是否是限制项。
3. 论文回填:附录 H 增补本表;正文 §3 规模证据措辞按"必要但不充分"改写。

---

# 追加(2026-09-10 午后):根因实证 + 工程修复

## A. 教师真实轨迹探针(comfort 300 题,同协议真实生成)

| 模型 | strict | semantic | answer_marker | boxed | mean_len |
|---|---:|---:|---:|---:|---:|
| teacher Qwen3-8B(固定) | 53.3% | 59.7% | 69.3% | 55.7% | 1273 |
| student 4B Base(未训练) | 47.7% | 61.3% | 36.0% | 47.0% | 1202 |
| student A(纯 GRPO 160 步) | **60.7%** | 60.7% | 56.0% | 49.0% | 1128 |

配对(同 300 题):A 独对 67 题 vs 教师独对 45 题,**McNemar p=0.0467**。
结论:**在训练所用题目上,固定教师 strict(53.3%)低于训后学生(60.7%)** —— 无条件 hard-CE 蒸馏等于把学生往更弱的目标上拉。这与"运行健康、B 从第 4 步起在线 reward 即低于 A、熵 −32%、变长、解析率降"的曲线证据一致。

## B. 工程修复(已落地)

1. **门控臂(纯配置,ARM=C)**:`teacher_after_advantage=True + advantage_gate + teacher_gate`
   —— 只蒸馏"负优势轨迹 ∩ 教师自己答对的题";启动于 12:45,160 步。
2. **fork 代码补齐(ARM=D)**:
   - `distillation_loss.reverse_kl=True`:top-k 内**反向 KL(mode-seeking)** 分支(替代 hard argmax CE);
   - `distillation_loss.coef_schedule=warmup_anneal`(+warmup/anneal 参数):按步数缩放 `opd_weights`,实现 SAF-OPD 式"先升后降"系数;
   - 三处改动已上传云端 fork 并通过 `py_compile`。
3. **对比脚本泛化**:`scripts/analyze_4b8b_verify.py` 支持任意两臂配对(Δ/CI/McNemar)。

## C. 实验矩阵现状

| 臂 | 配置 | 状态 |
|---|---|---|
| A | 纯 GRPO | ✅ 42.09%(strict,固定1024) |
| B | 无条件 RL+OPD(hard CE,coef 1.0) | ✅ 34.96% |
| C | 门控 AGOPD(优势门+教师胜任门) | 🟡 运行中(12:45 起,160 步) |
| D | 门控 + 反向 KL + 系数 warmup/anneal | ⏳ 代码就绪,待 C 完成后启动 |
