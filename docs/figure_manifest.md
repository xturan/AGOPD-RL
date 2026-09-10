# 图表→脚本 清单(single source of truth)

> 原则:每张图一个 `scripts/fig<N>_<name>.py`;`python3 scripts/figN_xxx.py` 重画后,
> 用装配脚本(见装配段)把 `reports/figures/` 的 PNG 以 base64 嵌回 HTML。
> 样式与调色板一律走 `scripts/figstyle.py`(docs/figure_style.md)。

## 数据图(有脚本,可复现)
| 图 | 脚本 | 数据源 | 嵌入对象 |
|---|---|---|---|
| 图 1 · 能力分布与有效组率 | `scripts/fig1_census_egr.py` | 表 A1 + EGR 公式 | `fig1a_census.png` / `fig1b_egr.png` |
| 图 2 · 教师监督:题目级→状态级 | `scripts/fig2_teacher_state_rescue.py` | `teacher_base_capability_map_corrected.json`、`cptr/teacher_rescue_summary.json`、`cptr/random_control_summary.json` | `fig2a/b/c_*.png` |
| 图 4 · 难度分层雷达(六模型含教师) | `scripts/fig4_difficulty_radar.py` | `five_methods_per_question.csv` + `eval-dapo-v1/grpo-4b-50step-ckpt2.jsonl` | `fig4_difficulty_radar.png` |
| 图 5 · 门控行为与蒸馏作用域 | `scripts/fig5_gate_dynamics.py` | `metrics_agopd.json`、`metrics_rlopd.json` | `fig5_gate_dynamics_180.png` |
| 图 6 · Pass@k | `scripts/fig6_passk.py` | `passk_five_methods.json` | `fig6_passk_redesigned.png` |
| 图 D2 · RFT-SFT 检查点探针 | `scripts/figD2_rft_probe.py` | 探针点估计(脚本内) | `figD2_rft_probe_redesigned.png` |

## 图注/图例等特殊图
| 图 | 类型 | 处理 | 状态 |
|---|---|---|---|
| 图 3 · AGOPD 方法结构 | 手绘流程图(资产) | 原 PNG 资产 | 无数据脚本;如需可改为绘图脚本 |
| 图 D1 · 多阶段谱系流程 | 流程图(资产) | 原 PNG 资产 | 同上 |
| 图 B1 / B2 · 更新通道/学习率诊断 | 历史图像资产 | 语义(哪条 run)不可从现存数据唯一重建 | 保留原图,标记 legacy |
| 图 C1 · 四臂训练动态 | metrics 可重建但未纳入本轮 | 待:从 metrics_{grpo,opd,rlopd,agopd}.json 生成 | 挂起(下一轮) |

## 装配方式(历史命令摘要)
替换某图块图片:`python3 - <<'PY' ... base64 ... PY`(正则定位 `<figure …>` 内含 caption 关键字,按 img 顺序换 src)。任何数据图改完后**必须**重跑脚本并重嵌,保证 HTML 内像素 = `reports/figures/` 现文件。
