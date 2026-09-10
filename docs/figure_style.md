# AGOPD 论文视觉定义(锁定)

> 唯一来源:`scripts/figstyle.py`。所有数据图一律 `import figstyle as fs`,禁止再手写一份颜色/字号常量。

## 1. 调色板(不引入 palette 外的色相)
| 角色 | 色值 | 用途 |
|---|---|---|
| 主色 primary | `#155b66` | 主序列、教师/GRPO/我们的方法、正效应 |
| 辅色 light | `#9fb9b7` | 轻量辅助序列(Base)、正区域填充 |
| 中阶 mid | `#5d8f8b` | 同色相 ramp 中间步(单量程由浅到深) |
| 负色 red | `#9c4b50` | **只用于负面语义**:错误状态、负效应、受损区、有害标注 |
| 中性灰 | `#6f7779` | 对照/控制组、参照线、纯长度对照 |

规则:
- 红色=bad,禁止把红当普通第 3/5 个序列色。
- 表达"程度/强弱"时用同一色相 light→dark,不跨色相。
- 禁止金色/蓝色等外来色相(旧 `#b39a55`/`#3b6d8a` 已废弃)。

## 2. 方法 5 臂身份映射(identity ≠ color alone)
| 方法 | 颜色 | 线型 | 粗细 |
|---|---|---|---|
| Base | `#9fb9b7` | solid | 1.3 |
| GRPO(RL 参照) | `#155b66` | dashed | 1.4 |
| OPD(无 RL 蒸馏) | `#6f7779` | solid | 1.3 |
| RL+OPD(负迁移臂) | `#9c4b50` | solid | 1.5 |
| AGOPD(本文方法) | `#155b66` | solid **加粗** | 2.2 |

GRPO 与 AGOPD 同色相,靠"虚线 vs 实线加粗"区分;每图必须带图例。

## 3. 画布与字距
- 无网格;仅保留左/下轴框线;轴与刻度文字显式。
- 面板机制(**以图 1 为准**):图内**不放大标题/字母/注记**,图只放数据本身(轴、标签、图例、必要的数值标注)。
  面板**序号+短图题放到每张图下方**(HTML `.fig-panel-label`,如 "(a) Capability census of the base model (n = 8)"),字号颜色统一;整图长描述放 `<figcaption>`。
- 间距:横排子图 `wspace≥0.34`;图内数值标注偏移≥4px;图例放画布外或空隙处,**不得与曲线/柱子相交**;不做任何干扰数据的附加标记(如竖直参照线须经确认);统一 `figstyle.save()`(bbox tight)。
- 图内不出现底部注记串/说明行;数据口径等说明放 `.fig-panel-label` 或 `<figcaption>`。

## 4. 一图一脚本
每个图一个 `scripts/fig<N>_<name>.py`(见表 single-source-of-truth 清单 `docs/figure_manifest.md`),禁止多个图共用一个杂烩脚本;`save()` 输出到 `reports/figures/`,随后由装配脚本以 base64 嵌入 HTML。
