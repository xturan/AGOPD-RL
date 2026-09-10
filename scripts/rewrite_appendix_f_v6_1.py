#!/usr/bin/env python3
"""Rewrite Appendix F with the reviewed real-trajectory state-probe evidence.

The v6.0 report is treated as immutable input.  The output is a self-contained
HTML copy with embedded PNG panels and actual trajectory excerpts generated
from the reviewed pilot artifacts.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/AGOPD_paper_v6_0_restructured_20260908.html"
OUTPUT = ROOT / "reports/AGOPD_paper_v6_2_state_probe_20260909.html"
REVIEWED = ROOT / "reports/appendix_f_state_probe/error_audit_reviewed8.jsonl"
SUMMARY = ROOT / "reports/appendix_f_state_probe/matched16_summary.json"
TEACHER_CONT = ROOT / "reports/appendix_f_state_probe/teacher_same_state_matched16.jsonl"
FIG_PREFIX = ROOT / "reports/figures/figF_state"
FIG_COMPOSITE = ROOT / "reports/figures/figF_state_composite.png"
STATE_LABELS = {
    "root": "根状态",
    "error_onset": "首次错误锚点",
    "post_error_64": "错误后 64 token",
}
ERROR_TYPE_LABELS = {
    "interpretation": "题意/结构解释",
    "arithmetic": "算术/数值计算",
    "unsupported_inference": "无依据推断",
    "algebra": "代数变形",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def count_rate(value: float, n_prompts: int, n: int = 8) -> str:
    count = round(value * n_prompts * n)
    total = n_prompts * n
    return f"{count} / {total} ({pct(value)})"


def state_label(state: str) -> str:
    return STATE_LABELS.get(state, state)


def error_type_label(error_type: str) -> str:
    return ERROR_TYPE_LABELS.get(error_type, error_type)


def excerpt(text: str, start: int, before: int = 250, after: int = 560) -> str:
    left = max(0, start - before)
    right = min(len(text), start + after)
    return text[left:right]


def correct_excerpt(text: str, tokenizer: object, offset: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    left = max(0, offset - 90)
    right = min(len(ids), offset + 110)
    return tokenizer.decode(ids[left:right], skip_special_tokens=False)


def compact_text(text: str, limit: int = 760) -> str:
    """Keep real text while collapsing pathological repeated answer lines."""
    kept = []
    previous = None
    repeated = 0
    for line in text.replace("\r", "").splitlines():
        normalized = line.strip()
        if normalized and normalized == previous:
            repeated += 1
            if repeated == 2:
                kept.append("[repeated line omitted]")
            continue
        kept.append(line)
        previous = normalized
        repeated = 0
    result = "\n".join(kept).strip()
    if len(result) > limit:
        result = result[:limit].rstrip() + "\n[… excerpt truncated; full trajectory is retained in the JSONL artifact]"
    return result


def trajectory_html(text: str, limit: int = 760) -> str:
    """Render a readable excerpt while retaining the model's mathematical text."""
    text = compact_text(text.replace("<think>", "").replace("</think>", ""), limit)
    text = plain_math_text(text)
    lines = []
    for raw_line in html.escape(text, quote=False).splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
        elif line.startswith("### ") or line.startswith("## "):
            lines.append(f'<span class="trajectory-heading">{line.lstrip("# ")}</span>')
        elif line in {"---", "***"}:
            lines.append('<span class="trajectory-rule"></span>')
        elif line.startswith("- ") or line.startswith("* "):
            lines.append(f'<span class="trajectory-bullet">· {line[2:]}</span>')
        elif line.startswith("Answer:"):
            lines.append(f'<span class="trajectory-answer">{line}</span>')
        else:
            lines.append(raw_line)
    return "<br>".join(lines)


def plain_math_text(text: str) -> str:
    """Remove raw TeX delimiters from model excerpts without changing the source."""
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: " " + " ".join(m.group(1).split()) + " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    replacements = {
        r"\\cdot": "·", r"\\times": "×", r"\\leq": "≤", r"\\geq": "≥",
        r"\\neq": "≠", r"\\in": "∈", r"\\notin": "∉", r"\\pm": "±",
        r"\\approx": "≈", r"\\quad": " ", r"\\qquad": " ", r"\\,": " ",
        r"\\;": " ", r"\\!": "", r"\\boxed": "", r"\\left": "", r"\\right": "",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = text.replace("$$", "").replace("$", "")
    text = text.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return text


def problem_only(prompt: str) -> str:
    parts = prompt.replace("\r", "").split("\n\n")
    if parts and parts[0].startswith("Solve the following"):
        parts = parts[1:]
    parts = [part for part in parts if not part.startswith("Remember to put") and part.strip() != "/no_think"]
    return "\n\n".join(parts).strip()


def teacher_sample_map() -> dict[tuple[str, str, str], dict]:
    result = {}
    for row in load_jsonl(TEACHER_CONT):
        result[(row["prompt_id"], row["trajectory_kind"], row["state"])] = row
    return result


def make_exhibits(rows: list[dict], tokenizer: object) -> str:
    teacher = teacher_sample_map()
    cards = []
    for case_no, row in enumerate(rows, 1):
        pid = row["prompt_id"]
        error_start = row["error_onset_char"]
        quote = row["error_quote"]
        wrong_before = row["student_response"][max(0, error_start - 420):error_start].strip()
        wrong_after_start = error_start + len(quote)
        wrong_after = row["student_response"][wrong_after_start:wrong_after_start + 620].strip()
        correct_context = correct_excerpt(row["matched_correct_response"], tokenizer, row["error_onset_token"]) if row.get("matched_correct_response") else "(matched correct trajectory unavailable)"
        state = teacher.get((pid, "student_wrong", "error_onset"), {})
        samples = state.get("samples", [])
        rescue = next((sample for sample in samples if sample.get("correct")), None)
        failure = next((sample for sample in samples if not sample.get("correct")), None)
        if rescue is None and samples:
            rescue = samples[0]
        if failure is None and samples:
            failure = samples[0]
        rescue_text = compact_text(rescue["text"]) if rescue else "(no continuation recorded)"
        failure_text = compact_text(failure["text"]) if failure else "(no continuation recorded)"
        rescue_label = "verified correct" if rescue and rescue.get("correct") else "no verified rescue in selected sample"
        failure_label = "verified wrong" if failure and not failure.get("correct") else "no wrong sample in selected batch"
        if rescue and rescue.get("correct") and failure and not failure.get("correct"):
            case_reading = "同一错误状态下，Teacher 的续写同时出现通过与未通过样本；该 Case 展示的是状态条件下的分支结果，不把单次续写当作模型总体准确率。"
        elif rescue and rescue.get("correct"):
            case_reading = "该错误状态下观察到至少一个通过验证的 Teacher 续写；是否具有稳定救援能力仍以 F2 的 prompt-level 汇总为准。"
        else:
            case_reading = "该错误状态下的选定 Teacher 续写未通过验证；该单条案例不能推出 Teacher 在所有错误状态上均无法恢复。"
        cards.append(
            f'''<article class="state-probe-card">
<div class="state-probe-card-head"><strong>Case {case_no}</strong><span>参考答案 = {esc(plain_math_text(row["gt"]))} · 错误类型 = {esc(error_type_label(row["error_type"]))} · 锚点 = {row["error_onset_token"]} tokens</span></div>
<div class="state-probe-meta"><div><b>题目</b>{esc(plain_math_text(problem_only(row["prompt"])))}</div><div><b>人工审计结论</b>{esc(plain_math_text(row["error_unit"]))}<br><span class="state-probe-quote">“{esc(plain_math_text(row["error_quote"]))}”</span></div></div>
<div class="state-probe-exhibit-grid">
<div class="state-probe-sample"><div class="state-probe-label wrong-label">错误轨迹 · 首次错误之前</div><div class="trajectory-text">{trajectory_html(wrong_before, 560)}</div></div>
<div class="state-probe-sample"><div class="state-probe-label wrong-label">错误轨迹 · 首个无效推理单元</div><div class="trajectory-text">{trajectory_html(quote + "\n\n" + wrong_after, 520)}</div></div>
<div class="state-probe-sample"><div class="state-probe-label correct-label">同题正确轨迹 · 匹配状态窗口</div><div class="trajectory-text">{trajectory_html(correct_context, 720)}</div></div>
<div class="state-probe-sample"><div class="state-probe-label correct-label">教师同状态续写 · {"验证正确" if rescue and rescue.get("correct") else "未观察到验证正确样本"}</div><div class="trajectory-text">{trajectory_html(rescue_text)}</div></div>
<div class="state-probe-sample"><div class="state-probe-label wrong-label">教师同状态续写 · {"验证错误" if failure and not failure.get("correct") else "未观察到验证错误样本"}</div><div class="trajectory-text">{trajectory_html(failure_text)}</div></div>
<div class="state-probe-sample case-reading"><div class="state-probe-label">本 Case 的证据判读</div><div class="trajectory-text reading-text">{esc(case_reading)}</div></div>
</div>
<p class="state-probe-card-note">阅读顺序为“错误前缀 → 首个无效推理单元 → 同题正确状态 → Teacher 同状态续写”。数学表达式由 MathJax 渲染；案例只用于展示真实状态与真实输出，总体估计来自全部 prompt-level continuation，而不是单条案例。</p>
</article>'''
        )
    return "\n".join(cards)


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    reviewed = load_jsonl(REVIEWED)
    panels = {"composite": b64(FIG_COMPOSITE)}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-1.7B", trust_remote_code=True)
    except Exception:
        tokenizer = None

    style = r'''
.state-probe-figure{margin:30px auto 38px;max-width:860px;width:100%}
.state-probe-composite img{width:100%;height:auto;border:0;background:#fff;display:block}
.state-probe-composite figcaption{text-align:center;max-width:820px;margin:9px auto 0;font-size:11.5px;line-height:1.55}
.state-probe-composite .figure-title{font-weight:400;color:var(--ink2)}
.state-probe-panel-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px 20px;align-items:start}
.state-probe-panel-grid img{width:100%;height:auto;border:0;background:#fff;display:block}
.state-probe-panel-label{margin:7px 0 0;text-align:center;color:var(--muted);font:400 10.5px/1.45 var(--sans)}
.state-probe-card{margin:24px 0 30px;padding:18px 0 22px;border-top:1.4px solid var(--ink);border-bottom:1px solid var(--rule)}
.state-probe-card-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.state-probe-card-head strong{font:560 17px var(--serif);color:var(--ink)}
.state-probe-card-head span{font:10.5px var(--mono);color:var(--muted)}
.state-probe-meta{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;margin-bottom:14px;padding:11px 12px;background:#f6f8f6;border-left:2px solid var(--accent);font-size:11.7px;line-height:1.6;color:var(--ink2)}
.state-probe-meta b{display:block;margin-bottom:3px;color:var(--muted);font:650 9.5px var(--mono);text-transform:uppercase}
.state-probe-quote{color:var(--bad)}
.state-probe-exhibit-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px 18px;align-items:stretch}
.state-probe-sample{min-width:0}
.state-probe-label{margin:0 0 5px;font:650 10px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.01em}
.wrong-label{color:var(--bad)}.correct-label{color:var(--accent)}
.trajectory-text{margin:0;height:250px;box-sizing:border-box;overflow:auto;padding:11px 12px;border:1px solid var(--rule);background:#fbfbfa;color:var(--ink2);font:10.7px/1.6 var(--mono);white-space:normal}
.reading-text{font-family:var(--sans);font-size:11.3px;line-height:1.7;color:var(--ink2);display:flex;align-items:flex-start}
.trajectory-heading{display:inline-block;margin:4px 0 1px;color:var(--ink);font:650 10.7px/1.5 var(--sans)}
.trajectory-rule{display:block;border-top:1px solid var(--rule);margin:7px 0}
.trajectory-bullet{display:inline-block;padding-left:8px}
.trajectory-answer{color:var(--accent);font-weight:650}
.state-probe-card-note{margin:9px 0 0;color:var(--muted);font-size:10.8px;line-height:1.55}
.equation-block{margin:16px auto 17px;text-align:center;overflow-x:auto}
.equation-block .equation-note{margin-top:3px;text-align:center;color:var(--muted);font-size:10.5px;line-height:1.45}
.display-formula{display:flex;align-items:center;justify-content:center;gap:22px;min-width:max-content;padding:4px 10px;font:400 16px/1.7 var(--serif);color:var(--ink)}
.display-formula sub,.display-formula sup{font-size:.68em;line-height:0}
.display-formula .formula-tag{font:400 10.5px var(--mono);color:var(--muted)}
.state-probe-chain{margin:22px 0 28px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.state-probe-chain li{padding:11px 0 11px 6px;border-bottom:1px solid var(--rule);margin:0;color:var(--ink2)}
.state-probe-chain li:last-child{border-bottom:0}
.state-probe-chain strong{color:var(--accent)}
.state-probe-small{font-size:12px;color:var(--muted)}
@media(max-width:760px){.state-probe-panel-grid,.state-probe-exhibit-grid,.state-probe-meta{grid-template-columns:1fr}.state-probe-sample pre{max-height:300px}}
'''
    source = source.replace("</style>", style + "</style>", 1)
    source = source.replace("<title>AGOPD — Academic Revision v5.8</title>", "<title>AGOPD — Academic Revision v6.2</title>")
    source = source.replace("08 Sep 2026 · v5.8 diagnosis prose/figure revision", "09 Sep 2026 · v6.2 state-probe revision")

    def metric(kind: str, state: str, key: str) -> float:
        return float(summary["by_state_kind"][f"{kind}:{state}"][key])

    n = 8
    table_rows = []
    for state, label in (("root", "根状态"), ("error_onset", "首次错误锚点"), ("post_error_64", "错误后 64 token")):
        tw = metric("student_wrong", state, "teacher_rescue_rate")
        tc = metric("student_correct_matched", state, "teacher_rescue_rate")
        sw = metric("student_wrong", state, "student_rescue_rate")
        sc = metric("student_correct_matched", state, "student_rescue_rate")
        table_rows.append(
            f"<tr><td>{label}</td><td class=\"num\">{count_rate(tw,n)}</td><td class=\"num\">{count_rate(tc,n)}</td><td class=\"num\">{pp(tw-tc)}</td><td class=\"num\">{count_rate(sw,n)}</td><td class=\"num\">{count_rate(sc,n)}</td></tr>"
        )
    entropy_rows = []
    for state, label in (("root", "根状态"), ("error_onset", "首次错误锚点"), ("post_error_64", "错误后 64 token")):
        tw = metric("student_wrong", state, "teacher_entropy_mean")
        tc = metric("student_correct_matched", state, "teacher_entropy_mean")
        sw = metric("student_wrong", state, "student_entropy_mean")
        sc = metric("student_correct_matched", state, "student_entropy_mean")
        entropy_rows.append(f"<tr><td>{label}</td><td class=\"num\">{tw:.3f}</td><td class=\"num\">{tc:.3f}</td><td class=\"num\">{sw:.3f}</td><td class=\"num\">{sc:.3f}</td></tr>")
    paired_rows = []
    for row in summary["paired_state_deltas"]:
        ci = row["teacher_wrong_minus_correct_bootstrap95"]
        paired_rows.append(f"<tr><td>{state_label(row['state'])}</td><td class=\"num\">{row['teacher_wrong_minus_correct']*100:+.2f} pp</td><td class=\"num\">[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]</td></tr>")

    if tokenizer is not None:
        exhibits = make_exhibits(reviewed, tokenizer)
    else:
        exhibits = "<p>Trajectory exhibits unavailable because the local tokenizer could not be loaded.</p>"
    f = rf'''<h2 id="appendix-f-state-probe"><span class="secno">APPENDIX F</span>真实错误轨迹上的状态级可恢复性</h2>
<p>本附录回答一个比“错误前缀越长是否越难”更严格的问题：学生模型在真实生成中何时首次离开可验证的推理路径，离开之后错误状态是否持续，以及教师在同一状态上是否仍然能够恢复。与附录旧版的固定比例前缀不同，本轮先对真实 Student trajectory 进行语义审计，再以首次可验证错误单元的起始 token 作为状态锚点。该设计把“错误进入”“错误持续”和“教师救援”分成三个可观测事件。</p>
<div class="claim"><strong>证据合同</strong>正式采样使用 Qwen3-1.7B Base 在固定验证题池上的 200 道题、每题 8 条真实轨迹；其中 86 道题同时出现正确与错误轨迹。附录 F 的状态级 pilot 只使用其中 8 道题的高置信人工复核配对，因而所有聚合结果均标为方向性证据，不代表 200 题总体估计。</div>
<h3>F.1 真实轨迹与首次错误标注</h3>
<p>对每道题，Student 在 temperature=0.6、top-p=0.95、top-k=20、seed=42、最大输出 2,048 token 的协议下采样 8 条回答。结果验证器只负责判断最终答案；对进入状态级分析的错误轨迹，另按 reasoning unit 逐步审阅，并定义 <code>t*</code> 为第一处违反题意、数学规则或前序结论的可验证错误单元起点。格式重复、答案抽取失败、输出截断但没有可定位语义错误的样本不被当作“错误路径”。</p>
<p>本轮 8 条高置信轨迹的错误起点具有不同类型：互素条件误读、平方根算术错误、中心对称常数错误、比例值误读、有序因子对重复计数、约数位置误读、几何对象误读和代数展开错误。每条错误轨迹保留同题一条正确轨迹作为 matched control，控制轨迹在错误轨迹的 token offset 处进行状态对照。</p>
<div class="table-wrap"><table><thead><tr><th>Case</th><th>错误类别</th><th>首次错误锚点</th><th>人工审计的失效机制</th></tr></thead><tbody>{''.join(f'<tr><td>Case {i}</td><td>{esc(error_type_label(row["error_type"]))}</td><td class="num">{row["error_onset_token"]} tokens</td><td>{esc(plain_math_text(row["error_unit"]))}</td></tr>' for i, row in enumerate(reviewed, 1))}</tbody><caption>表 F1 · 真实 Student 错误轨迹的首次错误标注。Case 编号仅用于附录内部索引；token offset 相对于 Student response，不包含用户 prompt。原始错误 quote 在 F3 的逐案例轨迹中展示。</caption></table></div>
<h3>F.2 同状态探针与续写协议</h3>
<p>给定题目 <i>x</i> 与 Student 前缀 <i>y</i><sub>&lt;t</sub>，两模型均接收同一段 <i>s</i><sub>t</sub> = (<i>x</i>, <i>y</i><sub>&lt;t</sub>)，其实现为同一 chat template 渲染出的 user message 加同一 assistant prefix；不额外添加“请纠错”“忽略草稿”或 <code>&lt;student_draft&gt;</code> 指令。对 Student 与固定 Qwen3-4B-GRPO Teacher，分别记录状态 <i>s</i><sub>t</sub> 后一个 token 的 top-16 条件分布。为避免将分布形状、模型正确性和训练效果混为一谈，本附录使用以下两个定义：</p>
<div class="equation-block"><div class="display-formula"><span><i>H</i><sub>K</sub><sup>(16)</sup>(<i>s</i><sub>t</sub>) = −∑<sub><i>v</i> ∈ Top16(<i>K</i>)</sub> p̃<sub>K</sub>(<i>v</i> | <i>s</i><sub>t</sub>) log p̃<sub>K</sub>(<i>v</i> | <i>s</i><sub>t</sub>)</span><span class="formula-tag">(F.1)</span></div><div class="equation-note">式 (F.1) · 模型 K 在状态 s<sub>t</sub> 上 top-16 条件分布的熵，单位为 nat。</div></div>
<div class="equation-block"><div class="display-formula"><span>p̃<sub>K</sub>(<i>v</i> | <i>s</i><sub>t</sub>) = p<sub>K</sub>(<i>v</i> | <i>s</i><sub>t</sub>) / ∑<sub><i>u</i> ∈ Top16(<i>K</i>)</sub> p<sub>K</sub>(<i>u</i> | <i>s</i><sub>t</sub>)</span><span class="formula-tag">(F.2)</span></div><div class="equation-note">式 (F.2) · 在 top-16 集合内部重新归一化；因此 H<sub>K</sub><sup>(16)</sup> 不是全词表熵，也不直接表示答案正确率。</div></div>
<p>在 root、首次错误锚点和锚点后 64 token 三个位置，Student 与 Teacher 各采样 8 条 continuation，并由相同数学验证器判断最终答案。统计单位是 prompt，而不是 token；图中的浅色点保留每个 prompt 的原始值，实线/虚线表示 8 个 prompt 的均值。root 状态在 wrong/correct alias 间完全复用同一批 continuation，因此其配对差严格为零。</p>
<div class="table-wrap"><table class="main-results-table"><thead><tr><th>状态</th><th>Teacher · 错误轨迹</th><th>Teacher · 同题正确控制</th><th>Teacher Δ</th><th>Student · 错误轨迹</th><th>Student · 同题正确控制</th></tr></thead><tbody>{''.join(table_rows)}</tbody><caption>表 F2 · 同状态 continuation 的正确率。每个单元为 8 个 prompt × 8 次 continuation 的合计；Teacher Δ 为错误轨迹减去同题正确控制。配对 bootstrap 区间另列于表 F4。</caption></table></div>
<div class="table-wrap"><table><thead><tr><th>状态</th><th class="num">Teacher H · 错误轨迹</th><th class="num">Teacher H · 同题正确控制</th><th class="num">Student H · 错误轨迹</th><th class="num">Student H · 同题正确控制</th></tr></thead><tbody>{''.join(entropy_rows)}</tbody><caption>表 F3 · 同状态 top-16 条件化熵，单位 nat。熵是局部分布形状指标，不等价于答案正确率；低熵错误状态表示可能存在自信地延续错误的情况。</caption></table></div>
<h4>F.2.1 指标定义与解释边界</h4>
<div class="table-wrap"><table><thead><tr><th>指标</th><th>操作性定义</th><th>可支持的解释</th></tr></thead><tbody>
<tr><td><code>t*</code> · 首次错误锚点</td><td>人工审计发现的第一处可验证无效 reasoning unit 的起始 token</td><td>定位状态转折；不等于模型在单个 token 上突然“知道自己错了”</td></tr>
<tr><td><code>R_teacher(s)</code> · Teacher rescue</td><td>从状态 <code>s</code> 独立采样 8 条 continuation，其中最终答案通过验证器的比例</td><td>衡量 Teacher 在该状态上的可恢复性</td></tr>
<tr><td><code>R_student(s)</code> · Student continuation correctness</td><td>从同一状态由 Student 继续采样，最终答案通过验证器的比例</td><td>衡量错误状态对 Student 自身后续生成的约束</td></tr>
<tr><td><code>H16(K,s)</code> · top-16 entropy</td><td>模型 <code>K</code> 下一 token 的 top-16 概率重新归一化后的熵，单位 nat</td><td>描述局部分布的集中或分散；不能单独判断正确性</td></tr>
<tr><td><code>Δ_teacher</code> · 配对差异</td><td>同一题错误状态的 Teacher rescue 减去 matched correct 状态的 Teacher rescue</td><td>控制题目难度后，描述状态条件带来的方向性差异</td></tr>
<tr><td>统计单位</td><td>以 prompt 为聚类单位进行 bootstrap；token 和 continuation 不视为独立题目</td><td>避免因单条长轨迹含有大量 token 而夸大证据强度</td></tr>
</tbody><caption>表 F5 · 附录 F 指标字典。所有指标均绑定到状态、模型和采样协议；训练过程中的 reward/entropy collapse 不由本表直接定义。</caption></table></div>
<figure class="state-probe-figure state-probe-composite"><img alt="Four aligned panels showing rescue and top-16 entropy for wrong and matched-correct trajectories" src="data:image/png;base64,{panels['composite']}"><figcaption><b>图 F1 · 真实错误轨迹上的状态级 rescue 与 top-16 分布 pilot。</b>(a) Teacher continuation 正确数；(b) Student continuation 正确数；(c) Teacher top-16 条件化熵；(d) Student top-16 条件化熵。深色实线为错误 trajectory，浅色虚线为同题 matched correct control；点为 prompt-level 原始值，线为 8 个 prompt 的均值。图号、图题和图释置于图下，组合图内仅保留小号面板标识与坐标信息；本图为单 seed、小样本 pilot，不承担总体显著性结论。</figcaption></figure>
<h3>F.3 真实 trajectory exhibit</h3>
<p>以下案例展示完整证据单位：同题 Student 错误轨迹、第一处错误上下文、同题正确轨迹，以及 Teacher 从错误状态继续生成的一个成功和一个失败样本。每个 Case 固定使用同一阅读顺序，先给出可验证的错误单元，再给出匹配状态和续写结果；这样读者可以把“错误进入”“状态条件”和“后续是否恢复”逐一对应起来，而不必从一条长文本中猜测证据关系。展示文本是原始模型输出的可读摘录，完整未改写轨迹保存在 JSONL 工件中。</p>
<div class="state-probe-exhibit">{exhibits}</div>
<h3>F.4 证据链与解释边界</h3>
<ol class="state-probe-chain">
<li><strong>错误进入：</strong>200 道题的 1,600 条真实 Student trajectory 中，1,285 条最终未通过结果验证；86 道题同时产生正、负轨迹，说明 Student 的访问状态不是单一正确路径，而是包含可比较的成功和失败分支。</li>
<li><strong>错误定位：</strong>只有被审计为存在第一处可验证语义错误的轨迹才进入状态级分析；格式重复、答案抽取和截断问题被单独排除，避免把输出协议问题写成推理偏移。</li>
<li><strong>状态差异：</strong>在同一题、相同相对位置上，错误轨迹与正确轨迹形成 matched control。若错误后的 Teacher rescue 低于控制，差异才可归因于状态条件，而不是题目难度本身。</li>
<li><strong>错误持续：</strong>错误点后的 +64 token 状态用于检验错误是否继续约束后续生成。pilot 中 Teacher rescue 的错误路径与正确控制差距扩大，支持“错误状态具有累积性”的方向性解释。</li>
<li><strong>对 OPD 的含义：</strong>OPD 在 Student 已访问状态上提供局部 token 监督；如果该状态已偏离可验证路径，Teacher 的局部分布可能不再是可靠的纠错目标。因此，Student 的状态访问质量是 OPD 健康性的关键调节变量，但不是唯一因素。</li>
</ol>
<div class="table-wrap"><table><thead><tr><th>状态</th><th class="num">Teacher 错误轨迹 − 正确控制</th><th class="num">Prompt-bootstrap 95% CI</th></tr></thead><tbody>{''.join(paired_rows)}</tbody><caption>表 F4 · Teacher 在错误 trajectory 与同题正确 control 之间的配对差异。bootstrap 重采样单位为 prompt；由于 pilot 只有 8 个 prompt，区间仅用于表达不确定性，不进行总体结论推断。</caption></table></div>
<p>本附录支持的最强表述是：Student 真实生成中的错误状态与 Teacher 的后续可恢复性相关，而且偏离持续后差异扩大。它不支持“Teacher 在所有错误状态上都无法救回”，也不支持仅凭 entropy 证明模型发生全局坍塌。本文将“trajectory-level irrecoverability”与训练过程中的 entropy collapse、reward collapse 分开使用；后者必须由逐 step 的训练动态另行证明。</p>
<h3>F.5 数据与复现工件</h3>
<p>完整 200 题 Student 轨迹、86 条同题配对审计队列、8 条人工复核记录、same-state top-16 probe、Student/Teacher continuation、汇总 JSON 与图表源脚本均保存在项目的 Appendix-F state-probe 工件目录中。图 F1 的原始数据保留 prompt-level 点，PNG 用于 HTML 内嵌，SVG/PDF 用于后续论文排版。正式扩展仍需将人工复核规模提高到至少 20–30 个 prompt-level pairs，并在此基础上重新估计总体救援差异。</p>'''

    start = source.find('<h2><span class="secno">APPENDIX F</span>')
    if start < 0:
        raise RuntimeError("Appendix F heading not found")
    end = source.find("</section>", start)
    if end < 0:
        raise RuntimeError("Appendix F closing section not found")
    f = f.replace('<figcaption><b>图 F1 ·', '<figcaption><span class="figure-title">图 F1 ·')
    f = f.replace('分布 pilot。</b>(a)', '分布 pilot。</span>(a)')
    source = source[:start] + f + source[end:]
    OUTPUT.write_text(source, encoding="utf-8")
    print(f"written={OUTPUT}")
    print(f"appendix_f_chars={len(f)}")


if __name__ == "__main__":
    main()
