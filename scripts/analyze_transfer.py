from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["prompt_id"]: row for row in rows}


def _correct(row: dict) -> bool:
    return bool(row.get("semantic_correct", row.get("correct", False)))


def _repetition(text: str) -> dict[str, float | int]:
    tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text.lower())
    ngrams = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    counts = Counter(ngrams)
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines() if line.strip()]
    line_counts = Counter(lines)
    return {
        "token_count": len(tokens),
        "rep4_rate": (len(ngrams) - len(counts)) / len(ngrams) if ngrams else 0.0,
        "max_rep4_count": max(counts.values(), default=0),
        "repeated_line_rate": (len(lines) - len(line_counts)) / len(lines) if lines else 0.0,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = (x_var * y_var) ** 0.5
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transfer and degeneration in saved rollouts.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--opd", required=True, type=Path)
    parser.add_argument("--position-kl", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = _load(args.base)
    teacher = _load(args.teacher)
    opd = _load(args.opd)
    ids = sorted(set(base) & set(teacher) & set(opd))

    quadrants = Counter()
    for prompt_id in ids:
        b, t, o = _correct(base[prompt_id]), _correct(teacher[prompt_id]), _correct(opd[prompt_id])
        quadrants["A_base_teacher_correct"] += int(b and t)
        quadrants["B_base_wrong_teacher_correct"] += int(not b and t)
        quadrants["C_base_correct_teacher_wrong"] += int(b and not t)
        quadrants["D_base_teacher_wrong"] += int(not b and not t)

    b_ids = [prompt_id for prompt_id in ids if not _correct(base[prompt_id]) and _correct(teacher[prompt_id])]
    c_ids = [prompt_id for prompt_id in ids if _correct(base[prompt_id]) and not _correct(teacher[prompt_id])]
    ptr = sum(_correct(opd[prompt_id]) for prompt_id in b_ids) / len(b_ids) if b_ids else None
    ntr = sum(not _correct(opd[prompt_id]) for prompt_id in c_ids) / len(c_ids) if c_ids else None

    model_repetition = {}
    for name, rows in (("base", base), ("teacher", teacher), ("opd100", opd)):
        metrics = [_repetition(rows[prompt_id]["output"]) for prompt_id in ids]
        model_repetition[name] = {
            "samples": len(metrics),
            "rep4_rate_mean": sum(item["rep4_rate"] for item in metrics) / len(metrics) if metrics else 0.0,
            "repeated_line_rate_mean": sum(item["repeated_line_rate"] for item in metrics) / len(metrics)
            if metrics
            else 0.0,
            "max_rep4_count_mean": sum(item["max_rep4_count"] for item in metrics) / len(metrics)
            if metrics
            else 0.0,
        }

    result = {
        "samples": len(ids),
        "quadrants": dict(quadrants),
        "positive_transfer_rate": ptr,
        "negative_transfer_rate": ntr,
        "net_transfer": (ptr - ntr) if ptr is not None and ntr is not None else None,
        "repetition": model_repetition,
    }

    if args.position_kl:
        kl_rows = _load(args.position_kl)
        paired = [
            (float(kl_rows[prompt_id]["final_75_100_teacher_student_kl"]), float(_correct(opd[prompt_id])))
            for prompt_id in ids
            if prompt_id in kl_rows and "final_75_100_teacher_student_kl" in kl_rows[prompt_id]
        ]
        result["final_kl_vs_correct_correlation"] = _pearson(
            [item[0] for item in paired], [item[1] for item in paired]
        )
        result["final_kl_vs_correct_samples"] = len(paired)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
