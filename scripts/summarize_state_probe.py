#!/usr/bin/env python3
"""Summarize the Appendix-F same-state pilot without treating tokens as IID."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--student-continuations", type=Path, required=True)
    p.add_argument("--teacher-continuations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def bootstrap_ci(values: list[float], seed: int = 42, draws: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    return [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def main() -> None:
    a = parse_args()
    probes = load(a.probe)
    student = {(r["prompt_id"], r.get("trajectory_kind", "student_wrong"), r["state"]): r for r in load(a.student_continuations)}
    teacher = {(r["prompt_id"], r.get("trajectory_kind", "student_wrong"), r["state"]): r for r in load(a.teacher_continuations)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in probes:
        for state, item in row["probes"].items():
            kind = row.get("trajectory_kind", "student_wrong")
            key = (row["prompt_id"], kind, state)
            if key not in student or key not in teacher:
                continue
            s = item["student"]
            t = item["teacher"]
            sr = student[key]
            tr = teacher[key]
            grouped[state].append(
                {
                    "prompt_id": row["prompt_id"],
                    "trajectory_kind": kind,
                    "state": state,
                    "error_type": row["error_type"],
                    "response_token_offset": item["response_token_offset"],
                    "student_entropy": s["topk_entropy"],
                    "teacher_entropy": t["topk_entropy"],
                    "entropy_delta_teacher_minus_student": t["topk_entropy"] - s["topk_entropy"],
                    "student_top16_mass": s["topk_mass"],
                    "teacher_top16_mass": t["topk_mass"],
                    "student_correct_count": sr["correct_count"],
                    "teacher_correct_count": tr["correct_count"],
                    "n_continuations": len(sr["samples"]),
                }
            )
    summary = {
        "protocol": {
            "prompts": len({r["prompt_id"] for r in probes}),
            "states_per_prompt": sorted(grouped),
            "continuations_per_model_state": 8,
            "probe_topk": 16,
            "cluster_unit": "prompt_id",
        },
        "by_state": {},
        "by_state_kind": {},
        "rows": [item for state in grouped.values() for item in state],
    }
    order = ["root", "pre_error_64", "pre_error_16", "error_onset", "post_error_16", "post_error_64"]
    for state in order:
        rows = grouped.get(state, [])
        if not rows:
            continue
        summary["by_state"][state] = {
            "n_prompts": len(rows),
            "student_entropy_mean": mean([r["student_entropy"] for r in rows]),
            "teacher_entropy_mean": mean([r["teacher_entropy"] for r in rows]),
            "entropy_delta_mean": mean([r["entropy_delta_teacher_minus_student"] for r in rows]),
            "student_top16_mass_mean": mean([r["student_top16_mass"] for r in rows]),
            "teacher_top16_mass_mean": mean([r["teacher_top16_mass"] for r in rows]),
            "student_rescue_rate": mean([r["student_correct_count"] / r["n_continuations"] for r in rows]),
            "teacher_rescue_rate": mean([r["teacher_correct_count"] / r["n_continuations"] for r in rows]),
        }
    for kind in sorted({r["trajectory_kind"] for rows in grouped.values() for r in rows}):
        for state in order:
            rows = [r for r in grouped.get(state, []) if r["trajectory_kind"] == kind]
            if not rows:
                continue
            summary["by_state_kind"][f"{kind}:{state}"] = {
                "trajectory_kind": kind,
                "state": state,
                "n_prompts": len(rows),
                "student_entropy_mean": mean([r["student_entropy"] for r in rows]),
                "teacher_entropy_mean": mean([r["teacher_entropy"] for r in rows]),
                "entropy_delta_mean": mean([r["entropy_delta_teacher_minus_student"] for r in rows]),
                "student_rescue_rate": mean([r["student_correct_count"] / r["n_continuations"] for r in rows]),
                "teacher_rescue_rate": mean([r["teacher_correct_count"] / r["n_continuations"] for r in rows]),
            }
    paired = []
    for state in order:
        wrong = {r["prompt_id"]: r for r in grouped.get(state, []) if r["trajectory_kind"] == "student_wrong"}
        correct = {r["prompt_id"]: r for r in grouped.get(state, []) if r["trajectory_kind"] == "student_correct_matched"}
        common = sorted(set(wrong) & set(correct))
        if not common:
            continue
        teacher_diffs = [
            wrong[key]["teacher_correct_count"] / wrong[key]["n_continuations"]
            - correct[key]["teacher_correct_count"] / correct[key]["n_continuations"]
            for key in common
        ]
        student_diffs = [
            wrong[key]["student_correct_count"] / wrong[key]["n_continuations"]
            - correct[key]["student_correct_count"] / correct[key]["n_continuations"]
            for key in common
        ]
        paired.append({
            "state": state,
            "n_prompts": len(common),
            "teacher_wrong_minus_correct": mean(teacher_diffs),
            "teacher_wrong_minus_correct_bootstrap95": bootstrap_ci(teacher_diffs, seed=42),
            "student_wrong_minus_correct": mean(student_diffs),
            "student_wrong_minus_correct_bootstrap95": bootstrap_ci(student_diffs, seed=43),
            "prompt_differences": [
                {"prompt_id": key, "teacher": teacher_diffs[i], "student": student_diffs[i]}
                for i, key in enumerate(common)
            ],
        })
    summary["paired_state_deltas"] = paired
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_state"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary_saved={a.output}", flush=True)


if __name__ == "__main__":
    main()
