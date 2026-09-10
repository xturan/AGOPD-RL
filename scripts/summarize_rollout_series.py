from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-step rollout reward variance and effective group rate."
    )
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _group_key(row: dict[str, Any]) -> str:
    for key in ("prompt", "input", "prompt_key"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Rollout row has no prompt grouping key")


def summarize_file(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(float(row.get("score", 0.0)))
    effective = sum(len(set(scores)) > 1 for scores in groups.values())
    all_zero = sum(all(score == 0.0 for score in scores) for scores in groups.values())
    all_one = sum(all(score == 1.0 for score in scores) for scores in groups.values())
    scores = [float(row.get("score", 0.0)) for row in rows]
    return {
        "step": int(path.stem) if path.stem.isdigit() else path.stem,
        "rows": len(rows),
        "groups": len(groups),
        "reward_mean": sum(scores) / len(scores) if scores else 0.0,
        "effective_groups": effective,
        "effective_group_rate": effective / len(groups) if groups else 0.0,
        "all_zero_groups": all_zero,
        "all_one_groups": all_one,
    }


def main() -> None:
    args = _parse_args()
    summaries = [summarize_file(path) for path in args.rollout_dir.glob("*.jsonl")]
    summaries.sort(
        key=lambda item: int(item["step"]) if str(item["step"]).isdigit() else str(item["step"])
    )
    output = "".join(json.dumps(item) + "\n" for item in summaries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
