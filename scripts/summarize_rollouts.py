from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agopd.rollout.quality import summarize_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize verl rollout JSONL quality.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", type=int, default=4)
    parser.add_argument("--tail-chars", type=int, default=800)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    summary = summarize_rollouts(rows)
    reasons = Counter(str(row.get("reason", "unknown")) for row in rows)

    print(f"rows={summary.total}")
    print(f"correct={summary.correct}")
    print(f"accuracy={summary.accuracy:.4f}")
    print(f"answer_format_rate={summary.answer_format_rate:.4f}")
    print(f"boxed_format_rate={summary.boxed_format_rate:.4f}")
    print(f"closed_thinking_rate={summary.closed_thinking_rate:.4f}")
    print(f"reasons={dict(sorted(reasons.items()))}")

    for index, row in enumerate(rows[: args.show]):
        print(f"\n===== sample={index} score={row.get('score')} reason={row.get('reason')} =====")
        print(f"pred={row.get('pred')}")
        print(str(row.get("output", ""))[-args.tail_chars :])


if __name__ == "__main__":
    main()
