from __future__ import annotations

import argparse
import json
from pathlib import Path

from agopd.reward.math_reward import answers_equivalent, extract_answer_line, extract_semantic_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute semantic and contract metrics from saved outputs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    rows = []
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        semantic_answer = extract_semantic_answer(row["output"])
        answer_line = extract_answer_line(row["output"])
        semantic_correct = semantic_answer is not None and answers_equivalent(semantic_answer, row["ground_truth"])
        strict_contract_correct = bool(answer_line and "=" not in answer_line and semantic_correct)
        row.update(
            {
                "semantic_answer": semantic_answer or "[INVALID]",
                "semantic_correct": semantic_correct,
                "strict_contract_correct": strict_contract_correct,
                "parse_success": semantic_answer is not None,
                "truncated": row["response_length"] >= args.max_new_tokens,
            }
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    total = len(rows)
    summary = {
        "input": str(args.input),
        "rows": total,
        "raw_accuracy": sum(bool(row["correct"]) for row in rows) / total if total else 0.0,
        "semantic_accuracy": sum(bool(row["semantic_correct"]) for row in rows) / total if total else 0.0,
        "strict_contract_accuracy": sum(bool(row["strict_contract_correct"]) for row in rows) / total if total else 0.0,
        "answer_parse_rate": sum(bool(row["parse_success"]) for row in rows) / total if total else 0.0,
        "answer_prefix_rate": sum(bool(row["answer_marker"]) for row in rows) / total if total else 0.0,
        "boxed_rate": sum(bool(row["boxed_marker"]) for row in rows) / total if total else 0.0,
        "truncation_rate": sum(bool(row["truncated"]) for row in rows) / total if total else 0.0,
        "non_truncated_semantic_accuracy": (
            sum(bool(row["semantic_correct"]) for row in rows if not row["truncated"])
            / max(1, sum(not row["truncated"] for row in rows))
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
