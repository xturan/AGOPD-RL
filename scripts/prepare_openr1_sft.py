from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agopd.data.openr1 import build_math_sft_prompt, finalize_solution, is_compact_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one OpenR1-Math parquet shard into compact SFT JSONL.")
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", default="data/openr1-sft/train.jsonl")
    parser.add_argument("--limit", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-solution-chars", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for shard_index, input_path in enumerate(args.input):
            dataframe = pd.read_parquet(input_path, columns=["problem", "solution", "answer", "uuid"])
            dataframe = dataframe.dropna(subset=["problem", "solution", "answer"])
            dataframe = dataframe.sample(frac=1, random_state=args.seed + shard_index)
            for row in dataframe.to_dict("records"):
                raw_solution = str(row["solution"]).strip()
                answer = str(row["answer"]).strip()
                if not is_compact_solution(raw_solution, args.max_solution_chars):
                    continue
                solution = finalize_solution(raw_solution, answer)
                record = {
                    "messages": [
                        {"role": "user", "content": build_math_sft_prompt(str(row["problem"]))},
                        {"role": "assistant", "content": solution},
                    ],
                    "ground_truth": answer,
                    "source": "open-r1/OpenR1-Math-220k",
                    "source_id": str(row["uuid"]),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                if accepted >= args.limit:
                    break
            if accepted >= args.limit:
                break

    print(f"inputs={len(args.input)}")
    print(f"output={output_path}")
    print(f"accepted={accepted}")


if __name__ == "__main__":
    main()
