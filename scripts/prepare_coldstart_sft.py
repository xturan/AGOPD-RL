"""Clean and convert cold-start SFT trajectories into HF chat-format jsonl.

Filters (overfitting / distribution-mismatch guards):
  - trajectories longer than max_tokens (they would be truncated in RL rollout)
  - outputs with >= 3 consecutive repeated lines (struggle artifacts)
  - outputs without an Answer:/boxed contract

Splits train/val (seeded) so the SFT run can early-stop on val loss.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean cold-start SFT data into HF chat format.")
    parser.add_argument("--input", default="outputs/coldstart-5000/sft_data.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("data/coldstart-sft"))
    parser.add_argument("--max-tokens", type=int, default=1800, help="Drop longer trajectories")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _has_contract(text: str) -> bool:
    return ("Answer:" in text) or ("boxed" in text) or ("answer" in text.lower())


def _has_repeat_artifact(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i in range(2, len(lines)):
        if lines[i] == lines[i - 1] == lines[i - 2]:
            return True
    return False


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped_long, dropped_rep, dropped_contract = [], 0, 0, 0
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output = row["output"]
        approx_tokens = len(output) / 3.5
        if approx_tokens > args.max_tokens:
            dropped_long += 1
            continue
        if _has_repeat_artifact(output):
            dropped_rep += 1
            continue
        if not _has_contract(output):
            dropped_contract += 1
            continue
        kept.append(
            {
                "index": row["index"],
                "source": row["source"],
                "messages": [
                    {"role": "user", "content": row["prompt_messages"][0]["content"]},
                    {"role": "assistant", "content": output},
                ],
            }
        )

    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_fraction))
    val_rows, train_rows = kept[:n_val], kept[n_val:]

    def write(name: str, rows: list[dict]) -> None:
        with (out_dir / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write("train.jsonl", train_rows)
    write("val.jsonl", val_rows)
    print(
        f"kept={len(kept)} (train={len(train_rows)}, val={len(val_rows)}), "
        f"dropped: long={dropped_long} repeat={dropped_rep} no_contract={dropped_contract}"
    )
    print(f"wrote {out_dir}/train.jsonl and {out_dir}/val.jsonl")


if __name__ == "__main__":
    main()
