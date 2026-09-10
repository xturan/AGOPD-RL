from __future__ import annotations

import argparse

from agopd.data.splits import create_dapo_verl_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create de-duplicated DAPO train/val parquet files for verl smoke runs.")
    parser.add_argument("--input", default="data/dapo-math-17k/data/dapo-math-17k.parquet")
    parser.add_argument("--output-dir", default="data/dapo-verl-smoke")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append-no-think", action="store_true")
    args = parser.parse_args()

    result = create_dapo_verl_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        append_no_think=args.append_no_think,
    )
    print(f"train_path={result.train_path}")
    print(f"train_rows={result.train_rows}")
    print(f"val_path={result.val_path}")
    print(f"val_rows={result.val_rows}")
    print("prepare_dapo_splits=ok")


if __name__ == "__main__":
    main()
