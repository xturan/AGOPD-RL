from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import swanlab
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _read_scalars(event_path: str) -> list[tuple[str, int, float]]:
    accumulator = EventAccumulator(event_path, size_guidance={"scalars": 0})
    accumulator.Reload()
    values = []
    for tag in accumulator.Tags().get("scalars", []):
        values.extend((tag, item.step, item.value) for item in accumulator.Scalars(tag))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge TensorBoard scalar events into an offline SwanLab run.")
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--swanlog-dir", required=True, type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("local", "offline"), default="local")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()

    args.swanlog_dir.mkdir(parents=True, exist_ok=True)
    run = swanlab.init(
        mode=args.mode,
        project=args.project,
        name=args.name,
        log_dir=str(args.swanlog_dir),
        config={"tensorboard_dir": str(args.tensorboard_dir)},
    )
    seen: set[tuple[str, str, int]] = set()
    try:
        while True:
            for event_path in glob.glob(str(args.tensorboard_dir / "**" / "events.out.tfevents.*"), recursive=True):
                for tag, step, value in _read_scalars(event_path):
                    key = (event_path, tag, step)
                    if key not in seen:
                        run.log({tag: value}, step=step)
                        seen.add(key)
            if not args.follow:
                break
            time.sleep(args.interval)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
