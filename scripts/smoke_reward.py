from __future__ import annotations

import argparse
from pathlib import Path

from agopd.data import load_dapo_math_samples
from agopd.grpo import compute_group_relative_advantages
from agopd.reward import score_math_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test DAPO loader, math reward, and GRPO advantage.")
    parser.add_argument(
        "--dataset",
        default="data/dapo-math-17k/data/dapo-math-17k.parquet",
        help="Path to local DAPO-Math parquet.",
    )
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    samples = load_dapo_math_samples(Path(args.dataset), limit=args.limit)
    if not samples:
        raise RuntimeError("No samples loaded.")

    print(f"loaded_samples={len(samples)}")
    first = samples[0]
    print(f"first_prompt_id={first.prompt_id}")
    print(f"first_ground_truth={first.ground_truth}")
    print(f"first_prompt_preview={first.prompt[:220].replace(chr(10), ' ')}")

    self_check_rewards = []
    for sample in samples:
        result = score_math_response(f"Reasoning omitted.\nAnswer: {sample.ground_truth}", sample.ground_truth)
        self_check_rewards.append(result.reward)
        if not result.correct:
            raise AssertionError(f"Self-check failed for {sample.prompt_id}: {result}")

    wrong_result = score_math_response("Reasoning omitted.\nAnswer: definitely_wrong", first.ground_truth)
    if wrong_result.reward != 0.0:
        raise AssertionError(f"Wrong-answer check failed: {wrong_result}")

    demo_stats = compute_group_relative_advantages([1, 0, 0, 1])
    zero_stats = compute_group_relative_advantages([0, 0, 0, 0])
    print(f"self_check_reward_mean={sum(self_check_rewards) / len(self_check_rewards):.3f}")
    print(f"wrong_answer_reward={wrong_result.reward:.1f}")
    print(f"demo_advantages={[round(value, 4) for value in demo_stats.advantages]}")
    print(f"zero_variance_group={zero_stats.zero_variance}")
    print("smoke_reward=ok")


if __name__ == "__main__":
    main()
