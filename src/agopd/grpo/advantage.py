from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class AdvantageStats:
    advantages: list[float]
    reward_mean: float
    reward_std: float
    zero_variance: bool


def compute_group_relative_advantages(
    rewards: Sequence[float],
    eps: float = 1.0e-6,
) -> AdvantageStats:
    if not rewards:
        raise ValueError("rewards must be non-empty")

    reward_values = [float(reward) for reward in rewards]
    reward_mean = mean(reward_values)
    variance = sum((reward - reward_mean) ** 2 for reward in reward_values) / len(reward_values)
    reward_std = sqrt(variance)

    if reward_std < eps:
        return AdvantageStats(
            advantages=[0.0 for _ in reward_values],
            reward_mean=reward_mean,
            reward_std=reward_std,
            zero_variance=True,
        )

    return AdvantageStats(
        advantages=[(reward - reward_mean) / (reward_std + eps) for reward in reward_values],
        reward_mean=reward_mean,
        reward_std=reward_std,
        zero_variance=False,
    )
