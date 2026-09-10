from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AdvantageGateConfig:
    threshold: float = 0.0
    max_weight: float = 2.0


@dataclass(frozen=True)
class GateDecision:
    trajectory_weight: float
    token_mask: list[bool]
    token_weights: list[float]
    reason: str
    budget_used: int = 0

    @property
    def activated_tokens(self) -> int:
        return sum(self.token_mask)


class AdvantageGate:
    def __init__(self, config: AdvantageGateConfig | None = None) -> None:
        self.config = config or AdvantageGateConfig()

    def decide(self, advantage: float | None, loss_mask: Sequence[int | bool]) -> GateDecision:
        response_mask = [bool(value) for value in loss_mask]
        if advantage is None:
            return GateDecision(0.0, [False for _ in response_mask], [0.0 for _ in response_mask], "missing_advantage")

        if advantage >= self.config.threshold:
            return GateDecision(0.0, [False for _ in response_mask], [0.0 for _ in response_mask], "non_negative_advantage")

        trajectory_weight = min(max(-float(advantage), 0.0), self.config.max_weight)
        token_mask = response_mask
        token_weights = [trajectory_weight if enabled else 0.0 for enabled in token_mask]
        return GateDecision(trajectory_weight, token_mask, token_weights, "negative_advantage")
