from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutQualitySummary:
    total: int
    correct: int
    answer_marked: int
    boxed: int
    closed_thinking: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def answer_format_rate(self) -> float:
        return self.answer_marked / self.total if self.total else 0.0

    @property
    def boxed_format_rate(self) -> float:
        return self.boxed / self.total if self.total else 0.0

    @property
    def closed_thinking_rate(self) -> float:
        return self.closed_thinking / self.total if self.total else 0.0


def summarize_rollouts(rows: Iterable[Mapping[str, object]]) -> RolloutQualitySummary:
    total = correct = answer_marked = boxed = closed_thinking = 0
    for row in rows:
        total += 1
        output = str(row.get("output", ""))
        score = float(row.get("score", 0.0))
        correct += int(score > 0.0)
        answer_marked += int("answer:" in output.lower())
        boxed += int("\\boxed{" in output)
        closed_thinking += int("</think>" in output)

    return RolloutQualitySummary(
        total=total,
        correct=correct,
        answer_marked=answer_marked,
        boxed=boxed,
        closed_thinking=closed_thinking,
    )
