from __future__ import annotations

import os
from typing import Any

from agopd.reward.math_reward import score_math_response

# ---- Wrong-length penalty (configurable, default OFF) ----
# Reward for a wrong answer: -C * min(1, len(chars) / MAX_CHARS).
# Targets the "write-to-fill" failure mode (verbose dead-ends get the same
# reward as quick failures today): the longer the wrong attempt, the more
# negative. Correct answers stay +1 regardless of length, so long *correct*
# reasoning (the 4096-budget direction) is not taxed.
# Enable by setting WRONG_LEN_PENALTY_C (e.g. "1.0") in the training env.
_WLP_C = float(os.environ.get("WRONG_LEN_PENALTY_C", "0.0"))
_WLP_MAX_CHARS = int(os.environ.get("WRONG_LEN_MAX_CHARS", "8000"))  # ~2048 tok


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt the project 0/1 math verifier to verl's reward API."""
    del data_source, extra_info
    result = score_math_response(solution_str, ground_truth)
    score = result.reward
    if (not result.correct) and _WLP_C > 0.0:
        frac = min(1.0, len(solution_str) / max(1, _WLP_MAX_CHARS))
        score = -_WLP_C * frac
    return {
        "score": score,
        "acc": result.correct,
        "pred": result.predicted_answer or "[INVALID]",
        "reason": result.reason,
    }
