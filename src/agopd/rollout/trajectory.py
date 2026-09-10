from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trajectory:
    prompt_id: str
    prompt: str
    response_tokens: list[int]
    response_text: str
    reward: float
    advantage: float | None
    policy_logprobs: list[float]
    loss_mask: list[int]
    ref_logprobs: list[float] | None = None
    teacher_candidate: bool = False
    teacher_topk_ids: list[list[int]] | None = None
    teacher_topk_logprobs: list[list[float]] | None = None
    disagreement: list[float] | None = None
    opd_weights: list[float] | None = None

    def validate(self) -> None:
        response_len = len(self.response_tokens)
        if len(self.policy_logprobs) != response_len:
            raise ValueError("policy_logprobs length must match response_tokens length")
        if len(self.loss_mask) != response_len:
            raise ValueError("loss_mask length must match response_tokens length")
        if self.ref_logprobs is not None and len(self.ref_logprobs) != response_len:
            raise ValueError("ref_logprobs length must match response_tokens length")
        if self.opd_weights is not None and len(self.opd_weights) != response_len:
            raise ValueError("opd_weights length must match response_tokens length")
