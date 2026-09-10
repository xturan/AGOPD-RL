from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)


_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


@dataclass(frozen=True)
class MathRewardResult:
    reward: float
    predicted_answer: str | None
    ground_truth: str
    correct: bool
    reason: str


def score_math_response(response_text: str, ground_truth: str) -> MathRewardResult:
    predicted = extract_final_answer(response_text)
    if predicted is None:
        return MathRewardResult(0.0, None, str(ground_truth), False, "missing_answer")

    correct = answers_equivalent(predicted, ground_truth)
    return MathRewardResult(
        reward=1.0 if correct else 0.0,
        predicted_answer=predicted,
        ground_truth=str(ground_truth),
        correct=correct,
        reason="correct" if correct else "wrong_answer",
    )


def extract_final_answer(response_text: str) -> str | None:
    text = response_text.strip()
    if not text:
        return None

    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed.strip()

    answer_matches = list(re.finditer(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text))
    if answer_matches:
        return answer_matches[-1].group(1).strip()

    inline_matches = list(re.finditer(r"(?i)answer\s*[:=]\s*([^\n]+)", text))
    if inline_matches:
        return inline_matches[-1].group(1).strip()

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return None
    return nonempty_lines[-1]


def extract_answer_line(response_text: str) -> str | None:
    """Extract the last explicit Answer: line without applying semantic cleanup."""
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:\*\*)?\s*(?:final\s+)?answer\s*:\s*(.+?)\s*(?:\*\*)?\s*$",
            response_text.strip(),
        )
    )
    if not matches:
        return None
    value = matches[-1].group(1).strip()
    if value.startswith("**"):
        value = value[2:].strip()
    if value.endswith("**"):
        value = value[:-2].strip()
    return value


def extract_semantic_answer(response_text: str) -> str | None:
    """Extract an answer for evaluation, preferring the explicit Answer: contract.

    This path tolerates a simple final equality such as ``11 + 60 = 71``. The
    training reward intentionally continues to use ``extract_final_answer``.
    """
    answer_line = extract_answer_line(response_text)
    if answer_line is not None:
        return _semantic_candidate(answer_line)

    boxed = _extract_last_boxed(response_text)
    if boxed:
        return _semantic_candidate(boxed.strip())

    text = response_text.strip()
    if not text:
        return None
    candidate = _semantic_candidate(text.splitlines()[-1].strip())
    return candidate if _is_numeric_expression(candidate) else None


def _semantic_candidate(value: str) -> str:
    """Unwrap a final answer while preserving simple equality handling."""
    boxed = _extract_last_boxed(value)
    if boxed:
        value = boxed.strip()
    return _rhs_if_equality(value)


def _rhs_if_equality(value: str) -> str:
    parts = [part.strip() for part in value.split("=") if part.strip()]
    if len(parts) < 2:
        return value.strip()
    if _is_numeric_expression(parts[-1]) and _is_numeric_expression(parts[-2]):
        return parts[-1]
    return value.strip()


def _is_numeric_expression(value: str) -> bool:
    normalized = normalize_answer(value)
    expression = _to_sympy_expr(normalized)
    return isinstance(expression, sp.Expr) and not expression.free_symbols


def answers_equivalent(predicted: str, ground_truth: str) -> bool:
    pred = normalize_answer(predicted)
    gold = normalize_answer(ground_truth)
    if pred == gold:
        return True

    pred_expr = _to_sympy_expr(pred)
    gold_expr = _to_sympy_expr(gold)
    if pred_expr is None or gold_expr is None:
        return False

    try:
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def normalize_answer(answer: str) -> str:
    value = str(answer).strip()
    value = _strip_math_wrappers(value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", value)
    value = _replace_latex_frac(value)
    value = _replace_latex_sqrt(value)
    value = value.replace("^", "**")
    return value


def _strip_math_wrappers(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("$"):
        stripped = stripped[1:]
    if stripped.endswith("$"):
        stripped = stripped[:-1]
    if stripped.startswith("\\(") and stripped.endswith("\\)"):
        stripped = stripped[2:-2]
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        stripped = stripped[2:-2]
    return stripped.strip().rstrip(".")


def _replace_latex_frac(value: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"((\1)/(\2))", value)
    return value


def _replace_latex_sqrt(value: str) -> str:
    return re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    idx = start + len(marker)
    depth = 1
    chars = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
        chars.append(char)
        idx += 1
    return None


def _to_sympy_expr(value: str) -> sp.Expr | None:
    try:
        return sp.parse_expr(value, transformations=_TRANSFORMATIONS, evaluate=True)
    except Exception:
        return None
