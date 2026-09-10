#!/usr/bin/env python3
"""Compare the matched cloud RFT-root and on-policy OPD evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "reports" / "eval-dapo-v1" / "cptr_final"
OPD_FILE = EVAL_DIR / "cptr_standard_opd120_4096.jsonl"
RFT_FILE = EVAL_DIR / "cptr_rft_root_batch8_120step_4096.jsonl"
OUT = ROOT / "reports" / "cptr_rft_vs_opd_comparison_4096.json"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(opd: list[dict], rft: list[dict], key: str, rng: np.random.Generator) -> dict:
    left = np.asarray([bool(row[key]) for row in opd], dtype=np.int8)
    right = np.asarray([bool(row[key]) for row in rft], dtype=np.int8)
    diff = right - left
    draws = rng.integers(0, len(diff), size=(100_000, len(diff)))
    bootstrap = diff[draws].mean(axis=1) * 100
    rft_wins = int(((right == 1) & (left == 0)).sum())
    opd_wins = int(((left == 1) & (right == 0)).sum())
    discordant = rft_wins + opd_wins
    p_value = (
        float(binomtest(rft_wins, n=discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "opd_correct": int(left.sum()),
        "rft_correct": int(right.sum()),
        "difference_pp": float(diff.mean() * 100),
        "paired_bootstrap_95ci_pp": [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
        "rft_wins": rft_wins,
        "opd_wins": opd_wins,
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    opd = load(OPD_FILE)
    rft = load(RFT_FILE)
    if [row["prompt_id"] for row in opd] != [row["prompt_id"] for row in rft]:
        raise ValueError("OPD and RFT files are not in the same prompt order")
    rng = np.random.default_rng(42)
    result = {
        "protocol": {
            "rows": len(opd),
            "max_new_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 42,
            "rft_training": "Base, batch=8, 120 optimizer steps, root verified trajectories",
            "opd_training": "Base, standard on-policy OPD, batch=8, 120 optimizer steps",
        },
        "strict_verifier": compare(opd, rft, "correct", rng),
        "semantic_verifier": compare(opd, rft, "semantic_correct", rng),
        "strict_contract": compare(opd, rft, "strict_contract_correct", rng),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
