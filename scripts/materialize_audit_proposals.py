#!/usr/bin/env python3
"""Map model-proposed error quotes to exact response-token offsets.

Only semantic error categories enter the probe candidate set.  Format-only and
answer-extraction failures remain in the source audit queue but are excluded
from the state-level semantic analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEMANTIC_TYPES = {"arithmetic", "algebra", "interpretation", "unsupported_inference", "other"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="models/Qwen3-1.7B")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = []
    rejected = {"parse": 0, "nonsemantic": 0, "missing_quote": 0}
    for row in rows:
        proposal = row.get("auditor", {})
        if proposal.get("parse_status") != "ok":
            rejected["parse"] += 1
            continue
        error_type = proposal.get("error_type")
        if error_type not in SEMANTIC_TYPES:
            rejected["nonsemantic"] += 1
            continue
        quote = str(proposal.get("error_quote") or "")
        response = row["student_response"]
        occurrences = []
        start = 0
        while True:
            found = response.find(quote, start)
            if found < 0:
                break
            occurrences.append(found)
            start = found + max(1, len(quote))
        if not occurrences:
            rejected["missing_quote"] += 1
            continue
        char_offset = occurrences[0]
        token_offset = len(tokenizer(response[:char_offset], add_special_tokens=False)["input_ids"])
        candidates.append(
            {
                **row,
                "error_quote": quote,
                "error_unit": proposal.get("error_unit"),
                "error_type": error_type,
                "error_evidence": proposal.get("reason"),
                "annotation_confidence": proposal.get("confidence", "low"),
                "auditor": f"candidate:{row.get('auditor_model', 'unknown')}",
                "annotation_status": "candidate_requires_semantic_review",
                "error_onset_char": char_offset,
                "error_onset_token": token_offset,
                "quote_occurrences": len(occurrences),
            }
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"candidate_saved={a.output} candidates={len(candidates)} rejected={rejected}", flush=True)


if __name__ == "__main__":
    main()
