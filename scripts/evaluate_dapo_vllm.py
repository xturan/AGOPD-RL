from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from agopd.reward.math_reward import (
    answers_equivalent,
    extract_answer_line,
    extract_semantic_answer,
    score_math_response,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one Hugging Face model on the fixed DAPO validation set with vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-adapter", type=Path)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--data", default="data/dapo-verl-v1/val.parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Remove a trailing /no_think marker and enable Qwen3 thinking mode.",
    )
    return parser.parse_args()


def _messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return [{"role": "user", "content": str(value)}]
    return [
        {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
        for message in value
        if isinstance(message, dict)
    ]


def _id(value: Any) -> str:
    return str(value.get("index") or value.get("id") or "") if isinstance(value, dict) else ""


def _ground_truth(value: Any) -> str:
    return str(value.get("ground_truth") or "") if isinstance(value, dict) else ""


def _render_with_mode(
    tokenizer: Any, messages: list[dict[str, str]], thinking: bool
) -> str:
    if thinking:
        messages = [dict(message) for message in messages]
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] = re.sub(r"\s*/no_think\s*$", "", message["content"])
                break
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _has_boxed(text: str) -> bool:
    return "\\boxed{" in text


def _has_answer_marker(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*answer\s*:", text))


def main() -> None:
    args = _parse_args()
    dataframe = pd.read_parquet(args.data)
    if args.limit is not None:
        dataframe = dataframe.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        _render_with_mode(tokenizer, _messages(value), args.thinking)
        for value in dataframe["prompt"]
    ]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len or max(4096, args.max_new_tokens + 1024),
        enforce_eager=True,
    )
    lora_request = None
    if args.lora_adapter is not None:
        llm_kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=args.lora_rank)
        lora_request = LoRARequest("evaluation_adapter", 1, str(args.lora_adapter))
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    rows: list[dict[str, Any]] = []
    for source, result in zip(dataframe.to_dict("records"), outputs):
        output = result.outputs[0].text
        score = score_math_response(output, _ground_truth(source["reward_model"]))
        semantic_answer = extract_semantic_answer(output)
        answer_line = extract_answer_line(output)
        semantic_correct = semantic_answer is not None and answers_equivalent(
            semantic_answer, _ground_truth(source["reward_model"])
        )
        strict_contract_correct = bool(
            answer_line
            and "=" not in answer_line
            and semantic_correct
        )
        rows.append(
            {
                "prompt_id": _id(source["extra_info"]),
                "ground_truth": score.ground_truth,
                "output": output,
                "pred": score.predicted_answer or "[INVALID]",
                "score": score.reward,
                "correct": score.correct,
                "semantic_answer": semantic_answer or "[INVALID]",
                "semantic_correct": semantic_correct,
                "strict_contract_correct": strict_contract_correct,
                "parse_success": semantic_answer is not None,
                "reason": score.reason,
                "answer_marker": _has_answer_marker(output),
                "boxed_marker": _has_boxed(output),
                "closed_thinking": "<think>" not in output or "</think>" in output,
                "response_length": len(result.outputs[0].token_ids),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    total = len(rows)
    summary = {
        "model": args.model,
        "data": args.data,
        "rows": total,
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(float(row["score"]) for row in rows) / total if total else 0.0,
        "semantic_accuracy": sum(bool(row["semantic_correct"]) for row in rows) / total if total else 0.0,
        "strict_contract_accuracy": sum(bool(row["strict_contract_correct"]) for row in rows) / total if total else 0.0,
        "answer_parse_rate": sum(bool(row["parse_success"]) for row in rows) / total if total else 0.0,
        "answer_marker_rate": sum(bool(row["answer_marker"]) for row in rows) / total if total else 0.0,
        "boxed_marker_rate": sum(bool(row["boxed_marker"]) for row in rows) / total if total else 0.0,
        "closed_thinking_rate": sum(bool(row["closed_thinking"]) for row in rows) / total if total else 0.0,
        "truncation_rate": sum(row["response_length"] >= args.max_new_tokens for row in rows) / total if total else 0.0,
        "response_length_mean": sum(row["response_length"] for row in rows) / total if total else 0.0,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "thinking": args.thinking,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
