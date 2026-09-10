from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agopd.data.sft import build_compact_teacher_prompt
from agopd.reward.math_reward import score_math_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verifier-filtered compact SFT examples with a teacher.")
    parser.add_argument("--input", default="data/dapo-math-17k/data/dapo-math-17k.parquet")
    parser.add_argument("--output", default="data/teacher-sft-qwen3-4b/train.jsonl")
    parser.add_argument(
        "--attempt-output",
        default=None,
        help="Optional JSONL path that records accepted and rejected teacher attempts.",
    )
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher SFT generation")

    rows = _load_unique_rows(Path(args.input), args.limit, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path = Path(args.attempt_output) if args.attempt_output else None
    if attempt_path is not None:
        attempt_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model.eval()

    sampling = (
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        if args.enable_thinking
        else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    )
    accepted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        attempt_handle = attempt_path.open("w", encoding="utf-8") if attempt_path is not None else None
        for index, row in enumerate(rows, start=1):
            problem = _user_content(row["prompt"])
            target = str(row["reward_model"]["ground_truth"])
            prompt = build_compact_teacher_prompt(problem)
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda:0")
            started_at = time.perf_counter()
            print(f"generating={index}/{len(rows)}", flush=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    **sampling,
                )
            response = tokenizer.decode(
                output_ids[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
            reward = score_math_response(response, target)
            attempt_record = {
                "source_index": row["extra_info"]["index"],
                "ground_truth": target,
                "response": response,
                "correct": reward.correct,
                "pred": reward.predicted_answer,
                "reason": reward.reason,
            }
            if attempt_handle is not None:
                attempt_handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
            if reward.correct:
                record = {
                    "messages": messages + [{"role": "assistant", "content": response}],
                    "ground_truth": target,
                    "source_index": row["extra_info"]["index"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
            elapsed = time.perf_counter() - started_at
            print(
                f"generated={index}/{len(rows)} accepted={accepted} "
                f"correct={reward.correct} seconds={elapsed:.1f}",
                flush=True,
            )
        if attempt_handle is not None:
            attempt_handle.close()

    print(f"output={output_path}")
    print(f"requested={len(rows)}")
    print(f"accepted={accepted}")
    if attempt_path is not None:
        print(f"attempt_output={attempt_path}")


def _load_unique_rows(path: Path, limit: int, seed: int) -> list[dict]:
    dataframe = pd.read_parquet(path)
    dataframe = dataframe.assign(_index=dataframe["extra_info"].map(lambda value: value["index"]))
    dataframe = dataframe.drop_duplicates("_index", keep="first").drop(columns="_index")
    return dataframe.sample(n=min(limit, len(dataframe)), random_state=seed).to_dict("records")


def _user_content(messages: list[dict]) -> str:
    return "\n".join(str(message["content"]) for message in messages if message["role"] == "user")


if __name__ == "__main__":
    main()
