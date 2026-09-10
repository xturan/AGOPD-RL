from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _messages(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in value
        if isinstance(item, dict)
    ]


def _prompt_text(tokenizer, value):
    return tokenizer.apply_chat_template(
        _messages(value), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _eos_probability(model, tokenizer, prompt, output):
    text = prompt + output
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
        probability = torch.softmax(logits.float(), dim=-1)[0, tokenizer.eos_token_id].item()
    return probability, int(inputs["input_ids"].shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EOS probabilities at saved rollout endpoints.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--model", action="append", nargs=3, metavar=("NAME", "MODEL", "ROLLOUT"), required=True)
    args = parser.parse_args()

    dataframe = pd.read_parquet(args.data)
    prompts = {}
    for _, row in dataframe.iterrows():
        prompts[str(row["extra_info"].get("index", ""))] = row["prompt"]

    rollout_sets = {}
    for name, model_path, rollout_path in args.model:
        rows = [json.loads(line) for line in Path(rollout_path).read_text().splitlines()]
        rollout_sets[name] = (model_path, rows)

    common_ids = [row["prompt_id"] for row in rollout_sets[next(iter(rollout_sets))][1]]
    for _, rows in rollout_sets.values():
        common_ids = [prompt_id for prompt_id in common_ids if any(row["prompt_id"] == prompt_id for row in rows)]
    common_ids = common_ids[: args.limit]

    records = []
    for name, (model_path, rows) in rollout_sets.items():
        by_id = {row["prompt_id"]: row for row in rows}
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.model_max_length = 50000
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for prompt_id in common_ids:
            row = by_id[prompt_id]
            prompt = _prompt_text(tokenizer, prompts[prompt_id])
            eos_probability, sequence_length = _eos_probability(model, tokenizer, prompt, row["output"])
            records.append(
                {
                    "model": name,
                    "prompt_id": prompt_id,
                    "response_length": row["response_length"],
                    "truncated": row["response_length"] >= 2048,
                    "correct": row["correct"],
                    "eos_probability_at_endpoint": eos_probability,
                    "sequence_length": sequence_length,
                }
            )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = {}
    for name in rollout_sets:
        values = [row["eos_probability_at_endpoint"] for row in records if row["model"] == name]
        summary[name] = {
            "samples": len(values),
            "eos_probability_mean": sum(values) / len(values) if values else 0.0,
            "eos_probability_min": min(values) if values else 0.0,
            "eos_probability_max": max(values) if values else 0.0,
        }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
