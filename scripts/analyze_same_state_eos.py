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


def _prompt(tokenizer, value):
    return tokenizer.apply_chat_template(
        _messages(value), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EOS probabilities on identical OPD prefixes.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = parser.parse_args()

    dataframe = pd.read_parquet(args.data)
    prompt_by_id = {
        str(row["extra_info"].get("index", "")): row["prompt"] for _, row in dataframe.iterrows()
    }
    rows = [json.loads(line) for line in args.rollout.read_text().splitlines()][: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model[0][1])
    tokenizer.model_max_length = 50000

    inputs_by_id = {}
    for row in rows:
        prompt = _prompt(tokenizer, prompt_by_id[row["prompt_id"]])
        encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids[0]
        encoded_full = tokenizer(prompt + row["output"], return_tensors="pt", truncation=True, max_length=40960)
        inputs_by_id[row["prompt_id"]] = (encoded_prompt, encoded_full.input_ids[0], row)

    fractions = (0.25, 0.50, 0.75, 0.90, 1.00)
    records = {prompt_id: {"prompt_id": prompt_id} for prompt_id in inputs_by_id}
    for name, model_path in args.model:
        model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for prompt_id, (encoded_prompt, encoded_full, row) in inputs_by_id.items():
            prompt_len = min(len(encoded_prompt), len(encoded_full))
            response_len = max(1, len(encoded_full) - prompt_len)
            input_ids = encoded_full.unsqueeze(0).to(model.device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits[0].float()
            values = []
            for fraction in fractions:
                prefix_len = min(response_len, max(1, int(response_len * fraction)))
                logit_index = min(len(logits) - 1, prompt_len - 1 + prefix_len)
                values.append(torch.softmax(logits[logit_index], dim=-1)[model_tokenizer.eos_token_id].item())
            records[prompt_id][name] = values
            records[prompt_id]["response_tokens"] = response_len
            records[prompt_id]["truncated"] = row["response_length"] >= 2048
        del model, model_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    result_rows = list(records.values())
    summary = {"samples": len(result_rows), "fractions": fractions}
    for name, _ in args.model:
        values = [row[name] for row in result_rows]
        for index, fraction in enumerate(fractions):
            bucket = f"p{int(fraction * 100):02d}"
            summary[f"{name}_{bucket}_eos_mean"] = sum(item[index] for item in values) / len(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in result_rows))
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
