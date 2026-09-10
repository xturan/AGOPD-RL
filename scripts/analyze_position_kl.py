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


def _logits(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=40960)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs).logits[0].float()
    return output, int(inputs["input_ids"].shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure teacher-student KL by response position.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    dataframe = pd.read_parquet(args.data)
    prompt_by_id = {
        str(row["extra_info"].get("index", "")): row["prompt"] for _, row in dataframe.iterrows()
    }
    rows = [json.loads(line) for line in args.rollout.read_text().splitlines()][: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    tokenizer.model_max_length = 50000
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map="cuda")
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, device_map="cuda")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    for model in (teacher, student, base):
        model.eval()

    bucket_names = ("early_0_25", "mid_25_50", "late_50_75", "final_75_100")
    records = []
    for row in rows:
        text = _prompt(tokenizer, prompt_by_id[row["prompt_id"]]) + row["output"]
        teacher_logits, sequence_length = _logits(teacher, tokenizer, text)
        student_logits, _ = _logits(student, tokenizer, text)
        base_logits, _ = _logits(base, tokenizer, text)
        prompt_tokens = len(tokenizer(_prompt(tokenizer, prompt_by_id[row["prompt_id"]])).input_ids)
        response_tokens = min(row["response_length"], sequence_length - prompt_tokens)
        start = max(0, prompt_tokens - 1)
        end = min(sequence_length - 1, start + response_tokens)

        teacher_logp = torch.log_softmax(teacher_logits[start:end], dim=-1)
        student_logp = torch.log_softmax(student_logits[start:end], dim=-1)
        base_logp = torch.log_softmax(base_logits[start:end], dim=-1)
        teacher_prob = teacher_logp.exp()
        teacher_student_kl = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
        teacher_base_kl = (teacher_prob * (teacher_logp - base_logp)).sum(dim=-1)
        values = {"prompt_id": row["prompt_id"], "response_tokens": int(end - start)}
        for index, bucket in enumerate(bucket_names):
            left = (end - start) * index // 4
            right = (end - start) * (index + 1) // 4
            values[bucket + "_teacher_student_kl"] = teacher_student_kl[left:right].mean().item()
            values[bucket + "_teacher_base_kl"] = teacher_base_kl[left:right].mean().item()
        records.append(values)
        del teacher_logits, student_logits, base_logits, teacher_logp, student_logp, base_logp
        gc.collect()
        torch.cuda.empty_cache()

    summary = {"samples": len(records)}
    for bucket in bucket_names:
        for comparison in ("teacher_student", "teacher_base"):
            key = bucket + "_" + comparison + "_kl"
            values = [row[key] for row in records]
            summary[key] = sum(values) / len(values) if values else 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"output={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
