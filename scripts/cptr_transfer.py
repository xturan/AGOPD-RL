#!/usr/bin/env python3
"""Matched short transfer for the CRPTR experiment.

Modes:
  rft_root: supervised CE on verifier-confirmed teacher root solutions.
  root_kd:  teacher token distribution on the same root solutions.
  path_kd:  teacher token distribution after a student wrong prefix.

All modes start from the same base student and use the same prompt subset,
number of optimizer steps, sequence cap, and learning rate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("rft_root", "root_kd", "path_kd"), required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model")
    p.add_argument("--student-input", type=Path, required=True)
    p.add_argument("--teacher-input", type=Path, required=True)
    p.add_argument("--control-input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-prompts", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strip_no_think(content: str) -> str:
    return content.replace("\n\n/no_think", "").replace("\n/no_think", "")


def with_no_think(content: str) -> str:
    return strip_no_think(content) + "\n\n/no_think"


def path_context(content: str, prefix: str) -> str:
    return (
        strip_no_think(content)
        + "\n\nThe draft below may be wrong. Continue the solution, correct any mistakes you detect, and give the final answer.\n\n"
        + "<student_draft>\n"
        + prefix
        + "\n</student_draft>\n\n/no_think"
    )


def make_examples(args: argparse.Namespace) -> list[dict]:
    students = {r["idx"]: r for r in load_jsonl(args.student_input)}
    teacher = defaultdict(dict)
    for r in load_jsonl(args.teacher_input):
        teacher[r["idx"]][r["condition"]] = r
    control = defaultdict(dict)
    for r in load_jsonl(args.control_input):
        control[r["idx"]][r["condition"]] = r

    examples = []
    for idx in sorted(students):
        student = students[idx]
        root = teacher[idx].get("root")
        path = control[idx].get("wrong_path_matched")
        if not root or not path:
            continue
        root_solution = next((s["text"] for s in root["teacher_samples"] if s.get("correct")), None)
        path_solution = path["teacher_samples"][0]["text"] if path.get("teacher_samples") else None
        if not root_solution or not path_solution:
            continue
        if args.mode in {"rft_root", "root_kd"}:
            content, completion = with_no_think(student["content"]), root_solution
        else:
            prefix = path.get("student_prefix", "")
            content, completion = path_context(student["content"], prefix), path_solution
        examples.append({
            "idx": idx,
            "gt": student["gt"],
            "content": content,
            "completion": completion,
            "mode": args.mode,
        })
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    return examples[: args.max_prompts]


def encode_example(tokenizer, example: dict, max_length: int) -> tuple:
    prompt_ids = tokenizer(example["content"], add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    # Keep the response end, where the verifier-relevant answer lives.
    room = max_length - len(prompt_ids)
    if room <= 8:
        raise ValueError(f"prompt too long for {max_length}: {example['idx']}")
    completion_ids = completion_ids[:room]
    ids = prompt_ids + completion_ids
    response_start = len(prompt_ids)
    return ids, response_start


def main() -> None:
    args = parse_args()
    if args.mode != "rft_root" and not args.teacher_model:
        raise ValueError("--teacher-model is required for KD modes")
    random.seed(args.seed)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(f"cuda:{args.gpu}")
    examples = make_examples(args)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    encoded = []
    for example in examples:
        try:
            ids, response_start = encode_example(tokenizer, example, args.max_seq_length)
        except ValueError:
            continue
        encoded.append((example, ids, response_start))
    if not encoded:
        raise RuntimeError("no examples fit the sequence length")

    dtype = torch.bfloat16
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    student.config.use_cache = False
    student.gradient_checkpointing_enable()
    student.train()
    teacher = None
    if args.mode != "rft_root":
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)
        teacher.config.use_cache = False
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "metrics.jsonl"
    rng = random.Random(args.seed)
    with log_path.open("w") as log:
        for step in range(1, args.steps + 1):
            batch = [
                encoded[((step - 1) * args.batch_size + offset) % len(encoded)]
                for offset in range(args.batch_size)
            ]
            max_length = max(len(ids) for _, ids, _ in batch)
            input_ids = torch.full(
                (len(batch), max_length), pad_token_id, dtype=torch.long, device=device
            )
            labels = torch.full_like(input_ids, -100)
            for row_index, (_, ids, response_start) in enumerate(batch):
                length = len(ids)
                input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
                labels[row_index, response_start:length] = input_ids[row_index, response_start:length]
            attention_mask = input_ids.ne(pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "rft_root":
                output = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
            else:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
                target = teacher_logits.float().log_softmax(dim=-1).exp()
                log_student = student_logits.float().log_softmax(dim=-1)
                token_loss = -(target * log_student).sum(dim=-1)
                mask = labels[:, 1:] != -100
                loss = token_loss.masked_select(mask).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.cpu()),
                "idx": [example["idx"] for example, _, _ in batch],
                "batch_size": len(batch),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"mode={args.mode} step={step}/{args.steps} batch={len(batch)} "
                f"loss={row['loss']:.6f} grad_norm={row['grad_norm']:.5f}",
                flush=True,
            )

    student.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"transfer_saved={args.output} examples={len(encoded)}", flush=True)


if __name__ == "__main__":
    main()
