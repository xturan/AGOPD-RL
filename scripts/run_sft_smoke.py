from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact Qwen3 LoRA SFT smoke job.")
    parser.add_argument("--data", default="data/openr1-sft/smoke-8000.jsonl")
    parser.add_argument("--eval-data", default=None, help="Validation jsonl for early stopping")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=0, help="Save checkpoint every N steps (0=off)")
    parser.add_argument("--patience", type=int, default=4, help="Early-stop patience in eval checks")
    parser.add_argument("--max-epochs", type=float, default=3.0)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--output", default="outputs/sft-smoke-qwen3-1.7b")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    return parser.parse_args()


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.examples = []
        for row in rows:
            messages = row["messages"]
            # transformers >= 5.8 的 apply_chat_template(tokenize=True) 返回
            # 重构后的 BatchEncoding(Encoding 包装),不再是 list;改用
            # 文本渲染 → 常规 tokenize 的经典路径
            prompt_text = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
            full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            prompt_ids = tokenizer(prompt_text)["input_ids"]
            full_ids = tokenizer(full_text)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("SFT chat template prefix does not match completion template")
            if len(full_ids) > max_length:
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "labels": torch.tensor([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :], dtype=torch.long),
                }
            )
        if not self.examples:
            raise RuntimeError("No SFT examples fit within max_length")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].shape[0] for feature in features)
        input_ids = torch.full((len(features), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for index, feature in enumerate(features):
            length = feature["input_ids"].shape[0]
            input_ids[index, :length] = feature["input_ids"]
            labels[index, :length] = feature["labels"]
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules="all-linear",
        ),
    )

    dataset = ChatSFTDataset(Path(args.data), tokenizer, args.max_length)
    print(f"sft_examples={len(dataset)}")
    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.max_steps,
        num_train_epochs=args.max_epochs,
        eval_strategy="steps" if args.eval_data else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or args.eval_steps,
        load_best_model_at_end=True if args.eval_data else False,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        report_to="none",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=42,
    )
    eval_dataset = (
        ChatSFTDataset(Path(args.eval_data), tokenizer, args.max_length) if args.eval_data else None
    )
    if eval_dataset is not None:
        print(f"eval_examples={len(eval_dataset)}")
    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)] if eval_dataset else [],
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
