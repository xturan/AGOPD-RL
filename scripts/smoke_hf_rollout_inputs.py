from __future__ import annotations

import argparse

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce verl HF rollout inputs without FSDP.")
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--data", default="data/dapo-verl-smoke/train.parquet")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--pass-position-ids", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model smoke test")

    table = pq.read_table(args.data, columns=["prompt"])
    messages = table.slice(args.row, 1).to_pylist()[0]["prompt"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenized = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids, attention_mask = postprocess_data(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_length=args.max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="error",
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model.eval()

    generation_kwargs = {
        "input_ids": input_ids.to("cuda:0"),
        "attention_mask": attention_mask.to("cuda:0"),
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": args.max_response_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.pass_position_ids:
        generation_kwargs["position_ids"] = position_ids.to("cuda:0")

    with torch.inference_mode():
        output_ids = model.generate(
            **generation_kwargs,
        )

    generated_ids = output_ids[0, args.max_prompt_length :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
