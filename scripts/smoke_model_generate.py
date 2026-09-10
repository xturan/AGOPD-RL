from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3 generation without verl, Ray, or FSDP.")
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument(
        "--prompt",
        default="Compute 17 times 23. End your response with a line in the form Answer: number.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model smoke test")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention,
    ).to("cuda:0")
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")

    sampling = (
        {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        if args.enable_thinking
        else {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            **sampling,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
