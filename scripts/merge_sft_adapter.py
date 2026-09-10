from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Qwen LoRA adapter into a standalone model directory.")
    parser.add_argument("--base-model", default="models/Qwen3-1.7B")
    parser.add_argument("--adapter", default="outputs/sft-smoke-qwen3-1.7b")
    parser.add_argument("--output", default="models/Qwen3-1.7B-sft-smoke")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to merge the SFT adapter")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    missing = [name for name in required_files if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Merged model is incomplete; missing {missing} in {output}")
    print(f"merged_model={output}")
    print(f"merged_files={','.join(required_files)}")


if __name__ == "__main__":
    main()
