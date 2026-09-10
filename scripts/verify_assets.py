from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer


ASSETS = {
    "dataset": Path("data/dapo-math-17k/data/dapo-math-17k.parquet"),
    "student": Path("models/Qwen3-1.7B"),
    "teacher": Path("models/Qwen3-4B"),
}


def main() -> None:
    dataset_path = ASSETS["dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    parquet = pq.ParquetFile(dataset_path)
    print(f"dataset_path={dataset_path}")
    print(f"dataset_rows={parquet.metadata.num_rows}")
    print(f"dataset_columns={parquet.schema_arrow.names}")

    for label in ["student", "teacher"]:
        model_dir = ASSETS[label]
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors shards found in {model_dir}")
        print(f"{label}_path={model_dir}")
        print(f"{label}_model_type={config.model_type}")
        print(f"{label}_hidden_size={getattr(config, 'hidden_size', 'unknown')}")
        print(f"{label}_num_layers={getattr(config, 'num_hidden_layers', 'unknown')}")
        print(f"{label}_tokenizer={tokenizer.__class__.__name__}")
        print(f"{label}_safetensors={[shard.name for shard in shards]}")

    print("verify_assets=ok")


if __name__ == "__main__":
    main()
