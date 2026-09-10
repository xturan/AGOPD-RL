#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
DATA_FILE=${DATA_FILE:-${PROJECT_DIR}/data/openr1-sft/smoke-8000.jsonl}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft-smoke-qwen3-1.7b}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" scripts/run_sft_smoke.py \
  --data "${DATA_FILE}" \
  --model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}" \
  --max-length 4096 \
  --max-steps 50 "$@"
