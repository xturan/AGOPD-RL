#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
PYTHON=${PYTHON:-${AGOPD_VLLM_VENV}/bin/python}
DATA=${DATA:-${PROJECT_DIR}/data/dapo-verl-v1/val.parquet}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/reports/eval-dapo-v1/teacher-budget}
LIMIT=${LIMIT:-256}

mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

for budget in 2048 4096 8192; do
  output="${OUT_DIR}/teacher-${budget}.jsonl"
  echo "starting model=teacher budget=${budget}"
  "${PYTHON}" scripts/evaluate_dapo_vllm.py \
    --model "${PROJECT_DIR}/models/Qwen3-4B" \
    --data "${DATA}" \
    --output "${output}" \
    --limit "${LIMIT}" \
    --max-new-tokens "${budget}" \
    --max-model-len "$((budget + 1024))" \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 42 \
    --gpu-memory-utilization 0.5
done
