#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
PYTHON=${PYTHON:-${AGOPD_VLLM_VENV}/bin/python}
DATA=${DATA:-${PROJECT_DIR}/data/dapo-verl-v1/val.parquet}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/reports/eval-dapo-v1/budget-sweep}
LIMIT=${LIMIT:-256}

mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

for model_name in base opd100; do
  if [[ "${model_name}" == "base" ]]; then
    model="${PROJECT_DIR}/models/Qwen3-1.7B"
  else
    model="${PROJECT_DIR}/models/Qwen3-1.7B-opd100"
  fi

  for budget in 2048 4096 8192; do
    output="${OUT_DIR}/${model_name}-${budget}.jsonl"
    echo "starting model=${model_name} budget=${budget}"
    "${PYTHON}" scripts/evaluate_dapo_vllm.py \
      --model "${model}" \
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
done
