#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"

MODEL=${MODEL:-${AGOPD_MODEL_PATH}}
DATA=${DATA:-${AGOPD_VAL_FILE}}
OUTPUT=${OUTPUT:-${AGOPD_PROJECT_DIR}/reports/local-eval.jsonl}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

[[ -x "${AGOPD_ENV_DIR}/bin/python" ]] || {
  echo "Missing local environment. Run scripts/bootstrap_local_vllm.sh first." >&2
  exit 1
}
[[ -d "${MODEL}" && -f "${DATA}" ]] || {
  echo "Model or evaluation parquet is missing: ${MODEL} / ${DATA}" >&2
  exit 1
}

cd "${AGOPD_PROJECT_DIR}"
export PYTHONPATH="${AGOPD_PROJECT_DIR}/src:${PYTHONPATH:-}"
export PATH="${AGOPD_ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${AGOPD_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${AGOPD_ENV_DIR}/bin/python" scripts/evaluate_dapo_vllm.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "$@"
