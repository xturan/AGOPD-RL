#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"

[[ -x "${AGOPD_ENV_DIR}/bin/python" ]] || {
  echo "Missing local environment. Run scripts/bootstrap_local_vllm.sh first." >&2
  exit 1
}

cd "${AGOPD_PROJECT_DIR}"
export PATH="${AGOPD_ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${AGOPD_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PROJECT_DIR="${AGOPD_PROJECT_DIR}"
export DATA_FILE="${DATA_FILE:-${AGOPD_PROJECT_DIR}/data/openr1-sft/smoke-8000.jsonl}"
export MODEL_PATH="${MODEL_PATH:-${AGOPD_MODEL_PATH}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${AGOPD_PROJECT_DIR}/outputs/local-sft-qwen3-1.7b}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${AGOPD_NGPUS}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec bash "${PROJECT_DIR}/scripts/run_sft_smoke.sh" "$@"
