#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"
PYTHON=${AGOPD_ENV_DIR}/bin/python
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${AGOPD_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

[[ -x "${PYTHON}" ]] || {
  echo "Missing ${PYTHON}; run scripts/bootstrap_local_vllm.sh first." >&2
  exit 1
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable in this shell. Run this check inside the local GPU container/host." >&2
  exit 2
fi

echo "-- NVIDIA --"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

echo "-- Python packages --"
PYTHONPATH="${AGOPD_PROJECT_DIR}/src:${AGOPD_VERL_DIR}" "${PYTHON}" \
  "${AGOPD_PROJECT_DIR}/scripts/check_env.py"

GPU_COUNT=$(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "${PYTHON}" -c \
  'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "Expected at least 2 CUDA devices, found ${GPU_COUNT}." >&2
  exit 3
fi
echo "local_gpu_check=ok (${GPU_COUNT} CUDA devices visible)"
