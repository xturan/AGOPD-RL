#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"

[[ -x "${AGOPD_ENV_DIR}/bin/python" ]] || {
  echo "Missing local environment. Run scripts/bootstrap_local_vllm.sh first." >&2
  exit 1
}
[[ -f "${AGOPD_TRAIN_FILE}" && -f "${AGOPD_VAL_FILE}" ]] || {
  echo "Missing local GRPO parquet files. Set AGOPD_TRAIN_FILE/AGOPD_VAL_FILE or prepare the split." >&2
  exit 1
}

cd "${AGOPD_PROJECT_DIR}"
export PATH="${AGOPD_ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${AGOPD_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export RAY_TMPDIR="${AGOPD_RUNTIME_DIR}/ray"
mkdir -p "${RAY_TMPDIR}"
export PROJECT_DIR="${AGOPD_PROJECT_DIR}"
export VERL_DIR="${AGOPD_VERL_DIR}"
export MODEL_PATH="${AGOPD_MODEL_PATH}"
export TRAIN_FILE="${AGOPD_TRAIN_FILE}"
export VAL_FILE="${AGOPD_VAL_FILE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS="${AGOPD_NGPUS}"
export TRAIN_BATCH_SIZE="${AGOPD_TRAIN_BATCH_SIZE}"
export PPO_MINI_BATCH_SIZE="${AGOPD_TRAIN_BATCH_SIZE}"
export TRAIN_STEPS="${AGOPD_TRAIN_STEPS}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${AGOPD_PROJECT_DIR}/tensorboard/local_grpo_vllm}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-local_grpo_1p7b_50step}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${AGOPD_PROJECT_DIR}/outputs/local_grpo_1p7b_50step/rollouts}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
RUN_LOG="${RUN_LOG:-${AGOPD_PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
exec >>"${RUN_LOG}" 2>&1

# Keep the same global effective batch as the cloud run. With two GPUs the
# per-GPU micro batch remains one, so memory pressure is controlled separately.
exec bash "${PROJECT_DIR}/scripts/run_grpo_smoke_vllm.sh" "$@"
