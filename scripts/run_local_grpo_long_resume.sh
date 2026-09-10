#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-matrix_local_e1_lora_long_440step}
TOTAL_STEPS=${TOTAL_STEPS:-440}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-${PROJECT_DIR}/checkpoints/agopd-rl/matrix_local_e1_lora_60step/global_step_60}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/agopd-rl/${EXPERIMENT_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${EXPERIMENT_NAME}}
RUN_LOG=${RUN_LOG:-${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log}

[[ -d "${RESUME_CHECKPOINT}" ]] || {
  echo "Resume checkpoint is missing: ${RESUME_CHECKPOINT}" >&2
  exit 1
}

mkdir -p "${CHECKPOINT_DIR}" "${OUTPUT_DIR}/rollouts" "${TENSORBOARD_DIR}" "$(dirname "${RUN_LOG}")"

export AGOPD_PROJECT_DIR="${PROJECT_DIR}"
export AGOPD_TRAIN_STEPS="${TOTAL_STEPS}"
export AGOPD_TRAIN_BATCH_SIZE=8
export AGOPD_NGPUS=2
export MODEL_LORA_RANK=8
export MODEL_LORA_ALPHA=16
export MODEL_LORA_MERGE=True
export ROLLOUT_N=4
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.50
export SAVE_FREQ=10
export DATALOADER_NUM_WORKERS=0
export EXPERIMENT_NAME
export ROLLOUT_DATA_DIR="${OUTPUT_DIR}/rollouts"
export TENSORBOARD_DIR
export RUN_LOG

cd "${PROJECT_DIR}"
exec bash "${PROJECT_DIR}/scripts/run_local_grpo_vllm.sh" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_CHECKPOINT}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}"
