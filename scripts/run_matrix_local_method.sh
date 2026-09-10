#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
METHOD=${MATRIX_METHOD:?Set MATRIX_METHOD to e1, e2, or e3}
STEPS=${TRAIN_STEPS:-3}
NAME=${EXPERIMENT_NAME:-matrix_local_${METHOD}_smoke}
OUT=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${NAME}}
TB=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${NAME}}

cd "${PROJECT_DIR}"
if [[ "${METHOD}" == "e1" ]]; then
    exec env \
      AGOPD_MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B}" \
      AGOPD_TRAIN_STEPS="${STEPS}" \
      AGOPD_TRAIN_BATCH_SIZE=8 \
      AGOPD_NGPUS=2 \
      ROLLOUT_N=4 \
      MODEL_LORA_RANK=8 \
      MODEL_LORA_ALPHA=16 \
      MODEL_LORA_MERGE=True \
      SAVE_FREQ=10 \
      DATALOADER_NUM_WORKERS=0 \
      EXPERIMENT_NAME="${NAME}" \
      ROLLOUT_DATA_DIR="${OUT}/rollouts" \
      TENSORBOARD_DIR="${TB}" \
      RUN_LOG="${PROJECT_DIR}/logs/${NAME}.log" \
      bash "${PROJECT_DIR}/scripts/run_local_grpo_vllm.sh"
fi

if [[ "${METHOD}" == "e2" || "${METHOD}" == "e3" ]]; then
    task_rewards=True
    [[ "${METHOD}" == "e2" ]] && task_rewards=False
    exec env \
      MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B}" \
      TEACHER_MODEL="${TEACHER_MODEL:-${PROJECT_DIR}/models/Qwen3-4B}" \
      TRAIN_STEPS="${STEPS}" \
      TRAIN_BATCH_SIZE=8 \
      PPO_MINI_BATCH_SIZE=8 \
      ROLLOUT_N=4 \
      DISTILLATION_USE_TASK_REWARDS="${task_rewards}" \
      EXPERIMENT_NAME="${NAME}" \
      OUTPUT_DIR="${OUT}" \
      TENSORBOARD_DIR="${TB}" \
      RUN_LOG="${PROJECT_DIR}/logs/${NAME}.log" \
      bash "${PROJECT_DIR}/scripts/run_local_opd_matrix.sh"
fi

echo "Unknown MATRIX_METHOD=${METHOD}" >&2
exit 2
