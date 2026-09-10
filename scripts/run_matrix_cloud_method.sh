#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
METHOD=${MATRIX_METHOD:?Set MATRIX_METHOD to e1, e2, or e3}
STUDENT_MODEL=${STUDENT_MODEL:-${PROJECT_DIR}/models/Qwen3-4B}
TEACHER_MODEL=${TEACHER_MODEL:-${PROJECT_DIR}/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/val.parquet}
STEPS=${TRAIN_STEPS:-3}
NAME=${EXPERIMENT_NAME:-matrix_cloud_${METHOD}_smoke}
OUT=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${NAME}}
TB=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${NAME}}

cd "${PROJECT_DIR}"
export PATH="${AGOPD_VLLM_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VERL_RAY_JOB_ID="${NAME}_$(date +%s)_$$"

if [[ "${METHOD}" == "e1" ]]; then
    exec env \
      PROJECT_DIR="${PROJECT_DIR}" \
      VERL_DIR="${PROJECT_DIR}/verl-v0.6.1" \
      MODEL_PATH="${STUDENT_MODEL}" \
      TRAIN_FILE="${TRAIN_FILE}" \
      VAL_FILE="${VAL_FILE}" \
      NGPUS=4 \
      TRAIN_BATCH_SIZE=8 \
      PPO_MINI_BATCH_SIZE=8 \
      TRAIN_STEPS="${STEPS}" \
      SAVE_FREQ=10 \
      DATALOADER_NUM_WORKERS=0 \
      ROLLOUT_N=4 \
      MODEL_LORA_RANK=8 \
      MODEL_LORA_ALPHA=16 \
      MODEL_LORA_MERGE=True \
      ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
      EXPERIMENT_NAME="${NAME}" \
      ROLLOUT_DATA_DIR="${OUT}/rollouts" \
      TENSORBOARD_DIR="${TB}" \
      bash "${PROJECT_DIR}/scripts/run_grpo_smoke_vllm.sh"
fi

if [[ "${METHOD}" == "e2" || "${METHOD}" == "e3" ]]; then
    task_rewards=True
    [[ "${METHOD}" == "e2" ]] && task_rewards=False
    exec env \
      PROJECT_DIR="${PROJECT_DIR}" \
      VERL_DIR="${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0" \
      OPD_PYTHON=${AGOPD_OPD_VENV}/bin/python \
      STUDENT_MODEL="${STUDENT_MODEL}" \
      TEACHER_MODEL="${TEACHER_MODEL}" \
      TRAIN_FILE="${TRAIN_FILE}" \
      VAL_FILE="${VAL_FILE}" \
      STUDENT_GPUS=1 \
      TEACHER_N_GPUS=1 \
      TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
      STUDENT_PARAM_OFFLOAD=False \
      STUDENT_OPTIMIZER_OFFLOAD=False \
      STUDENT_LORA_RANK=8 \
      STUDENT_LORA_ALPHA=16 \
      STUDENT_LORA_MERGE=True \
      TRAIN_BATCH_SIZE=8 \
      PPO_MINI_BATCH_SIZE=8 \
      TRAIN_STEPS="${STEPS}" \
      ROLLOUT_N=4 \
      MODEL_LORA_RANK=8 \
      MODEL_LORA_ALPHA=16 \
      MODEL_LORA_MERGE=True \
      ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
      TEACHER_GPU_MEMORY_UTILIZATION=0.35 \
      ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 \
      TEACHER_MAX_NUM_BATCHED_TOKENS=16384 \
      ROLLOUT_ENFORCE_EAGER=True \
      TEACHER_ENFORCE_EAGER=True \
      DISTILLATION_USE_TASK_REWARDS="${task_rewards}" \
      SAVE_FREQ=10 \
      EXPERIMENT_NAME="${NAME}" \
      OUTPUT_DIR="${OUT}" \
      TENSORBOARD_DIR="${TB}" \
      bash "${PROJECT_DIR}/scripts/run_grpo_vanilla_opd.sh" \
      actor_rollout_ref.rollout.max_model_len=3073
fi

echo "Unknown MATRIX_METHOD=${METHOD}" >&2
exit 2
