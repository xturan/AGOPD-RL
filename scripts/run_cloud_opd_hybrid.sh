#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
METHOD=${HYBRID_METHOD:-e3}
TRAIN_STEPS=${TRAIN_STEPS:-5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-matrix_cloud_${METHOD}_hybrid_2p2_smoke}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${EXPERIMENT_NAME}}

case "${METHOD}" in
  e2) USE_TASK_REWARDS=False ;;
  e3) USE_TASK_REWARDS=True ;;
  *)
    echo "HYBRID_METHOD must be e2 or e3, got: ${METHOD}" >&2
    exit 2
    ;;
esac

# Order logical devices so the two Student/FSDP ranks prefer physical GPUs 1-2,
# which share NUMA node 1. All four A100s remain visible to Ray.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,0,3}
export VERL_RAY_JOB_ID=${VERL_RAY_JOB_ID:-${EXPERIMENT_NAME}_$(date +%s)_$$}

cd "${PROJECT_DIR}"
exec env \
  PROJECT_DIR="${PROJECT_DIR}" \
  VERL_DIR="${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0" \
  OPD_PYTHON=${AGOPD_OPD_VENV}/bin/python \
  STUDENT_MODEL="${PROJECT_DIR}/models/Qwen3-4B" \
  TEACHER_MODEL="${PROJECT_DIR}/models/Qwen3-8B" \
  TRAIN_FILE="${PROJECT_DIR}/data/dapo-verl-v1/train.parquet" \
  VAL_FILE="${PROJECT_DIR}/data/dapo-verl-v1/val.parquet" \
  STUDENT_GPUS=2 \
  TEACHER_N_GPUS=2 \
  TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
  STUDENT_PARAM_OFFLOAD=False \
  STUDENT_OPTIMIZER_OFFLOAD=False \
  STUDENT_LORA_RANK=8 \
  STUDENT_LORA_ALPHA=16 \
  STUDENT_LORA_MERGE=True \
  TRAIN_BATCH_SIZE=8 \
  PPO_MINI_BATCH_SIZE=8 \
  TRAIN_STEPS="${TRAIN_STEPS}" \
  ROLLOUT_N=4 \
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 \
  TEACHER_GPU_MEMORY_UTILIZATION=0.45 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 \
  TEACHER_MAX_NUM_BATCHED_TOKENS=16384 \
  ROLLOUT_ENFORCE_EAGER=True \
  TEACHER_ENFORCE_EAGER=True \
  DISTILLATION_USE_TASK_REWARDS="${USE_TASK_REWARDS}" \
  SAVE_FREQ=-1 \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
  bash "${PROJECT_DIR}/scripts/run_grpo_vanilla_opd.sh" \
  actor_rollout_ref.rollout.nnodes=0 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=16 \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=3073 \
  distillation.teacher_models.teacher_model.inference.data_parallel_size=1 \
  "$@"
