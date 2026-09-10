#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ENV_DIR=${AGOPD_ENV_DIR:-${HOME}/.conda/envs/agopd-vllm-281}
VERL_DIR=${VERL_DIR:-${PROJECT_DIR}/.runtime/verl-v0.8.0}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-${PROJECT_DIR}/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/val.parquet}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_opd_1p7b_4b}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/${EXPERIMENT_NAME}}
TRAIN_STEPS=${TRAIN_STEPS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
STUDENT_GPU=${STUDENT_GPU:-0}
TEACHER_GPU=${TEACHER_GPU:-1}
DISTILLATION_USE_TASK_REWARDS=${DISTILLATION_USE_TASK_REWARDS:-True}
SAVE_FREQ=${SAVE_FREQ:-10}

[[ -x "${ENV_DIR}/bin/python" ]] || { echo "Missing local environment: ${ENV_DIR}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" && -d "${TEACHER_MODEL}" ]] || { echo "Missing model directory" >&2; exit 1; }
[[ -f "${TRAIN_FILE}" && -f "${VAL_FILE}" ]] || { echo "Missing parquet data" >&2; exit 1; }

cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export RAY_TMPDIR="${PROJECT_DIR}/.runtime/ray"
mkdir -p "${RAY_TMPDIR}"
export PYTHONPATH="${VERL_DIR}:${PROJECT_DIR}/src:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VERL_RAY_JOB_ID="${EXPERIMENT_NAME}_$(date +%s)_$$"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${STUDENT_GPU},${TEACHER_GPU}"
export TENSORBOARD_DIR

RUN_LOG=${RUN_LOG:-${PROJECT_DIR}/logs/${EXPERIMENT_NAME}.log}
mkdir -p "$(dirname "${RUN_LOG}")" "${OUTPUT_DIR}/rollouts" "${TENSORBOARD_DIR}"
exec >>"${RUN_LOG}" 2>&1

exec "${ENV_DIR}/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${PROJECT_DIR}/src/agopd/reward/verl_adapter.py" \
  custom_reward_function.name=compute_score \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.ref.use_remove_padding=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.num_replicas=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.temperature=1.0 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.80 \
  distillation.teacher_models.teacher_model.inference.enforce_eager=True \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=8192 \
  distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=16 \
  distillation.distillation_loss.use_task_rewards="${DISTILLATION_USE_TASK_REWARDS}" \
  distillation.distillation_loss.distillation_loss_coef=1.0 \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TRAIN_STEPS}" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  "$@"
