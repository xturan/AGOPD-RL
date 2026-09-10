#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
VERL_DIR=${VERL_DIR:-${PROJECT_DIR}/verl-v0.6.1}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B-sft-openr1-8k}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/dapo-verl-smoke-no-think/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/dapo-verl-smoke-no-think/val.parquet}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/grpo_vllm}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_smoke_vllm}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${PROJECT_DIR}/outputs/grpo_vllm_smoke/rollouts}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
NGPUS=${NGPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
TRAIN_STEPS=${TRAIN_STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:--1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
ROLLOUT_N=${ROLLOUT_N:-2}
MODEL_LORA_RANK=${MODEL_LORA_RANK:-0}
MODEL_LORA_ALPHA=${MODEL_LORA_ALPHA:-16}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${VERL_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TENSORBOARD_DIR

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${PROJECT_DIR}/src/agopd/reward/verl_adapter.py" \
  custom_reward_function.name=compute_score \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank="${MODEL_LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${MODEL_LORA_ALPHA}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.use_legacy_worker_impl=enable \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.total_training_steps="${TRAIN_STEPS}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  "$@"
