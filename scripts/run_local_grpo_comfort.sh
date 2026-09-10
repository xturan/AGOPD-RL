#!/usr/bin/env bash
# 本地单卡 GRPO(调参探测):comfort 桶数据、n8、lr/rank/offload 参数化
# 用法: CUDA_VISIBLE_DEVICES=0 env LR=3e-4 LORA_RANK=0 OPT_OFFLOAD=True \
#        EXPERIMENT_NAME=local_fp_comfort_n8 TRAIN_STEPS=10 bash scripts/run_local_grpo_comfort.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1

ENV_DIR=${AGOPD_ENV_DIR:-${HOME}/.conda/envs/agopd-vllm-281}
VERL_DIR=${AGOPD_ROOT}/.runtime/verl-v0.8.0
STUDENT_MODEL=models/Qwen3-1.7B
TRAIN_FILE=${TRAIN_FILE:-data/dapo-verl-v1/train_comfort.parquet}
LR=${LR:-3e-4}
LORA_RANK=${LORA_RANK:-16}
OPT_OFFLOAD=${OPT_OFFLOAD:-False}
STEPS=${TRAIN_STEPS:-10}
NAME=${EXPERIMENT_NAME:-local_grpo_comfort_smoke}
BATCH=${TRAIN_BATCH_SIZE:-6}
MINI=${PPO_MINI_BATCH_SIZE:-3}
N=${ROLLOUT_N:-8}

mkdir -p "logs" "outputs/${NAME}" "checkpoints/agopd-rl/${NAME}"

export PYTHONPATH="$(pwd)/src:${VERL_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1  # 单卡无 P2P 需求,防意外

echo "[$(date +%H:%M)] START $NAME lr=$LR rank=$LORA_RANK opt_offload=$OPT_OFFLOAD batch=$BATCH n=$N"

"${ENV_DIR}/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$(pwd)/src/agopd/reward/verl_adapter.py" \
  custom_reward_function.name=compute_score \
  data.train_files="['$(pwd)/${TRAIN_FILE}']" \
  data.val_files="['$(pwd)/data/dapo-verl-v1/val.parquet']" \
  data.train_batch_size="${BATCH}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  data.seed=42 \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.lora.merge=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPT_OFFLOAD}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.nnodes=0 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${N}" \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_model_len=3073 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.ref.use_remove_padding=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  distillation.enabled=False \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name="${NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${STEPS}" \
  trainer.default_local_dir="$(pwd)/checkpoints/agopd-rl/${NAME}" \
  "$@"
echo "[$(date +%H:%M)] DONE $NAME"
