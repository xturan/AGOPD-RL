#!/usr/bin/env bash
# 本地 1.7B:冷启动 SFT 底座(merged)+ Pure GRPO — 与云端 E1 严格可比
# 对照:云端 cloud_17_e1_puregrpo_200step(base 1.7B + GRPO)
# 同:seed42 / batch12 / mini6 / n4 / LoRA r16 lr1e-4 / max_resp 2048 / DAPO v1 reward / verl v0.8
# 异:底座 = Qwen3-1.7B-sft17(SFT 冷启动 ckpt-1100 merged),机器 = 本地 2×4090(FSDP2 + 2 replica)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1

ENV_DIR=${AGOPD_ENV_DIR:-${HOME}/.conda/envs/agopd-vllm-281}
VERL_DIR=${AGOPD_ROOT}/.runtime/verl-v0.8.0
STUDENT_MODEL=${STUDENT_MODEL:-models/Qwen3-1.7B-sft17}
TRAIN_STEPS=${TRAIN_STEPS:-200}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_17_sftcold_grpo_200step}

mkdir -p "logs"

export PYTHONPATH="$(pwd)/src:${VERL_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
# 本地 2×4090 之间 NCCL P2P 不可用(cu128/NCCL 2.27.3 报 peer access
# 217);广播/同步走 host 内存,1.7B 规模影响可忽略
export NCCL_P2P_DISABLE=1

"${ENV_DIR}/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$(pwd)/src/agopd/reward/verl_adapter.py" \
  custom_reward_function.name=compute_score \
  data.train_files="['$(pwd)/data/dapo-verl-v1/train.parquet']" \
  data.val_files="['$(pwd)/data/dapo-verl-v1/val.parquet']" \
  data.train_batch_size=12 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  data.seed=42 \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.lora.merge=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=6 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.nnodes=0 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.agent.num_workers=16 \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_model_len=3073 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.ref.use_remove_padding=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  distillation.enabled=False \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TRAIN_STEPS}" \
  "$@"
