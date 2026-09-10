#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
VERL_DIR=${VERL_DIR:-${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0}
OPD_PYTHON=${OPD_PYTHON:-${AGOPD_OPD_VENV}/bin/python}
STUDENT_MODEL=${STUDENT_MODEL:-${PROJECT_DIR}/models/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-${PROJECT_DIR}/models/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/dapo-verl-v1/val.parquet}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${PROJECT_DIR}/tensorboard/vanilla_opd_100step}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/vanilla_opd_100step}
STUDENT_GPUS=${STUDENT_GPUS:-1}
TRAIN_STEPS=${TRAIN_STEPS:-100}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${VERL_DIR}:${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0:${PYTHONPATH:-}"
export PYTHONPATH="${AGOPD_VLLM_VENV}/lib/python3.12/site-packages:${PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR

"${OPD_PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.8 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=True \
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
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.25 \
  distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=16 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name=vanilla_opd_100step \
  trainer.n_gpus_per_node="${STUDENT_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TRAIN_STEPS}" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  "$@"
