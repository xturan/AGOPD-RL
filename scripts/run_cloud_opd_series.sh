#!/usr/bin/env bash
# OPD 系列串行(GRPO 240 完成后):OPD → RL+OPD → AGOPD
# 与 GRPO(full-param lr1e-6 KL0.05 comfort 240)同配置可比
# 3+1 拓扑(FSDP3 + teacher 4B-GRPO 1 卡),每实验独立新名防 resume 污染
set -euo pipefail
PROJECT_DIR=${AGOPD_ROOT}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS=${SERIES_STEPS:-240}

run_method() {
  local name="$1"
  local use_task_rewards="$2"
  local taa="$3"
  local gates="$4"
  echo "[$(date +%H:%M)] START $name steps=$STEPS"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=3 TEACHER_N_GPUS=1 TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
      STUDENT_MODEL=${PROJECT_DIR}/models/Qwen3-1.7B \
      TEACHER_MODEL=${PROJECT_DIR}/models/Qwen3-4B-grpo-50step-ckpt2 \
      TRAIN_FILE=${PROJECT_DIR}/data/dapo-verl-v1/train_comfort.parquet \
      TRAIN_STEPS=${STEPS} TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=6 SAVE_FREQ=10 \
      OUTPUT_DIR=${PROJECT_DIR}/outputs/${name} \
      TENSORBOARD_DIR=${PROJECT_DIR}/tensorboard/${name} \
      EXPERIMENT_NAME="${name}" \
      DISTILLATION_USE_TASK_REWARDS="${use_task_rewards}" \
      bash scripts/run_grpo_vanilla_opd.sh \
      actor_rollout_ref.rollout.nnodes=0 actor_rollout_ref.rollout.n_gpus_per_node=1 \
      actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.agent.num_workers=16 \
      +actor_rollout_ref.rollout.enable_sleep_mode=True actor_rollout_ref.rollout.free_cache_engine=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.max_model_len=3073 actor_rollout_ref.actor.use_dynamic_bsz=True data.seed=42 \
      distillation.distillation_loss.hard_ce=True \
      distillation.teacher_after_advantage="${taa}" \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.kl_loss_coef=0.05 \
      ${gates} \
      > /tmp/${name}.log 2>&1
  echo "[$(date +%H:%M)] DONE $name"
}

# OPD:纯蒸馏(无 task reward)
run_method cloud_17_opd_comfort_240 False False ""

# RL+OPD:GRPO + 蒸馏
run_method cloud_17_rlopd_comfort_240 True False ""

# AGOPD:RL+OPD + advantage gate + teacher gate(taa=True)
run_method cloud_17_agopd_comfort_240 True True \
    "distillation.advantage_gate.enabled=True distillation.teacher_gate.enabled=True"

echo "[$(date +%H:%M)] OPD_SERIES_ALL_DONE"
