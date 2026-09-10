#!/usr/bin/env bash
# 续跑版:从 E2 开始(E1 已完成 200 步)
set -euo pipefail
PROJECT_DIR=${AGOPD_ROOT}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1

COMMON_OVERRIDES="distillation.teacher_models.teacher_model.inference.enforce_eager=False distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=16384 distillation.teacher_after_advantage=True distillation.distillation_loss.hard_ce=True actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.model.lora_rank=16 trainer.save_freq=10"

run_one() {
  local name="$1"; shift
  local method="$1"; shift
  local extra_env="${1:-}"; shift 2>/dev/null || true
  echo "[$(date +%H:%M)] START $name ($method)"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=3 TEACHER_N_GPUS=1 TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
      STUDENT_MODEL=${PROJECT_DIR}/models/Qwen3-1.7B \
      TEACHER_MODEL=${PROJECT_DIR}/models/Qwen3-4B-grpo-50step-ckpt2 \
      TRAIN_STEPS=200 TRAIN_BATCH_SIZE=12 PPO_MINI_BATCH_SIZE=6 ROLLOUT_N=4 SAVE_FREQ=10 \
      EXPERIMENT_NAME="${name}" DISTILLATION_COEF=0.3 ${extra_env} \
      bash scripts/run_grpo_vanilla_opd.sh \
      actor_rollout_ref.rollout.nnodes=0 actor_rollout_ref.rollout.n_gpus_per_node=1 \
      actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.agent.num_workers=16 \
      +actor_rollout_ref.rollout.enable_sleep_mode=True actor_rollout_ref.rollout.free_cache_engine=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.max_model_len=3073 actor_rollout_ref.actor.use_dynamic_bsz=True data.seed=42 \
      ${method} ${COMMON_OVERRIDES} "$@" > /tmp/${name}.log 2>&1
  echo "[$(date +%H:%M)] DONE $name"
}

run_one cloud_17_e2_pureopd_200step "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=False" \
    distillation.distillation_loss.hard_ce=False
run_one cloud_17_e3_grpoopd_200step "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=True"
run_one cloud_17_agopd_200step "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=True" \
    distillation.advantage_gate.enabled=True distillation.teacher_gate.enabled=True
echo "[$(date +%H:%M)] ALL_MATRIX_DONE"
