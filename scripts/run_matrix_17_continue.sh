#!/usr/bin/env bash
# 矩阵续跑 v3:E2 resume(60→200)+ E3(GRPO+OPD)+ AGOPD,每臂 200 步总预算
# 相对旧脚本修复:
#  1. run_one shift-2 吞参 bug → shift 4,extra 参数全部保留
#  2. teacher_after_advantage 移出 COMMON,每臂显式(无 gate 臂必须 False)
#  3. E2 从 global_step_60 resume(trainer.resume_mode=resume_path)
#  4. PYTORCH_CUDA_ALLOC_CONF=expandable_segments 防 step-61 式显存峰值 OOM
set -euo pipefail
PROJECT_DIR=${AGOPD_ROOT}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_ROOT="${PROJECT_DIR}/checkpoints/agopd-rl"

COMMON_OVERRIDES="distillation.teacher_models.teacher_model.inference.enforce_eager=False distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=16384 distillation.distillation_loss.hard_ce=True actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.model.lora_rank=16 trainer.save_freq=10"

run_one() {
  local log_name="$1"
  local name="$2"
  local method="$3"
  local extra_env="${4:-}"
  shift 4
  echo "[$(date +%H:%M)] START $name ($method) extra: $*"
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
      "${method}" ${COMMON_OVERRIDES} "$@" > /tmp/${log_name}.log 2>&1
  echo "[$(date +%H:%M)] DONE $name"
}

# E2 备份崩溃日志后 resume(step 60 → 200)
cp -n /tmp/cloud_17_e2_pureopd_200step.log /tmp/cloud_17_e2_pureopd_200step_crash.log 2>/dev/null || true
run_one cloud_17_e2_pureopd_200step_r cloud_17_e2_pureopd_200step \
    "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=False" \
    distillation.teacher_after_advantage=False \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT_ROOT}/cloud_17_e2_pureopd_200step/global_step_60"

# E3: GRPO + OPD(任务奖励 + 蒸馏)
run_one cloud_17_e3_grpoopd_200step cloud_17_e3_grpoopd_200step \
    "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=True" \
    distillation.teacher_after_advantage=False

# AGOPD: E3 + advantage gate + teacher gate(taa=True 是 AGOPD 的语义核心)
run_one cloud_17_agopd_200step cloud_17_agopd_200step \
    "distillation.enabled=True" "DISTILLATION_USE_TASK_REWARDS=True" \
    distillation.teacher_after_advantage=True \
    distillation.advantage_gate.enabled=True \
    distillation.teacher_gate.enabled=True

echo "[$(date +%H:%M)] ALL_MATRIX_DONE"
