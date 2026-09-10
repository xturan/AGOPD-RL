#!/usr/bin/env bash
# run_4b8b_verify.sh — 4B student verify, ARM=A (pure GRPO) / ARM=B (RL+OPD, 8B teacher)
# Channel = validated 4B LoRA recipe (Run C/R88): LoRA r16 + lr1e-4, batch12/mini6/n4, KL0.05, seed42.
# Usage:  ARM=A|B TRAIN_STEPS=N [SAVE_FREQ=..] bash scripts/run_4b8b_verify.sh
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1

ARM=${ARM:?set ARM=A|B}
STEPS=${TRAIN_STEPS:?set TRAIN_STEPS}
BATCH=${TRAIN_BATCH_SIZE:-12}
MINI=${PPO_MINI_BATCH_SIZE:-6}
N=${ROLLOUT_N:-4}
SF=${SAVE_FREQ:-10}

if [ "$ARM" = "A" ]; then
  NAME=scale4b_verify_a_grpo
  DIST_ON=False
  TASK_REW=False
  GATE_ARGS=""
elif [ "$ARM" = "B" ]; then
  NAME=scale4b_verify_b_8bopd
  DIST_ON=True
  TASK_REW=True
  GATE_ARGS=""
elif [ "$ARM" = "C" ]; then
  # Gated AGOPD: teacher scoring AFTER advantage, only below-average trajectories,
  # and only problems the teacher itself solves correctly (competence gate).
  NAME=scale4b_verify_c_agopd
  DIST_ON=True
  TASK_REW=True
  GATE_ARGS="distillation.teacher_after_advantage=True distillation.advantage_gate.enabled=True distillation.teacher_gate.enabled=True"
elif [ "$ARM" = "D" ]; then
  # Gated AGOPD + mode-seeking reverse-KL soft objective + coefficient schedule
  # (SAF-OPD style warmup->anneal), i.e. the 2026 consensus recipe.
  NAME=scale4b_verify_d_agopd_soft
  DIST_ON=True
  TASK_REW=True
  GATE_ARGS="distillation.teacher_after_advantage=True distillation.advantage_gate.enabled=True distillation.teacher_gate.enabled=True distillation.distillation_loss.hard_ce=False distillation.distillation_loss.reverse_kl=True distillation.distillation_loss.temperature=1.0 distillation.distillation_loss.coef_schedule=warmup_anneal distillation.distillation_loss.coef_warmup_steps=20 distillation.distillation_loss.coef_anneal_start=80 distillation.distillation_loss.coef_anneal_end=160 distillation.distillation_loss.coef_floor=0.0"
else
  echo "ARM must be A, B, C or D"; exit 2
fi

echo "[$(date +%H:%M:%S)] START $NAME steps=$STEPS batch=$BATCH mini=$MINI n=$N save=$SF"
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  STUDENT_GPUS=3 TEACHER_N_GPUS=1 TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
  STUDENT_MODEL="${PROJECT_DIR}/models/Qwen3-4B" \
  TEACHER_MODEL="${PROJECT_DIR}/models/Qwen3-8B" \
  TEACHER_GPU_MEMORY_UTILIZATION=0.85 \
  TEACHER_MAX_NUM_BATCHED_TOKENS=16384 \
  TRAIN_FILE="${PROJECT_DIR}/data/dapo-verl-v1/train_comfort.parquet" \
  TRAIN_STEPS="${STEPS}" TRAIN_BATCH_SIZE="${BATCH}" PPO_MINI_BATCH_SIZE="${MINI}" \
  ROLLOUT_N="${N}" SAVE_FREQ="${SF}" \
  STUDENT_LORA_RANK=16 STUDENT_LORA_ALPHA=16 STUDENT_LORA_MERGE=False \
  DISTILLATION_USE_TASK_REWARDS="${TASK_REW}" \
  OUTPUT_DIR="${PROJECT_DIR}/outputs/${NAME}" \
  TENSORBOARD_DIR="${PROJECT_DIR}/tensorboard/${NAME}" \
  EXPERIMENT_NAME="${NAME}" \
  bash scripts/run_grpo_vanilla_opd.sh \
    actor_rollout_ref.rollout.nnodes=0 actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.agent.num_workers=16 \
    +actor_rollout_ref.rollout.enable_sleep_mode=True actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=3073 actor_rollout_ref.actor.use_dynamic_bsz=True \
    data.seed=42 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.model.lora_rank=16 actor_rollout_ref.model.lora_alpha=16 \
    trainer.save_freq="${SF}" \
    distillation.enabled="${DIST_ON}" \
    distillation.distillation_loss.hard_ce=True \
    ${GATE_ARGS:-distillation.teacher_after_advantage=False} \
    > "/tmp/${NAME}.log" 2>&1
echo "[$(date +%H:%M:%S)] DONE $NAME (rc=$?)"
