#!/usr/bin/env bash
# Frontier 训练集 GRPO smoke(10 步)+ 正式入口(200 步)
# 数据:train_frontier.parquet(2752 题,1.7B p̂ 0.125-0.75)
# 配置与 E1 完全一致(LoRA r16 lr1e-4, seed42, batch12 mini6 n4, 3+1 拓扑)
# 核心验证:EGR(mixed groups) 实测 vs 理论 61% vs E1 全量 ~20%
set -euo pipefail
PROJECT_DIR=${AGOPD_ROOT}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1

NAME=${EXPERIMENT_NAME:-cloud_17_frontier_grpo_smoke}
STEPS=${TRAIN_STEPS:-10}

echo "[$(date +%H:%M)] START $NAME steps=$STEPS"
env CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=3 TEACHER_N_GPUS=1 TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
    STUDENT_MODEL=${PROJECT_DIR}/models/Qwen3-1.7B \
    TEACHER_MODEL=${PROJECT_DIR}/models/Qwen3-4B-grpo-50step-ckpt2 \
    TRAIN_FILE=${PROJECT_DIR}/data/dapo-verl-v1/train_frontier.parquet \
    TRAIN_STEPS=${STEPS} TRAIN_BATCH_SIZE=12 PPO_MINI_BATCH_SIZE=6 ROLLOUT_N=4 SAVE_FREQ=10 \
    OUTPUT_DIR=${PROJECT_DIR}/outputs/${NAME} \
    EXPERIMENT_NAME="${NAME}" \
    bash scripts/run_grpo_vanilla_opd.sh \
    actor_rollout_ref.rollout.nnodes=0 actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.agent.num_workers=16 \
    +actor_rollout_ref.rollout.enable_sleep_mode=True actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=3073 actor_rollout_ref.actor.use_dynamic_bsz=True data.seed=42 \
    distillation.enabled=False \
    distillation.distillation_loss.hard_ce=True \
    actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.model.lora_rank=16 trainer.save_freq=10 \
    > /tmp/${NAME}.log 2>&1
echo "[$(date +%H:%M)] DONE $NAME"
