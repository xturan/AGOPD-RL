#!/usr/bin/env bash
# 方向 I:冷启动 SFT 底座(v1-ddp3ep 25.20%)→ Pure GRPO 200 步
# 对照:cloud_17_e1_puregrpo_200step(base 1.7B 底座,E1 已完成)
# 唯一变量:底座模型(base → SFT 冷启动 merged)
# 验证梯度饥饿假说:SFT 底座全错率↓ → reward 应上升(对比 E1 的平)
set -euo pipefail
PROJECT_DIR=${AGOPD_ROOT}
cd "${PROJECT_DIR}"
export PYTHONNOUSERSITE=1

run_one() {
  local log_name="$1"
  local name="$2"
  shift 2
  echo "[$(date +%H:%M)] START $name"
  env CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=3 TEACHER_N_GPUS=1 TEACHER_TENSOR_MODEL_PARALLEL_SIZE=1 \
      STUDENT_MODEL=${PROJECT_DIR}/models/Qwen3-1.7B-sft17v1ddp \
      TEACHER_MODEL=${PROJECT_DIR}/models/Qwen3-4B-grpo-50step-ckpt2 \
      TRAIN_STEPS=200 TRAIN_BATCH_SIZE=12 PPO_MINI_BATCH_SIZE=6 ROLLOUT_N=4 SAVE_FREQ=10 \
      OUTPUT_DIR=${PROJECT_DIR}/outputs/${name} \
      EXPERIMENT_NAME="${name}" \
      bash scripts/run_grpo_vanilla_opd.sh \
      actor_rollout_ref.rollout.nnodes=0 actor_rollout_ref.rollout.n_gpus_per_node=1 \
      actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.agent.num_workers=16 \
      +actor_rollout_ref.rollout.enable_sleep_mode=True actor_rollout_ref.rollout.free_cache_engine=True \
      actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.max_model_len=3073 actor_rollout_ref.actor.use_dynamic_bsz=True data.seed=42 \
      distillation.enabled=False \
      distillation.distillation_loss.hard_ce=True \
      actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.model.lora_rank=16 trainer.save_freq=10 \
      "$@" > /tmp/${log_name}.log 2>&1
  echo "[$(date +%H:%M)] DONE $name"
}

run_one cloud_17_sftbase_grpo_200step cloud_17_sftbase_grpo_200step
echo "[$(date +%H:%M)] SFTBASE_GRPO_DONE"
