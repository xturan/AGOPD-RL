#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
VERL_DIR=${VERL_DIR:-${PROJECT_DIR}/verl}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-1.7B}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/dapo-verl-smoke/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/dapo-verl-smoke/val.parquet}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${PROJECT_DIR}/outputs/grpo_smoke_hf/rollouts}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-2}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${VERL_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export HYDRA_FULL_ERROR=1

if [[ ! -f "${VERL_DIR}/verl/workers/fsdp_workers.py" ]]; then
  cat >&2 <<EOF
This HF smoke test requires verl v0.5.0's legacy FSDP worker.
The installed verl checkout does not provide it. Run:

  cd ${VERL_DIR}
  git fetch --tags
  git switch --detach v0.5.0
  python3 -m pip install --no-deps --no-build-isolation -e . --break-system-packages
EOF
  exit 2
fi

if grep -q '"num_return_sequences": self.config.n' \
  "${VERL_DIR}/verl/workers/rollout/hf_rollout.py"; then
  cat >&2 <<EOF
verl v0.5.0 has an HF rollout double-repeat bug when rollout.n > 1.
Apply the project compatibility patch before training:

  cd ${VERL_DIR}
  patch -p1 < ${PROJECT_DIR}/patches/verl-v0.5.0-hf-rollout-repeat.patch
EOF
  exit 2
fi

if grep -q 'position_ids=position_ids,' \
  "${VERL_DIR}/verl/workers/rollout/hf_rollout.py"; then
  cat >&2 <<EOF
verl v0.5.0 passes left-padded position_ids into Qwen3 generate(), which causes
repetitive rollouts. Apply the project compatibility patch before training:

  cd ${VERL_DIR}
  patch -p1 < ${PROJECT_DIR}/patches/verl-v0.5.0-hf-rollout-position-ids.patch
EOF
  exit 2
fi

if grep -q '^[[:space:]]*position_ids=position_ids,' \
  "${VERL_DIR}/verl/workers/rollout/hf_rollout.py"; then
  cat >&2 <<EOF
verl v0.5.0 passes left-padded position_ids to Qwen3 generation, which corrupts rollouts.
Apply the project compatibility patch before training:

  cd ${VERL_DIR}
  patch -p1 < ${PROJECT_DIR}/patches/verl-v0.5.0-hf-rollout-position-ids.patch
EOF
  exit 2
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${PROJECT_DIR}/src/agopd/reward/verl_adapter.py" \
  custom_reward_function.name=compute_score \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
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
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger='["console"]' \
  trainer.project_name=agopd-rl \
  trainer.experiment_name=grpo_smoke_hf \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.use_legacy_worker_impl=enable \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 "$@"
