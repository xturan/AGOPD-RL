#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
TRAIN_LOG=${TRAIN_LOG:-/tmp/scale_grpo_4b_50step_ckpt.log}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints/agopd-rl/scale_grpo_4b_50step_ckpt/global_step_50}
MERGED_MODEL=${MERGED_MODEL:-${PROJECT_DIR}/models/Qwen3-4B-grpo-50step-ckpt}
EVAL_OUTPUT=${EVAL_OUTPUT:-${PROJECT_DIR}/reports/eval-dapo-v1/grpo-4b-50step-ckpt.jsonl}
WATCH_LOG=${WATCH_LOG:-/tmp/scale_grpo_4b_50step_ckpt_watch.log}

cd "${PROJECT_DIR}"
exec >>"${WATCH_LOG}" 2>&1
echo "watch_started=$(date '+%F %T %z')"

last_step=0
while true; do
    if [[ -f "${TRAIN_LOG}" ]]; then
        current_step=$(grep -a -o 'training/global_step:[0-9]*' "${TRAIN_LOG}" | tail -n 1 | sed 's/.*://' || true)
        current_step=${current_step:-0}
        if [[ "${current_step}" != "${last_step}" ]]; then
            echo "watch=$(date '+%F %T') step=${current_step}/50"
            last_step=${current_step}
        fi
    fi
    if [[ -f "${CHECKPOINT_DIR}/actor/model_world_size_4_rank_0.pt" ]]; then
        echo "checkpoint_ready=$(date '+%F %T %z')"
        break
    fi
    sleep 60
done

if [[ ! -f "${MERGED_MODEL}/model.safetensors" ]]; then
    mkdir -p "${MERGED_MODEL}"
    PYTHONPATH="${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0:${PROJECT_DIR}/src" \
      ${AGOPD_VLLM_VENV}/bin/python \
      "${PROJECT_DIR}/verl-v0.8.0-src/verl-v0.8.0/scripts/legacy_model_merger.py" merge \
      --backend fsdp \
      --local_dir "${CHECKPOINT_DIR}/actor" \
      --target_dir "${MERGED_MODEL}"
fi

if [[ ! -f "${EVAL_OUTPUT%.jsonl}.summary.json" ]]; then
    PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/verl-v0.6.1" \
      CUDA_VISIBLE_DEVICES=0 \
      ${AGOPD_VLLM_VENV}/bin/python \
      "${PROJECT_DIR}/scripts/evaluate_dapo_vllm.py" \
      --model "${MERGED_MODEL}" \
      --data "${PROJECT_DIR}/data/dapo-verl-v1/val.parquet" \
      --output "${EVAL_OUTPUT}" \
      --max-new-tokens 2048 \
      --temperature 0.6 \
      --top-p 0.95 \
      --top-k 20 \
      --seed 42 \
      --gpu-memory-utilization 0.5
fi

echo "watch_pipeline_complete=$(date '+%F %T %z')"
