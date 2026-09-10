#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${AGOPD_ROOT}}
E1_NAME=${E1_NAME:-matrix_cloud_e1_lora_60step}
E2_NAME=${E2_NAME:-matrix_cloud_e2_lora_60step}
E3_NAME=${E3_NAME:-matrix_cloud_e3_lora_60step}
TRAIN_STEPS=${TRAIN_STEPS:-60}

cd "${PROJECT_DIR}"
echo "sequence_started=$(date '+%F %T %z')"

while pgrep -f "trainer.experiment_name=${E1_NAME}" >/dev/null 2>&1; do
    step=$(grep -a -o 'training/global_step:[0-9]*' "/tmp/${E1_NAME}.log" 2>/dev/null | tail -n 1 | sed 's/.*://' || true)
    echo "watch_e1=$(date '+%F %T') step=${step:-0}/${TRAIN_STEPS}"
    sleep 60
done

if grep -aE 'Error executing|Traceback|OutOfMemory|CUDA error' "/tmp/${E1_NAME}.log" >/dev/null 2>&1; then
    echo "e1_failed=1"
    exit 1
fi

${AGOPD_VLLM_VENV_ALT}/bin/ray stop --force >/tmp/ray_stop_matrix_sequence.log 2>&1 || true

MATRIX_METHOD=e2 TRAIN_STEPS="${TRAIN_STEPS}" EXPERIMENT_NAME="${E2_NAME}" \
  OUTPUT_DIR="${PROJECT_DIR}/outputs/${E2_NAME}" \
  TENSORBOARD_DIR="${PROJECT_DIR}/tensorboard/${E2_NAME}" \
  bash "${PROJECT_DIR}/scripts/run_matrix_cloud_method.sh" \
  >"/tmp/${E2_NAME}.log" 2>&1

${AGOPD_VLLM_VENV_ALT}/bin/ray stop --force >/tmp/ray_stop_matrix_sequence.log 2>&1 || true

MATRIX_METHOD=e3 TRAIN_STEPS="${TRAIN_STEPS}" EXPERIMENT_NAME="${E3_NAME}" \
  OUTPUT_DIR="${PROJECT_DIR}/outputs/${E3_NAME}" \
  TENSORBOARD_DIR="${PROJECT_DIR}/tensorboard/${E3_NAME}" \
  bash "${PROJECT_DIR}/scripts/run_matrix_cloud_method.sh" \
  >"/tmp/${E3_NAME}.log" 2>&1

echo "sequence_completed=$(date '+%F %T %z')"
