#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
HF_BIN=${HF_BIN:-${PROJECT_DIR}/../.conda/envs/agopd-vllm-281/bin/hf}
if [[ ! -x "${HF_BIN}" ]]; then
  HF_BIN=${HOME}/.conda/envs/agopd-vllm-281/bin/hf
fi

export PYTHONNOUSERSITE=1
export HTTP_PROXY=${HTTP_PROXY:-http://<PROXY_HOST:PORT>}
export HTTPS_PROXY=${HTTPS_PROXY:-http://<PROXY_HOST:PORT>}
export http_proxy=${http_proxy:-${HTTP_PROXY}}
export https_proxy=${https_proxy:-${HTTPS_PROXY}}
export HF_HOME=${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-300}
export TMPDIR=${TMPDIR:-${PROJECT_DIR}/.runtime/tmp}

mkdir -p "${HF_HOME}" "${TMPDIR}" "${PROJECT_DIR}/models"

verify_file() {
  local path=$1
  local expected=$2
  local actual
  actual=$(stat -c '%s' "${path}")
  if [[ "${actual}" != "${expected}" ]]; then
    echo "size_mismatch path=${path} expected=${expected} actual=${actual}" >&2
    return 1
  fi
  echo "size_ok path=${path} bytes=${actual}"
}

download_file() {
  local repo=$1
  local filename=$2
  local target=$3
  local expected=$4
  local actual

  mkdir -p "$(dirname "${target}")"
  while true; do
    actual=0
    [[ -f "${target}" ]] && actual=$(stat -c '%s' "${target}")
    if [[ "${actual}" == "${expected}" ]]; then
      echo "size_ok path=${target} bytes=${actual}"
      return 0
    fi
    echo "http_resume repo=${repo} file=${filename} offset=${actual} time=$(date '+%F %T %z')"
    if "${HF_BIN}" download "${repo}" "${filename}" \
      --local-dir "$(dirname "${target}")" --max-workers 1; then
      actual=$(stat -c '%s' "${target}")
      if [[ "${actual}" == "${expected}" ]]; then
        echo "download_complete file=${target} bytes=${actual}"
        return 0
      fi
      echo "partial_complete file=${target} expected=${expected} actual=${actual}"
    else
      echo "http_retry file=${target}"
    fi
    sleep 10
  done
}

# T1 is the priority model. Each HTTP shard resumes independently, and the
# outer loop keeps retrying after transient CDN/proxy disconnects.
download_file lllyx/Qwen3-4B-Base-GRPO model-00001-of-00002.safetensors \
  "${PROJECT_DIR}/models/Qwen3-4B-GRPO/model-00001-of-00002.safetensors" 4995335936 &
t1_shard1_pid=$!
download_file lllyx/Qwen3-4B-Base-GRPO model-00002-of-00002.safetensors \
  "${PROJECT_DIR}/models/Qwen3-4B-GRPO/model-00002-of-00002.safetensors" 3827558496 &
t1_shard2_pid=$!
wait "${t1_shard1_pid}" "${t1_shard2_pid}"

download_file nvidia/OpenMath-Nemotron-1.5B model.safetensors \
  "${PROJECT_DIR}/models/OpenMath-Nemotron-1.5B/model.safetensors" 3087467144

download_file nvidia/OpenMath-Nemotron-7B model-00001-of-00002.safetensors \
  "${PROJECT_DIR}/models/OpenMath-Nemotron-7B/model-00001-of-00002.safetensors" 9979257128 &
t3_shard1_pid=$!
download_file nvidia/OpenMath-Nemotron-7B model-00002-of-00002.safetensors \
  "${PROJECT_DIR}/models/OpenMath-Nemotron-7B/model-00002-of-00002.safetensors" 5252014776 &
t3_shard2_pid=$!
wait "${t3_shard1_pid}" "${t3_shard2_pid}"

# Sync the small metadata files afterwards to complete each repository snapshot.
"${HF_BIN}" download lllyx/Qwen3-4B-Base-GRPO --local-dir "${PROJECT_DIR}/models/Qwen3-4B-GRPO" --exclude '*.safetensors' --max-workers 4 || true
"${HF_BIN}" download nvidia/OpenMath-Nemotron-1.5B --local-dir "${PROJECT_DIR}/models/OpenMath-Nemotron-1.5B" --exclude '*.safetensors' --max-workers 4 || true
"${HF_BIN}" download nvidia/OpenMath-Nemotron-7B --local-dir "${PROJECT_DIR}/models/OpenMath-Nemotron-7B" --exclude '*.safetensors' --max-workers 4 || true

echo "teacher_candidates_download=complete time=$(date '+%F %T %z')"
