#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${AGOPD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# shellcheck disable=SC1091
source "${PROJECT_DIR}/configs/local-4090.env"
ENV_DIR=${AGOPD_ENV_DIR:-${HOME}/.conda/envs/agopd-vllm-281}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-agopd-vllm-281}
RUNTIME_DIR=${AGOPD_RUNTIME_DIR:-${PROJECT_DIR}/.runtime}
VERL_DIR=${AGOPD_VERL_DIR:-${RUNTIME_DIR}/verl-v0.6.1}
VERL_ARCHIVE=${VERL_ARCHIVE:-${PROJECT_DIR}/verl-v0.6.1.tar.gz}
PYTHON=${ENV_DIR}/bin/python

cd "${PROJECT_DIR}"
unset PIP_CONSTRAINT
export PYTHONNOUSERSITE=1
if [[ "${USE_AGOPD_PROXY:-1}" == "1" && -z "${HTTP_PROXY:-}" && -z "${HTTPS_PROXY:-}" ]]; then
  export HTTP_PROXY="${AGOPD_PROXY_URL}"
  export HTTPS_PROXY="${AGOPD_PROXY_URL}"
  export ALL_PROXY="${AGOPD_PROXY_URL}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
fi
export PIP_INDEX_URL="${PIP_INDEX_URL:-${AGOPD_PIP_INDEX_URL}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_DIR}/.cache/pip}"
export TMPDIR="${TMPDIR:-${PROJECT_DIR}/.tmp}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
mkdir -p "${ENV_DIR%/*}" "${RUNTIME_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda create -n "${CONDA_ENV_NAME}" python=3.12 pip -y
    ENV_DIR=$(conda run -n "${CONDA_ENV_NAME}" python -c 'import sys; print(sys.prefix)')
    PYTHON=${ENV_DIR}/bin/python
  elif command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "${ENV_DIR}"
  else
    python3 -m venv "${ENV_DIR}"
  fi
fi

if [[ ! -f "${VERL_DIR}/pyproject.toml" ]]; then
  [[ -f "${VERL_ARCHIVE}" ]] || {
    echo "Missing ${VERL_ARCHIVE}; copy the project archive before bootstrapping." >&2
    exit 1
  }
  rm -rf "${VERL_DIR}"
  tar -xzf "${VERL_ARCHIVE}" -C "${RUNTIME_DIR}"
fi

"${PYTHON}" -m pip install --upgrade \
  "pip<26" "setuptools<82" wheel packaging ninja cmake

# Use the globally configured PyPI mirror. No -i flag is intentional.
"${PYTHON}" -m pip install \
  --constraint "${PROJECT_DIR}/configs/local-vllm-constraints.txt" \
  -r "${PROJECT_DIR}/configs/local-vllm-requirements.txt"

if [[ "${INSTALL_FLASHINFER:-0}" == "1" ]]; then
  # The mirror may expose only a source archive. Never trigger a source build
  # that can pull a second Torch tree into the isolated build environment.
  "${PYTHON}" -m pip install --only-binary=:all: "flashinfer-python==0.3.0"
else
  echo "Skipping FlashInfer source build; set INSTALL_FLASHINFER=1 only when a wheel is available."
fi

"${PYTHON}" -m pip install --no-deps --no-build-isolation -e "${PROJECT_DIR}"
"${PYTHON}" -m pip install --no-deps --no-build-isolation -e "${VERL_DIR}"

echo
echo "Local environment installed at: ${ENV_DIR}"
echo "verl source: ${VERL_DIR}"
echo "PyPI configuration used by pip:"
"${PYTHON}" -m pip config list || true
echo
echo "Next: bash scripts/check_local_env.sh"
