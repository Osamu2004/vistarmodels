#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[controlnet_second_prepare] no Python interpreter found" >&2
    exit 2
  fi
fi
SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
DATA_ROOT="${DATA_ROOT:-/root/data/experiment/controlnet_second_data}"
SECOND_SPLITS="${SECOND_SPLITS:-train,test}"
RESOLUTION="${RESOLUTION:-256}"
SECOND_STORAGE="${SECOND_STORAGE:-online}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
echo "[controlnet_second_prepare] python=${PYTHON_BIN}"
echo "[controlnet_second_prepare] second_root=${SECOND_ROOT}"
echo "[controlnet_second_prepare] data_root=${DATA_ROOT}"
echo "[controlnet_second_prepare] splits=${SECOND_SPLITS} storage=${SECOND_STORAGE} resolution=${RESOLUTION}"

COMMAND=("${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_paper_baseline_data.py" \
  --dataset second \
  --second_root "${SECOND_ROOT}" \
  --output "${DATA_ROOT}" \
  --second_splits "${SECOND_SPLITS}" \
  --second_storage "${SECOND_STORAGE}" \
  --second_size "${RESOLUTION}")
if [[ "${REBUILD_MANIFEST}" == "1" ]]; then
  COMMAND+=(--overwrite)
fi
if (( $# > 0 )); then
  COMMAND+=("$@")
fi
"${COMMAND[@]}"
