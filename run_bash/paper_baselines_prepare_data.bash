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
    echo "[paper_baselines_prepare_data] no Python interpreter found" >&2
    exit 2
  fi
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[paper_baselines_prepare_data] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
LOVEDA_ROOT="${LOVEDA_ROOT:-/data/vistar/datasets/atomic/loveda/source_data}"
SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
SECOND_SPLITS="${SECOND_SPLITS:-train,test}"
SECOND_STORAGE="${SECOND_STORAGE:-materialized}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/paper_baselines/data}"

echo "[paper_baselines_prepare_data] python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_paper_baseline_data.py" \
  --loveda_root "${LOVEDA_ROOT}" \
  --second_root "${SECOND_ROOT}" \
  --second_splits "${SECOND_SPLITS}" \
  --second_storage "${SECOND_STORAGE}" \
  --output "${OUTPUT_DIR}" "${@}"
