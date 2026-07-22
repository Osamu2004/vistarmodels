#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVALUATION_DIR="${1:?Usage: bash run_bash/compute_saved_rskt_chn6_wfm.bash EVALUATION_DIR}"
DATA_ROOT="${DATA_ROOT:-/root/data/CHN6-CUG/val}"
WORKERS="${WORKERS:-8}"
EXPECTED_NUM_SAMPLES="${EXPECTED_NUM_SAMPLES:-903}"

exec "${PYTHON_BIN}" \
  "${ROOT_DIR}/tools/compute_saved_rskt_chn6_wfm.py" \
  "${EVALUATION_DIR}" \
  --data_root "${DATA_ROOT}" \
  --workers "${WORKERS}" \
  --expected_num_samples "${EXPECTED_NUM_SAMPLES}"
