#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/data/LEVIR-CD}"
SPLIT="${SPLIT:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/experiment/figure5_levircd_change_detection}"

DATA_ROOT="${DATA_ROOT}" \
SPLIT="${SPLIT}" \
OUTPUT_DIR="${RCDNET_OUTPUT_DIR:-${OUTPUT_ROOT}/rcdnet_second_real}" \
PYTHON_BIN="${RCDNET_PYTHON_BIN:-${PYTHON_BIN:-python}}" \
  bash "${ROOT_DIR}/run_bash/rcdnet_levircd.bash"

DATA_ROOT="${DATA_ROOT}" \
SPLIT="${SPLIT}" \
OUTPUT_DIR="${DYNAMIC_EARTH_OUTPUT_DIR:-${OUTPUT_ROOT}/dynamicearth_mci}" \
PYTHON_BIN="${DYNAMIC_EARTH_PYTHON_BIN:-${PYTHON_BIN:-python}}" \
  bash "${ROOT_DIR}/run_bash/dynamicearth_levircd.bash"

echo "[levircd_figure5_change_detection] done: ${OUTPUT_ROOT}"
