#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
SECOND_SPLITS="${SECOND_SPLITS:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dit_b2_second_data}"

echo "[dit_b2_second_prepare] source=${SECOND_ROOT}"
echo "[dit_b2_second_prepare] splits=${SECOND_SPLITS}"
echo "[dit_b2_second_prepare] output=${OUTPUT_DIR}"
echo "[dit_b2_second_prepare] storage=online (raw RGB/label paths; no image copies)"

SECOND_ROOT="${SECOND_ROOT}" \
SECOND_SPLITS="${SECOND_SPLITS}" \
SECOND_STORAGE=online \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${ROOT_DIR}/run_bash/paper_baselines_prepare_data.bash" --dataset second "${@}"
