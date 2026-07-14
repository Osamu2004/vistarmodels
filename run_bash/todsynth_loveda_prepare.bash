#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:?set VISTAR_EVAL_DIR to the LoveDA eval directory}"
DATASET_TAG="${DATASET_TAG:-loveda_train}"
DATASET_DIR="${DATASET_DIR:-/root/data/experiment/todsynth_${DATASET_TAG}_data}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLITS="${SPLITS:-train}"
ARGS=(); if [[ "${MAX_SAMPLES}" != "0" ]]; then ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_todsynth_loveda.py" --eval_dir "${VISTAR_EVAL_DIR}" --output_dir "${DATASET_DIR}" --splits "${SPLITS}" "${ARGS[@]}"
