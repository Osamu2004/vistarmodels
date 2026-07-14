#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:?set VISTAR_EVAL_DIR to a Vistar SECOND eval directory}"
DATASET_DIR="${DATASET_DIR:-/root/data/experiment/changebridge_second_dataset}"
SPLIT="${SPLIT:-test}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/vistar_${SPLIT}.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ARGS=(); if [[ "${MAX_SAMPLES}" != "0" ]]; then ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_second_manifest_from_vistar_eval.py" --eval_dir "${VISTAR_EVAL_DIR}" --output "${MANIFEST}" "${ARGS[@]}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_changebridge_dataset.py" --manifest "${MANIFEST}" --output_dir "${DATASET_DIR}" --split "${SPLIT}"
