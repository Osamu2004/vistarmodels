#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TODSYNTH_ROOT="${TODSYNTH_ROOT:-${ROOT_DIR}/third_party/crfm}"
SD35_MODEL_DIR="${SD35_MODEL_DIR:-/data/vistar/weights/todsynth/sd3.5-medium}"
DATASET_TAG="${DATASET_TAG:-loveda_val}"
DATASET_DIR="${DATASET_DIR:-/data/vistar/runs/todsynth_${DATASET_TAG}_data}"
VECTORS_DIR="${VECTORS_DIR:-/data/vistar/runs/todsynth_vectors/${DATASET_TAG}}"
TODSYNTH_CHECKPOINT="${TODSYNTH_CHECKPOINT:?set trained TODSYNTH_CHECKPOINT folder}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/todsynth_loveda_val_512}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
accelerate launch --num_processes="${NUM_PROCESSES}" "${ROOT_DIR}/baselines/todsynth/run_todsynth_manifest.py" \
  --todsynth_root "${TODSYNTH_ROOT}" --pretrained_model "${SD35_MODEL_DIR}" --checkpoint "${TODSYNTH_CHECKPOINT}" \
  --data_root "${DATASET_DIR}" --json_file "${DATASET_DIR}/index_vectorized.jsonl" --vectors_path "${VECTORS_DIR}" \
  --output_dir "${OUTPUT_DIR}" "${@}"
python "${ROOT_DIR}/tools/merge_ranked_jsonl.py" --directory "${OUTPUT_DIR}"
