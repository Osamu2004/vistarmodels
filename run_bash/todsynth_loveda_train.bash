#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TODSYNTH_ROOT="${TODSYNTH_ROOT:-${ROOT_DIR}/third_party/crfm}"
SD35_MODEL_DIR="${SD35_MODEL_DIR:-/data/vistar/weights/todsynth/sd3.5-medium}"
DATASET_TAG="${DATASET_TAG:-loveda_train}"
DATASET_DIR="${DATASET_DIR:-/data/vistar/runs/todsynth_${DATASET_TAG}_data}"
VECTORS_DIR="${VECTORS_DIR:-/data/vistar/runs/todsynth_vectors/${DATASET_TAG}}"
WORK_DIR="${WORK_DIR:-/data/vistar/weights/todsynth/loveda_train}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
bash "${ROOT_DIR}/scripts/bootstrap_todsynth.sh"
cd "${TODSYNTH_ROOT}"
accelerate launch --num_processes="${NUM_PROCESSES}" train.py \
  --pretrained_model_name_or_path "${SD35_MODEL_DIR}" --data_root "${DATASET_DIR}" \
  --work_dir "${WORK_DIR}" --train_file "${DATASET_DIR}/index_vectorized.jsonl" \
  --vectors_path "${VECTORS_DIR}" --num_cls 7 "${@}"
