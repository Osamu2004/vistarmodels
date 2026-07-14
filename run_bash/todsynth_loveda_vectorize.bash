#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TODSYNTH_ROOT="${TODSYNTH_ROOT:-${ROOT_DIR}/third_party/crfm}"
SD35_MODEL_DIR="${SD35_MODEL_DIR:-/root/data/weight/todsynth/sd3.5-medium}"
DATASET_TAG="${DATASET_TAG:-loveda_train}"
DATASET_DIR="${DATASET_DIR:-/root/data/experiment/todsynth_${DATASET_TAG}_data}"
VECTORS_ROOT="${VECTORS_ROOT:-/root/data/experiment/todsynth_vectors}"
bash "${ROOT_DIR}/scripts/bootstrap_todsynth.sh"
cd "${TODSYNTH_ROOT}"
"${PYTHON_BIN}" preprocess/vectorize.py --pretrained_model_name_or_path "${SD35_MODEL_DIR}" \
  --data_root "${DATASET_DIR}" --src_json_file index.jsonl --out_json_file index_vectorized.jsonl \
  --work_dir "${VECTORS_ROOT}" --dataset "${DATASET_TAG}"
