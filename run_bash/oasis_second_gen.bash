#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
export TORCH_HOME="${TORCH_HOME:-/data/vistar/weights/torch_cache}"
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
OASIS_ROOT="${OASIS_ROOT:-${ROOT_DIR}/third_party/OASIS}"
DATA_ROOT="${DATA_ROOT:-/data/vistar/runs/paper_baselines/data/second}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/vistar/weights/oasis_second}"
RAW_RESULTS="${RAW_RESULTS:-/data/vistar/runs/oasis_second_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/oasis_second_test_256}"
MAX_SAMPLES="${MAX_SAMPLES:-3388}"
CKPT_ITER="${CKPT_ITER:-latest}"
export OASIS_MAX_SAMPLES="${MAX_SAMPLES}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/configure_oasis_second.py" --oasis_root "${OASIS_ROOT}"
cd "${OASIS_ROOT}"
"${PYTHON_BIN}" test.py --name oasis_second_256 --dataset_mode second --dataroot "${DATA_ROOT}" \
  --gpu_ids 0 --checkpoints_dir "${CHECKPOINTS_DIR}" --results_dir "${RAW_RESULTS}" --ckpt_iter "${CKPT_ITER}" "${@}"
cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/collect_second_semantic_outputs.py --method OASIS \
  --manifest "${DATA_ROOT}/test.jsonl" --pred_dir "${RAW_RESULTS}/oasis_second_256/${CKPT_ITER}/image" \
  --output_dir "${OUTPUT_DIR}" --max_samples "${MAX_SAMPLES}"
