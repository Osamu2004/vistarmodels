#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
export TORCH_HOME="${TORCH_HOME:-/data/vistar/weights/torch_cache}"
SPADE_ROOT="${SPADE_ROOT:-${ROOT_DIR}/third_party/SPADE}"
DATA_ROOT="${DATA_ROOT:-/data/vistar/runs/paper_baselines/data/second}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/vistar/weights/spade_second}"
RAW_RESULTS="${RAW_RESULTS:-/data/vistar/runs/spade_second_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/spade_second_test_256}"
MAX_SAMPLES="${MAX_SAMPLES:-3388}"
cd "${SPADE_ROOT}"
"${PYTHON_BIN}" test.py --name spade_second_256 --dataset_mode custom \
  --label_dir "${DATA_ROOT}/test/target_mask_ids" --image_dir "${DATA_ROOT}/test/target_rgb" \
  --label_nc 7 --no_instance --gpu_ids "${GPU_IDS:-0}" --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --results_dir "${RAW_RESULTS}" --how_many "${MAX_SAMPLES}" --load_size 256 --crop_size 256 "${@}"
cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/collect_second_semantic_outputs.py --method SPADE \
  --manifest "${DATA_ROOT}/test.jsonl" \
  --pred_dir "${RAW_RESULTS}/spade_second_256/test_latest/images/synthesized_image" \
  --output_dir "${OUTPUT_DIR}" --max_samples "${MAX_SAMPLES}"
