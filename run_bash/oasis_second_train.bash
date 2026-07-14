#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
export TORCH_HOME="${TORCH_HOME:-/data/vistar/weights/torch_cache}"
export OASIS_SKIP_FID="${OASIS_SKIP_FID:-1}"
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
OASIS_ROOT="${OASIS_ROOT:-${ROOT_DIR}/third_party/OASIS}"
DATA_ROOT="${DATA_ROOT:-/data/vistar/runs/paper_baselines/data/second}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/vistar/weights/oasis_second}"
RESUME_TRAINING="${RESUME_TRAINING:-auto}"
RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "auto" ]]; then
  if [[ -f "${CHECKPOINTS_DIR}/oasis_second_256/latest_iter.txt" && \
        -f "${CHECKPOINTS_DIR}/oasis_second_256/models/latest_G.pth" && \
        -f "${CHECKPOINTS_DIR}/oasis_second_256/models/latest_D.pth" ]]; then
    RESUME_ARGS+=(--continue_train --which_iter latest)
  fi
elif [[ "${RESUME_TRAINING}" == "1" || "${RESUME_TRAINING}" == "true" ]]; then
  RESUME_ARGS+=(--continue_train --which_iter latest)
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/configure_oasis_second.py" --oasis_root "${OASIS_ROOT}"
cd "${OASIS_ROOT}"
"${PYTHON_BIN}" train.py --name oasis_second_256 --dataset_mode second --dataroot "${DATA_ROOT}" \
  --gpu_ids 0 --checkpoints_dir "${CHECKPOINTS_DIR}" --batch_size "${BATCH_SIZE:-8}" \
  "${RESUME_ARGS[@]}" "${@}"
