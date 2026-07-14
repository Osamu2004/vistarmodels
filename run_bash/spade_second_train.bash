#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
export TORCH_HOME="${TORCH_HOME:-/data/vistar/weights/torch_cache}"
SPADE_ROOT="${SPADE_ROOT:-${ROOT_DIR}/third_party/SPADE}"
DATA_ROOT="${DATA_ROOT:-/data/vistar/runs/paper_baselines/data/second}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/data/vistar/weights/spade_second}"
GPU_IDS="${GPU_IDS:-0}"
RESUME_TRAINING="${RESUME_TRAINING:-auto}"
RESUME_ARGS=()
if [[ "${RESUME_TRAINING}" == "auto" ]]; then
  if [[ -f "${CHECKPOINTS_DIR}/spade_second_256/latest_net_G.pth" && \
        -f "${CHECKPOINTS_DIR}/spade_second_256/latest_net_D.pth" ]]; then
    RESUME_ARGS+=(--continue_train --which_epoch latest)
  fi
elif [[ "${RESUME_TRAINING}" == "1" || "${RESUME_TRAINING}" == "true" ]]; then
  RESUME_ARGS+=(--continue_train --which_epoch latest)
fi
cd "${SPADE_ROOT}"
"${PYTHON_BIN}" train.py --name spade_second_256 --dataset_mode custom \
  --label_dir "${DATA_ROOT}/train/target_mask_ids" --image_dir "${DATA_ROOT}/train/target_rgb" \
  --label_nc 7 --no_instance --gpu_ids "${GPU_IDS}" --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --batchSize "${BATCH_SIZE:-8}" --load_size 286 --crop_size 256 \
  "${RESUME_ARGS[@]}" "${@}"
