#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHANGEBRIDGE_ROOT="${CHANGEBRIDGE_ROOT:-${ROOT_DIR}/third_party/ChangeBridge}"
DATASET_DIR="${DATASET_DIR:-/root/data/experiment/changebridge_second_dataset}"
RUN_DIR="${RUN_DIR:-/root/data/experiment/changebridge_second_train}"
CONFIG="${CONFIG:-${RUN_DIR}/changebridge_second.yaml}"
CHANGEBRIDGE_VQGAN_CKPT="${CHANGEBRIDGE_VQGAN_CKPT:?set VQGAN checkpoint path}"
CHANGEBRIDGE_CLIP_CKPT="${CHANGEBRIDGE_CLIP_CKPT:?set SkyCLIP checkpoint path}"
GPU_IDS="${GPU_IDS:-0}"
bash "${ROOT_DIR}/scripts/bootstrap_changebridge.sh"
mkdir -p "${RUN_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/configure_changebridge.py" \
  --template "${CHANGEBRIDGE_ROOT}/configs/Template_LBBDM_f4_cd_semantic.yaml" --output "${CONFIG}" \
  --dataset_path "${DATASET_DIR}" --vqgan_ckpt "${CHANGEBRIDGE_VQGAN_CKPT}" --clip_ckpt "${CHANGEBRIDGE_CLIP_CKPT}"
cd "${CHANGEBRIDGE_ROOT}"
"${PYTHON_BIN}" main.py --config "${CONFIG}" --train --gpu_ids "${GPU_IDS}" --result_path "${RUN_DIR}" "${@}"
