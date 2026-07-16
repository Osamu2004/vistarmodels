#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[dit_b2_second_train] no Python interpreter found" >&2
    exit 2
  fi
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[dit_b2_second_train] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
DIT_ROOT="${DIT_ROOT:-${ROOT_DIR}/third_party/DiT}"
VAE_MODEL="${VAE_MODEL:?set VAE_MODEL to a local Stable Diffusion 1.5 Diffusers snapshot}"
MANIFEST="${MANIFEST:-/root/data/experiment/dit_b2_second_data/second/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dit_b2_second_source_mask_256_seed42}"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29631}"
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"
RESUME="${RESUME:-auto}"

if [[ "${DIST_BACKEND}" != "gloo" ]]; then
  echo "[dit_b2_second_train] only the Gloo backend is supported, got: ${DIST_BACKEND}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[dit_b2_second_train] dit_root=${DIT_ROOT}"
echo "[dit_b2_second_train] python=${PYTHON_BIN}"
echo "[dit_b2_second_train] manifest=${MANIFEST}"
echo "[dit_b2_second_train] vae=${VAE_MODEL}"
echo "[dit_b2_second_train] output=${OUTPUT_DIR}"
echo "[dit_b2_second_train] gpu_ids=${GPU_IDS} nproc=${NPROC_PER_NODE} per_gpu_batch=${PER_GPU_BATCH} grad_accum=${GRAD_ACCUM}"
echo "[dit_b2_second_train] dist_backend=${DIST_BACKEND}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/check_dit_deps.py" \
  --dit_root "${DIT_ROOT}" --manifest "${MANIFEST}" --vae "${VAE_MODEL}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${ROOT_DIR}/baselines/dit_second/train_dit_second.py" \
  --dit_root "${DIT_ROOT}" \
  --manifest "${MANIFEST}" \
  --vae "${VAE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${PER_GPU_BATCH}" \
  --grad_accum "${GRAD_ACCUM}" \
  --num_workers "${NUM_WORKERS}" \
  --dist_backend "${DIST_BACKEND}" \
  --resume "${RESUME}" \
  "${@}"
