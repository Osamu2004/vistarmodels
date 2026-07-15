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
    echo "[dit_b2_second_gen] no Python interpreter found" >&2
    exit 2
  fi
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[dit_b2_second_gen] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
DIT_ROOT="${DIT_ROOT:-${ROOT_DIR}/third_party/DiT}"
VAE_MODEL="${VAE_MODEL:?set VAE_MODEL to a local Stable Diffusion 1.5 Diffusers snapshot}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a trained DiT-B/2 checkpoint}"
MANIFEST="${MANIFEST:-/root/data/experiment/dit_b2_second_data/second/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dit_b2_second_source_mask_test_256_steps250_seed42}"
GPU_IDS="${GPU_IDS:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[dit_b2_second_gen] python=${PYTHON_BIN}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/baselines/dit_second/run_dit_second.py" \
  --dit_root "${DIT_ROOT}" \
  --manifest "${MANIFEST}" \
  --vae "${VAE_MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  "${@}"
