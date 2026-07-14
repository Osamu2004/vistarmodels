#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAE_MODEL="${VAE_MODEL:?set local SD 1.5 Diffusers snapshot}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/weights/dit_b2_second_256}"
RESUME="${RESUME:-auto}"
RESUME_ARGS=()
if [[ "${RESUME}" == "auto" ]]; then
  LATEST_CHECKPOINT=""
  if [[ -d "${OUTPUT_DIR}" ]]; then
    LATEST_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'checkpoint-*.pt' -printf '%f\n' | sort -V | tail -n 1)"
  fi
  if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    RESUME_ARGS+=(--resume "${OUTPUT_DIR}/${LATEST_CHECKPOINT}")
  fi
elif [[ -n "${RESUME}" && "${RESUME}" != "none" ]]; then
  RESUME_ARGS+=(--resume "${RESUME}")
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" "${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" -u \
  "${ROOT_DIR}/baselines/dit_second/train_dit_second.py" --dit_root "${ROOT_DIR}/third_party/DiT" \
  --manifest /data/vistar/runs/paper_baselines/data/second/train.jsonl --vae "${VAE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" "${RESUME_ARGS[@]}" "${@}"
