#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAE_MODEL="${VAE_MODEL:?set local SD 1.5 Diffusers snapshot}"
CHECKPOINT="${CHECKPOINT:?set trained DiT-B/2 checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/dit_b2_second_test_256}"
CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" "${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" -u \
  "${ROOT_DIR}/baselines/dit_second/run_dit_second.py" --dit_root "${ROOT_DIR}/third_party/DiT" \
  --manifest /data/vistar/runs/paper_baselines/data/second/test.jsonl --vae "${VAE_MODEL}" \
  --checkpoint "${CHECKPOINT}" --output_dir "${OUTPUT_DIR}" "${@}"
