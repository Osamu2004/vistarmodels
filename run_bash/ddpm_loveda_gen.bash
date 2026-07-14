#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/data/vistar/weights/ddpm_loveda_512}"
EVAL_DIR="${VISTAR_EVAL_DIR:-/data/vistar/runs/seg2any_loveda_val_512_steps32_cfg3p5_seed0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/ddpm_loveda_val_512_seed42}"
CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" "${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" -u \
  "${ROOT_DIR}/baselines/ddpm/run_ddpm_loveda.py" --model "${MODEL}" \
  --eval_dir "${EVAL_DIR}" --output_dir "${OUTPUT_DIR}" "${@}"
