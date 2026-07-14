#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/data/vistar/runs/seg2any_loveda_val_512_steps32_cfg3p5_seed0}"
LOVEDA_MANIFEST="${LOVEDA_MANIFEST:-}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/data/vistar/weights/syntheticgen}"
LAYOUT_CKPT="${LAYOUT_CKPT:-${WEIGHT_ROOT}/layout/checkpoint-79000}"
CONTROLNET_CKPT="${CONTROLNET_CKPT:-${WEIGHT_ROOT}/controlnet/checkpoint-112000}"
BASE_MODEL="${BASE_MODEL:-$({ find \
  /data/vistar/weights/hf_cache/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots \
  /data/vistar/weights/hf_cache/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots \
  -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1; } || true)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/syntheticgen_loveda_val_512_seed42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

[[ -n "${BASE_MODEL}" && -d "${BASE_MODEL}" ]] || { echo "Missing local Stable Diffusion 1.5 snapshot" >&2; exit 2; }
INPUT_ARGS=(--eval_dir "${VISTAR_EVAL_DIR}")
if [[ -n "${LOVEDA_MANIFEST}" ]]; then INPUT_ARGS=(--manifest "${LOVEDA_MANIFEST}"); fi
"${PYTHON_BIN}" -u "${ROOT_DIR}/baselines/syntheticgen/run_syntheticgen_manifest.py" \
  --syntheticgen_root "${ROOT_DIR}/third_party/SyntheticGen" \
  "${INPUT_ARGS[@]}" --output_dir "${OUTPUT_DIR}" \
  --layout_ckpt "${LAYOUT_CKPT}" --controlnet_ckpt "${CONTROLNET_CKPT}" \
  --base_model "${BASE_MODEL}" --max_samples "${MAX_SAMPLES}" "${@}"
