#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
export SEG2ANY_ROOT="${SEG2ANY_ROOT:-${ROOT_DIR}/third_party/Seg2Any}"
export BOOTSTRAP_SEG2ANY="${BOOTSTRAP_SEG2ANY:-0}"
export AUTO_DOWNLOAD_WEIGHTS="${AUTO_DOWNLOAD_WEIGHTS:-0}"

export HF_HOME="${HF_HOME:-/data/vistar/weights/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/vistar/weights/hf_cache/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/vistar/weights/hf_cache/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export SEG2ANY_LORA_LOCAL_DIR="${SEG2ANY_LORA_LOCAL_DIR:-/data/vistar/weights/seg2any}"
export SEG2ANY_LORA_CKPT="${SEG2ANY_LORA_CKPT:-/data/vistar/weights/seg2any/sacap_1m/seg2any/checkpoint-20000}"
export SEG2ANY_FLUX1_LOCAL_DIR="${SEG2ANY_FLUX1_LOCAL_DIR:-/data/vistar/weights/flux1/FLUX.1-dev}"
export SEG2ANY_FLUX1_MODEL="${SEG2ANY_FLUX1_MODEL:-${SEG2ANY_FLUX1_LOCAL_DIR}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/seg2any_loveda_val_mask_to_rgb_gen_resize256_steps32_cfg3p5_seed0}"
export MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

if [[ -z "${VISTAR_EVAL_DIR:-}" ]]; then
  echo "[seg2any_loveda_gen_local] Set VISTAR_EVAL_DIR to an eval directory containing cond_mask/ and gt_rgb/." >&2
  exit 2
fi

exec bash "${ROOT_DIR}/run_bash/seg2any_loveda_gen.bash" "$@"
