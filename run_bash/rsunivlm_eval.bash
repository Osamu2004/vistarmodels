#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${RSUNIVLM_DATASET:-both}"
TILE_SIZE="${TILE_SIZE:-512}"
GPU_IDS="${GPU_IDS:-0}"
TILE_BATCH_SIZE="${TILE_BATCH_SIZE:-1}"
ENTRY="${ROOT_DIR}/baselines/rsunivlm/evaluate.py"
EXTRA=()
if [[ "${TILE_BATCH_SIZE}" != "1" || -n "${IMPORT_V4_ROOT:-}" ]]; then
  if [[ "${DATASET}" != "loveda" || "${TILE_SIZE}" != "512" ]]; then
    echo "Batched inference is scoped to LoveDA with 512 source tiles." >&2
    exit 1
  fi
  ENTRY="${ROOT_DIR}/tools/eval_rsunivlm_batched.py"
  EXTRA=(--tile_batch_size "${TILE_BATCH_SIZE}" --verify_batch "${VERIFY_BATCH:-1}")
  if [[ -n "${IMPORT_V4_ROOT:-}" ]]; then EXTRA+=(--import_v4_root "${IMPORT_V4_ROOT}"); fi
fi
if [[ "${NPROC_PER_NODE:-1}" != "1" || "${GPU_IDS}" == *,* ]]; then
  echo "RSUniVLM uses one process/one GPU per output; launch separate dataset jobs for two GPUs." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RSUNIVLM_ROOT="${RSUNIVLM_ROOT:-${ROOT_DIR}/third_party/RSUniVLM}"
export RSUNIVLM_MODEL_PATH="${RSUNIVLM_MODEL_PATH:-/root/data/weight/rsunivlm/llava-qwen-gmoe}"
if [[ "${ENTRY}" == "${ROOT_DIR}/tools/eval_rsunivlm_batched.py" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_loveda_val_native_tile${TILE_SIZE}_rle_v5_bs${TILE_BATCH_SIZE}}"
elif [[ "${DATASET}" == "levircd" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_levircd_native_rle_v3}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_${DATASET}_native_tile${TILE_SIZE}_rle_v4}"
fi

CMD=("${PYTHON_BIN}" "${ENTRY}"
  --dataset "${DATASET}" --source_root "${RSUNIVLM_ROOT}"
  --model_path "${RSUNIVLM_MODEL_PATH}" --output_dir "${OUTPUT_DIR}"
  --loveda_root "${LOVEDA_ROOT:-/root/data/LoveDA}"
  --levircd_root "${LEVIRCD_ROOT:-/root/data/LEVIR-CD}"
  --split "${SPLIT:-test}" --device cuda:0
  --dtype "${DTYPE:-float16}" --attention "${ATTENTION:-eager}"
  --max_new_tokens "${MAX_NEW_TOKENS:-8192}"
  --format_retries "${FORMAT_RETRIES:-2}"
  --max_samples "${MAX_SAMPLES:-0}"
  --loveda_tile_size "${TILE_SIZE}"
  --cd_direction "${CD_DIRECTION:-symmetric_union}")
CMD+=("${EXTRA[@]}")
if [[ -n "${DATA_ROOT:-}" ]]; then CMD+=(--data_root "${DATA_ROOT}"); fi
if [[ -n "${RSUNIVLM_VISION_TOWER:-}" ]]; then CMD+=(--vision_tower "${RSUNIVLM_VISION_TOWER}"); fi
if [[ "${SAVE_IMAGES:-0}" == "0" ]]; then CMD+=(--no-save_images); fi
if [[ "${BOUNDARY_WFM:-1}" == "0" ]]; then CMD+=(--no-boundary_wfm); fi
if [[ "${ALLOW_NONSTANDARD_POPULATION:-0}" == "1" ]]; then CMD+=(--allow_nonstandard_population); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then CMD+=(--overwrite); fi
if [[ "${PREFLIGHT:-0}" == "1" ]]; then CMD+=(--preflight); fi
CMD+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
if [[ "${BOOTSTRAP_RSUNIVLM:-1}" == "1" ]]; then
  bash "${ROOT_DIR}/scripts/bootstrap_rsunivlm.sh"
fi
if [[ "${CHECK_DEPS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_rsunivlm_deps.py"
fi
exec "${CMD[@]}"
