#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${RSUNIVLM_DATASET:-both}"
TILE_SIZE="${TILE_SIZE:-512}"
GPU_IDS="${GPU_IDS:-0}"
if [[ "${NPROC_PER_NODE:-1}" != "1" || "${GPU_IDS}" == *,* ]]; then
  echo "RSUniVLM uses one process/one GPU per output; launch separate dataset jobs for two GPUs." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RSUNIVLM_ROOT="${RSUNIVLM_ROOT:-${ROOT_DIR}/third_party/RSUniVLM}"
export RSUNIVLM_MODEL_PATH="${RSUNIVLM_MODEL_PATH:-/root/data/weight/rsunivlm/llava-qwen-gmoe}"
if [[ "${DATASET}" == "levircd" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_levircd_native_rle24_v1}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_${DATASET}_native_tile${TILE_SIZE}_rle24_v2}"
fi

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/baselines/rsunivlm/evaluate.py"
  --dataset "${DATASET}" --source_root "${RSUNIVLM_ROOT}"
  --model_path "${RSUNIVLM_MODEL_PATH}" --output_dir "${OUTPUT_DIR}"
  --loveda_root "${LOVEDA_ROOT:-/root/data/LoveDA}"
  --levircd_root "${LEVIRCD_ROOT:-/root/data/LEVIR-CD}"
  --split "${SPLIT:-test}" --device cuda:0
  --dtype "${DTYPE:-float16}" --attention "${ATTENTION:-eager}"
  --max_new_tokens "${MAX_NEW_TOKENS:-8192}"
  --max_samples "${MAX_SAMPLES:-0}"
  --loveda_tile_size "${TILE_SIZE}"
  --cd_direction "${CD_DIRECTION:-symmetric_union}")
if [[ -n "${DATA_ROOT:-}" ]]; then CMD+=(--data_root "${DATA_ROOT}"); fi
if [[ -n "${RSUNIVLM_VISION_TOWER:-}" ]]; then CMD+=(--vision_tower "${RSUNIVLM_VISION_TOWER}"); fi
if [[ "${SAVE_IMAGES:-1}" == "0" ]]; then CMD+=(--no-save_images); fi
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
