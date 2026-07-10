#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=0

# SECOND/DreamCD-style dataset root with paired images and paired semantic masks.
# Recognized layouts include:
#   img_A/img_B/mask_A/mask_B/bcd_mask
#   im1/im2/label1/label2/change
SECOND_ROOT="${SECOND_ROOT:-/root/data/SECOND}"
SPLIT="${SPLIT:-test}"
DIRECTION="${DIRECTION:-both}"
ALLOW_MISSING_CHANGE_MASK="${ALLOW_MISSING_CHANGE_MASK:-1}"

DREAMCD_ROOT="${DREAMCD_ROOT:-${ROOT_DIR}/third_party/DreamCD}"
DREAMCD_WEIGHT_ROOT="${DREAMCD_WEIGHT_ROOT:-/root/data/weight/dreamcd}"
DREAMCD_CKPT="${DREAMCD_CKPT:-${DREAMCD_WEIGHT_ROOT}/second/ldm.ckpt}"
DREAMCD_VQVAE_CKPT="${DREAMCD_VQVAE_CKPT:-${DREAMCD_WEIGHT_ROOT}/second/vqvae.ckpt}"
DREAMCD_CONFIG="${DREAMCD_CONFIG:-${DREAMCD_ROOT}/configs/synthesis-wcsdm-second.yaml}"

BOOTSTRAP_DREAMCD="${BOOTSTRAP_DREAMCD:-1}"
RESOLUTION="${RESOLUTION:-256}"
EVAL_SIZE="${EVAL_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DDIM_STEPS="${DDIM_STEPS:-200}"
SEED="${SEED:-2025}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MANIFEST_MAX_SAMPLES="${MANIFEST_MAX_SAMPLES:-${MAX_SAMPLES}}"
OVERWRITE="${OVERWRITE:-0}"
WITH_ADAIN="${WITH_ADAIN:-0}"
NOISE_COND="${NOISE_COND:-1}"
CHANGE_BACKGROUND="${CHANGE_BACKGROUND:-1}"
ONLY_BUILDING="${ONLY_BUILDING:-0}"
WITH_PREVIEW="${WITH_PREVIEW:-0}"
PREVIEW_STEP="${PREVIEW_STEP:-50}"
CONTENT_CORRELATION_SCALE_LOW="${CONTENT_CORRELATION_SCALE_LOW:-0.7}"
SEMANTIC_RGB_MODE="${SEMANTIC_RGB_MODE:-nearest_dreamcd_palette}"
BINARY_CHANGE_MODE="${BINARY_CHANGE_MODE:-auto}"

_is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if _is_truthy "${WITH_ADAIN}"; then
  ADAIN_MODE="adain"
else
  ADAIN_MODE="noadain"
fi
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dreamcd_second_${SPLIT}_${DIRECTION}_${ADAIN_MODE}_vistar_layout_resize256_steps200_seed2025}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}.manifest.jsonl}"
RUNTIME_DIR="${RUNTIME_DIR:-}"

if _is_truthy "${BOOTSTRAP_DREAMCD}"; then
  PYTHON_BIN="${PYTHON_BIN}" \
  DREAMCD_ROOT="${DREAMCD_ROOT}" \
  DREAMCD_WEIGHT_ROOT="${DREAMCD_WEIGHT_ROOT}" \
  DREAMCD_CKPT="${DREAMCD_CKPT}" \
  DREAMCD_VQVAE_CKPT="${DREAMCD_VQVAE_CKPT}" \
    bash "${ROOT_DIR}/scripts/bootstrap_dreamcd.sh"
fi

if [[ ! -d "${SECOND_ROOT}" ]]; then
  echo "[dreamcd_second_gen] Missing SECOND_ROOT: ${SECOND_ROOT}" >&2
  echo "[dreamcd_second_gen] Set SECOND_ROOT to a paired SECOND/DreamCD dataset directory." >&2
  exit 1
fi
if [[ ! -f "${DREAMCD_CKPT}" ]]; then
  echo "[dreamcd_second_gen] Missing DreamCD LDM checkpoint: ${DREAMCD_CKPT}" >&2
  echo "[dreamcd_second_gen] Download weights from https://huggingface.co/tangkaii/DreamCD" >&2
  exit 1
fi
if [[ ! -f "${DREAMCD_VQVAE_CKPT}" ]]; then
  echo "[dreamcd_second_gen] Missing DreamCD VQ-VAE checkpoint: ${DREAMCD_VQVAE_CKPT}" >&2
  echo "[dreamcd_second_gen] Download weights from https://huggingface.co/tangkaii/DreamCD" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[dreamcd_second_gen] SECOND_ROOT=${SECOND_ROOT}"
echo "[dreamcd_second_gen] SPLIT=${SPLIT} DIRECTION=${DIRECTION}"
echo "[dreamcd_second_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[dreamcd_second_gen] DREAMCD_ROOT=${DREAMCD_ROOT}"
echo "[dreamcd_second_gen] DREAMCD_CONFIG=${DREAMCD_CONFIG}"
echo "[dreamcd_second_gen] DREAMCD_CKPT=${DREAMCD_CKPT}"
echo "[dreamcd_second_gen] DREAMCD_VQVAE_CKPT=${DREAMCD_VQVAE_CKPT}"
echo "[dreamcd_second_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[dreamcd_second_gen] MANIFEST=${MANIFEST}"
echo "[dreamcd_second_gen] RUNTIME_DIR=${RUNTIME_DIR:-<temporary>}"
echo "[dreamcd_second_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE}"
echo "[dreamcd_second_gen] ddim_steps=${DDIM_STEPS} seed=${SEED} max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[dreamcd_second_gen] with_adain=${WITH_ADAIN} noise_cond=${NOISE_COND} change_background=${CHANGE_BACKGROUND}"
echo "[dreamcd_second_gen] semantic_rgb_mode=${SEMANTIC_RGB_MODE} binary_change_mode=${BINARY_CHANGE_MODE}"

BUILD_ARGS=()
if _is_truthy "${ALLOW_MISSING_CHANGE_MASK}"; then
  BUILD_ARGS+=(--allow_missing_change_mask)
fi
if [[ "${MANIFEST_MAX_SAMPLES}" != "0" ]]; then
  BUILD_ARGS+=(--max_samples "${MANIFEST_MAX_SAMPLES}")
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_dreamcd_second_manifest.py" \
  --second_root "${SECOND_ROOT}" \
  --split "${SPLIT}" \
  --direction "${DIRECTION}" \
  --output "${MANIFEST}" \
  "${BUILD_ARGS[@]}"

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${RUNTIME_DIR}" ]]; then
  EXTRA_ARGS+=(--runtime_dir "${RUNTIME_DIR}")
fi
if _is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi
if _is_truthy "${WITH_ADAIN}"; then
  EXTRA_ARGS+=(--with_adain)
else
  EXTRA_ARGS+=(--no-with_adain)
fi
if ! _is_truthy "${NOISE_COND}"; then
  EXTRA_ARGS+=(--no-noise_cond)
fi
if ! _is_truthy "${CHANGE_BACKGROUND}"; then
  EXTRA_ARGS+=(--no-change_background)
fi
if _is_truthy "${ONLY_BUILDING}"; then
  EXTRA_ARGS+=(--only_building)
fi
if _is_truthy "${WITH_PREVIEW}"; then
  EXTRA_ARGS+=(--with_preview)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/dreamcd/run_dreamcd_manifest.py" \
  --dreamcd_root "${DREAMCD_ROOT}" \
  --config "${DREAMCD_CONFIG}" \
  --ckpt "${DREAMCD_CKPT}" \
  --vqvae_ckpt "${DREAMCD_VQVAE_CKPT}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --ddim_steps "${DDIM_STEPS}" \
  --seed "${SEED}" \
  --preview_step "${PREVIEW_STEP}" \
  --content_correlation_scale_low "${CONTENT_CORRELATION_SCALE_LOW}" \
  --semantic_rgb_mode "${SEMANTIC_RGB_MODE}" \
  --binary_change_mode "${BINARY_CHANGE_MODE}" \
  "${EXTRA_ARGS[@]}"

echo "[dreamcd_second_gen] done: ${OUTPUT_DIR}"
