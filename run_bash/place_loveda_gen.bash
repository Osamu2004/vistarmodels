#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TQDM_DISABLE=0

# Reuse the saved condition/GT folders from the Vistar LoveDA generation eval.
# Required layout:
#   ${VISTAR_EVAL_DIR}/cond_mask/*_cond_mask.png
#   ${VISTAR_EVAL_DIR}/gt_rgb/*_gt_rgb.png
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_flux2_loveda_val_mask_to_rgb_gen_resize256_checkpoint1_2gpu}"

PLACE_ROOT="${PLACE_ROOT:-${ROOT_DIR}/third_party/PLACE}"
PLACE_CKPT="${PLACE_CKPT:-/root/data/weight/place/coco_best.ckpt}"
PLACE_CONFIG="${PLACE_CONFIG:-${PLACE_ROOT}/configs/stable-diffusion/PLACE.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/place_loveda_val_mask_to_rgb_gen_resize512_steps50_cfg2_seed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

BOOTSTRAP_PLACE="${BOOTSTRAP_PLACE:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.0}"
INCLUDE_BACKGROUND="${INCLUDE_BACKGROUND:-0}"
INCLUDE_UNKNOWN="${INCLUDE_UNKNOWN:-0}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
PROMPT="${PROMPT:-A high-resolution remote sensing satellite image with buildings, roads, water, barren land, forest, and agriculture.}"

_is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if _is_truthy "${BOOTSTRAP_PLACE}"; then
  PLACE_ROOT="${PLACE_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_place.sh"
fi

if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" ]]; then
  echo "[place_loveda_gen] Missing condition folder: ${VISTAR_EVAL_DIR}/cond_mask" >&2
  echo "[place_loveda_gen] First run Vistar LoveDA gen eval, or set VISTAR_EVAL_DIR to an eval output with cond_mask and gt_rgb." >&2
  exit 1
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[place_loveda_gen] Missing GT folder: ${VISTAR_EVAL_DIR}/gt_rgb" >&2
  exit 1
fi
if [[ ! -f "${PLACE_CKPT}" ]]; then
  echo "[place_loveda_gen] Missing PLACE checkpoint: ${PLACE_CKPT}" >&2
  echo "[place_loveda_gen] Download coco_best.ckpt or ade20k_best.ckpt from the official Google Drive linked in baselines/place/README.md." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[place_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[place_loveda_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[place_loveda_gen] PLACE_ROOT=${PLACE_ROOT}"
echo "[place_loveda_gen] PLACE_CONFIG=${PLACE_CONFIG}"
echo "[place_loveda_gen] PLACE_CKPT=${PLACE_CKPT}"
echo "[place_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[place_loveda_gen] MANIFEST=${MANIFEST}"
echo "[place_loveda_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE}"
echo "[place_loveda_gen] steps=${STEPS} cfg=${GUIDANCE_SCALE} seed=${SEED} max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[place_loveda_gen] include_background=${INCLUDE_BACKGROUND} include_unknown=${INCLUDE_UNKNOWN}"
echo "[place_loveda_gen] manifest_prompt=${PROMPT}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_manifest_from_vistar_eval.py" \
  --eval_dir "${VISTAR_EVAL_DIR}" \
  --output "${MANIFEST}" \
  --prompt "${PROMPT}"

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if _is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi
if _is_truthy "${INCLUDE_BACKGROUND}"; then
  EXTRA_ARGS+=(--include_background)
fi
if _is_truthy "${INCLUDE_UNKNOWN}"; then
  EXTRA_ARGS+=(--include_unknown)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/place/run_place_manifest.py" \
  --place_root "${PLACE_ROOT}" \
  --config "${PLACE_CONFIG}" \
  --ckpt "${PLACE_CKPT}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[place_loveda_gen] done: ${OUTPUT_DIR}"
