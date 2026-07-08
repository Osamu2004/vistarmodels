#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Reuse the saved condition/GT folders from the Vistar LoveDA generation eval.
# Required layout:
#   ${VISTAR_EVAL_DIR}/cond_mask/*_cond_mask.png
#   ${VISTAR_EVAL_DIR}/gt_rgb/*_gt_rgb.png
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_flux2_loveda_val_mask_to_rgb_gen_resize256_checkpoint1_2gpu}"

CRSDIFF_ROOT="${CRSDIFF_ROOT:-${ROOT_DIR}/third_party/CRS-Diff}"
CRSDIFF_CKPT="${CRSDIFF_CKPT:-/root/data/weight/crsdiff/last.ckpt}"
CRSDIFF_CLIP_VERSION="${CRSDIFF_CLIP_VERSION:-openai/clip-vit-large-patch14}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/crsdiff_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_seed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

BOOTSTRAP_CRSDIFF="${BOOTSTRAP_CRSDIFF:-1}"
CONDITION_SLOT="${CONDITION_SLOT:-seg}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
DDIM_STEPS="${DDIM_STEPS:-50}"
SCALE="${SCALE:-7.5}"
STRENGTH="${STRENGTH:-1.0}"
GLOBAL_STRENGTH="${GLOBAL_STRENGTH:-1.0}"
ETA="${ETA:-0.2}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
PROMPT="${PROMPT:-A high-resolution remote sensing satellite image of an urban and rural scene with buildings, roads, water, barren land, forest, and agriculture, controlled by the semantic segmentation map.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-Low resolution, cropped, worst quality, low quality}"
ADDED_PROMPT="${ADDED_PROMPT:-best quality, extremely detailed}"

if [[ "${BOOTSTRAP_CRSDIFF}" == "1" || "${BOOTSTRAP_CRSDIFF}" == "true" || "${BOOTSTRAP_CRSDIFF}" == "yes" ]]; then
  CRSDIFF_ROOT="${CRSDIFF_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_crsdiff.sh"
fi

if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" ]]; then
  echo "[crsdiff_loveda_gen] Missing condition folder: ${VISTAR_EVAL_DIR}/cond_mask" >&2
  echo "[crsdiff_loveda_gen] First run Vistar LoveDA gen eval, or set VISTAR_EVAL_DIR to an eval output with cond_mask and gt_rgb." >&2
  exit 1
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[crsdiff_loveda_gen] Missing GT folder: ${VISTAR_EVAL_DIR}/gt_rgb" >&2
  exit 1
fi
if [[ ! -f "${CRSDIFF_CKPT}" ]]; then
  echo "[crsdiff_loveda_gen] Missing CRS-Diff checkpoint: ${CRSDIFF_CKPT}" >&2
  echo "[crsdiff_loveda_gen] Download official weights from https://huggingface.co/Sonetto702/AeroGen/tree/main" >&2
  echo "[crsdiff_loveda_gen] Suggested path: /root/data/weight/crsdiff/last.ckpt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[crsdiff_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[crsdiff_loveda_gen] CRSDIFF_ROOT=${CRSDIFF_ROOT}"
echo "[crsdiff_loveda_gen] CRSDIFF_CKPT=${CRSDIFF_CKPT}"
echo "[crsdiff_loveda_gen] CRSDIFF_CLIP_VERSION=${CRSDIFF_CLIP_VERSION}"
echo "[crsdiff_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[crsdiff_loveda_gen] MANIFEST=${MANIFEST}"
echo "[crsdiff_loveda_gen] condition_slot=${CONDITION_SLOT} resolution=${RESOLUTION} eval_size=${EVAL_SIZE}"
echo "[crsdiff_loveda_gen] ddim_steps=${DDIM_STEPS} scale=${SCALE} strength=${STRENGTH} global_strength=${GLOBAL_STRENGTH} eta=${ETA} seed=${SEED}"
echo "[crsdiff_loveda_gen] max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[crsdiff_loveda_gen] prompt=${PROMPT}"
echo "[crsdiff_loveda_gen] added_prompt=${ADDED_PROMPT}"
echo "[crsdiff_loveda_gen] negative_prompt=${NEGATIVE_PROMPT}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_manifest_from_vistar_eval.py" \
  --eval_dir "${VISTAR_EVAL_DIR}" \
  --output "${MANIFEST}" \
  --prompt "${PROMPT}"

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "yes" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/crsdiff/run_crsdiff_manifest.py" \
  --crsdiff_root "${CRSDIFF_ROOT}" \
  --ckpt "${CRSDIFF_CKPT}" \
  --clip_version "${CRSDIFF_CLIP_VERSION}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --condition_slot "${CONDITION_SLOT}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --ddim_steps "${DDIM_STEPS}" \
  --scale "${SCALE}" \
  --strength "${STRENGTH}" \
  --global_strength "${GLOBAL_STRENGTH}" \
  --eta "${ETA}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --added_prompt "${ADDED_PROMPT}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[crsdiff_loveda_gen] done: ${OUTPUT_DIR}"
