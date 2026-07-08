#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

# Reuse the saved condition/GT folders from the Vistar LoveDA generation eval.
# Required layout:
#   ${VISTAR_EVAL_DIR}/cond_mask/*_cond_mask.png
#   ${VISTAR_EVAL_DIR}/gt_rgb/*_gt_rgb.png
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_flux2_loveda_val_mask_to_rgb_gen_resize256_checkpoint1_2gpu}"

EARTHSYNTH_BASE_MODEL="${EARTHSYNTH_BASE_MODEL:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
EARTHSYNTH_CONTROLNET_MODEL="${EARTHSYNTH_CONTROLNET_MODEL:-jaychempan/EarthSynth}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/earthsynth_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_seed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
SCHEDULER="${SCHEDULER:-default}"
DTYPE="${DTYPE:-auto}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
ENABLE_XFORMERS="${ENABLE_XFORMERS:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-0}"

PROMPT="${PROMPT:-A satellite image of buildings, roads, water, barren land, forest, and agriculture, controlled by the semantic segmentation map.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-Low resolution, cropped, worst quality, low quality}"

if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" ]]; then
  echo "[earthsynth_loveda_gen] Missing condition folder: ${VISTAR_EVAL_DIR}/cond_mask" >&2
  echo "[earthsynth_loveda_gen] First run Vistar LoveDA gen eval, or set VISTAR_EVAL_DIR to an eval output with cond_mask and gt_rgb." >&2
  exit 1
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[earthsynth_loveda_gen] Missing GT folder: ${VISTAR_EVAL_DIR}/gt_rgb" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[earthsynth_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[earthsynth_loveda_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[earthsynth_loveda_gen] EARTHSYNTH_BASE_MODEL=${EARTHSYNTH_BASE_MODEL}"
echo "[earthsynth_loveda_gen] EARTHSYNTH_CONTROLNET_MODEL=${EARTHSYNTH_CONTROLNET_MODEL}"
echo "[earthsynth_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[earthsynth_loveda_gen] MANIFEST=${MANIFEST}"
echo "[earthsynth_loveda_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE}"
echo "[earthsynth_loveda_gen] steps=${NUM_INFERENCE_STEPS} cfg=${GUIDANCE_SCALE} control_scale=${CONTROLNET_CONDITIONING_SCALE} scheduler=${SCHEDULER}"
echo "[earthsynth_loveda_gen] dtype=${DTYPE} seed=${SEED} max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[earthsynth_loveda_gen] prompt=${PROMPT}"
echo "[earthsynth_loveda_gen] negative_prompt=${NEGATIVE_PROMPT}"

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
if [[ "${ENABLE_XFORMERS}" == "1" || "${ENABLE_XFORMERS}" == "true" || "${ENABLE_XFORMERS}" == "yes" ]]; then
  EXTRA_ARGS+=(--enable_xformers)
fi
if [[ "${CPU_OFFLOAD}" == "1" || "${CPU_OFFLOAD}" == "true" || "${CPU_OFFLOAD}" == "yes" ]]; then
  EXTRA_ARGS+=(--cpu_offload)
fi
if [[ "${PIPELINE_PROGRESS}" == "1" || "${PIPELINE_PROGRESS}" == "true" || "${PIPELINE_PROGRESS}" == "yes" ]]; then
  EXTRA_ARGS+=(--pipeline_progress)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/earthsynth/run_earthsynth_manifest.py" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model "${EARTHSYNTH_BASE_MODEL}" \
  --controlnet_model "${EARTHSYNTH_CONTROLNET_MODEL}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --controlnet_conditioning_scale "${CONTROLNET_CONDITIONING_SCALE}" \
  --scheduler "${SCHEDULER}" \
  --dtype "${DTYPE}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[earthsynth_loveda_gen] done: ${OUTPUT_DIR}"
