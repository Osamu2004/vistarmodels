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

INSTANCEDIFFUSION_DIFFUSERS_ROOT="${INSTANCEDIFFUSION_DIFFUSERS_ROOT:-${ROOT_DIR}/third_party/diffusers-instancediffusion}"
INSTANCEDIFFUSION_MODEL="${INSTANCEDIFFUSION_MODEL:-kyeongry/instancediffusion_sd15}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/instancediffusion_loveda_val_mask_to_rgb_gen_resize512_steps50_cfg7p5_seed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

BOOTSTRAP_INSTANCEDIFFUSION="${BOOTSTRAP_INSTANCEDIFFUSION:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
ALPHA="${ALPHA:-0.8}"
BETA="${BETA:-0.36}"
SCHEDULER="${SCHEDULER:-default}"
DTYPE="${DTYPE:-auto}"
MIN_BOX_AREA="${MIN_BOX_AREA:-64}"
MAX_BOXES="${MAX_BOXES:-30}"
BOX_PADDING="${BOX_PADDING:-0}"
INCLUDE_BACKGROUND="${INCLUDE_BACKGROUND:-0}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
ENABLE_XFORMERS="${ENABLE_XFORMERS:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-0}"
PROMPT="${PROMPT:-A high-resolution remote sensing satellite image with buildings, roads, water, barren land, forest, and agriculture.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-longbody, lowres, bad anatomy, cropped, worst quality, low quality}"

_is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if _is_truthy "${BOOTSTRAP_INSTANCEDIFFUSION}"; then
  INSTANCEDIFFUSION_DIFFUSERS_ROOT="${INSTANCEDIFFUSION_DIFFUSERS_ROOT}" \
    bash "${ROOT_DIR}/scripts/bootstrap_instancediffusion.sh"
fi

if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" ]]; then
  echo "[instancediffusion_loveda_gen] Missing condition folder: ${VISTAR_EVAL_DIR}/cond_mask" >&2
  echo "[instancediffusion_loveda_gen] First run Vistar LoveDA gen eval, or set VISTAR_EVAL_DIR to an eval output with cond_mask and gt_rgb." >&2
  exit 1
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[instancediffusion_loveda_gen] Missing GT folder: ${VISTAR_EVAL_DIR}/gt_rgb" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[instancediffusion_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[instancediffusion_loveda_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[instancediffusion_loveda_gen] INSTANCEDIFFUSION_DIFFUSERS_ROOT=${INSTANCEDIFFUSION_DIFFUSERS_ROOT}"
echo "[instancediffusion_loveda_gen] INSTANCEDIFFUSION_MODEL=${INSTANCEDIFFUSION_MODEL}"
echo "[instancediffusion_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[instancediffusion_loveda_gen] MANIFEST=${MANIFEST}"
echo "[instancediffusion_loveda_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE}"
echo "[instancediffusion_loveda_gen] steps=${NUM_INFERENCE_STEPS} cfg=${GUIDANCE_SCALE} alpha=${ALPHA} beta=${BETA}"
echo "[instancediffusion_loveda_gen] scheduler=${SCHEDULER} dtype=${DTYPE} seed=${SEED} max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[instancediffusion_loveda_gen] min_box_area=${MIN_BOX_AREA} max_boxes=${MAX_BOXES} box_padding=${BOX_PADDING} include_background=${INCLUDE_BACKGROUND}"
echo "[instancediffusion_loveda_gen] xformers=${ENABLE_XFORMERS} cpu_offload=${CPU_OFFLOAD} pipeline_progress=${PIPELINE_PROGRESS}"
echo "[instancediffusion_loveda_gen] prompt=${PROMPT}"

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
if _is_truthy "${ENABLE_XFORMERS}"; then
  EXTRA_ARGS+=(--enable_xformers)
fi
if _is_truthy "${CPU_OFFLOAD}"; then
  EXTRA_ARGS+=(--cpu_offload)
fi
if _is_truthy "${PIPELINE_PROGRESS}"; then
  EXTRA_ARGS+=(--pipeline_progress)
fi
if _is_truthy "${INCLUDE_BACKGROUND}"; then
  EXTRA_ARGS+=(--include_background)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/instancediffusion/run_instancediffusion_manifest.py" \
  --diffusers_root "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${INSTANCEDIFFUSION_MODEL}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --scheduler "${SCHEDULER}" \
  --dtype "${DTYPE}" \
  --negative_prompt "${NEGATIVE_PROMPT}" \
  --min_box_area "${MIN_BOX_AREA}" \
  --max_boxes "${MAX_BOXES}" \
  --box_padding "${BOX_PADDING}" \
  --seed "${SEED}" \
  --global_caption "${PROMPT}" \
  "${EXTRA_ARGS[@]}"

echo "[instancediffusion_loveda_gen] done: ${OUTPUT_DIR}"
