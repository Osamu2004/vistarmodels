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

SEG2ANY_ROOT="${SEG2ANY_ROOT:-${ROOT_DIR}/third_party/Seg2Any}"
SEG2ANY_FLUX1_MODEL="${SEG2ANY_FLUX1_MODEL:-black-forest-labs/FLUX.1-dev}"
SEG2ANY_LORA_CKPT="${SEG2ANY_LORA_CKPT:-/root/data/weight/seg2any/sacap_1m/seg2any/checkpoint-20000}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/seg2any_loveda_val_mask_to_rgb_gen_resize512_steps32_cfg3p5_seed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_val.jsonl}"

BOOTSTRAP_SEG2ANY="${BOOTSTRAP_SEG2ANY:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
COND_SCALE_FACTOR="${COND_SCALE_FACTOR:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-32}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
COND2IMAGE_ATTENTION_WEIGHT="${COND2IMAGE_ATTENTION_WEIGHT:-1.0}"
ATTENTION_MASK_METHOD="${ATTENTION_MASK_METHOD:-hard}"
HARD_ATTN_BLOCK_START="${HARD_ATTN_BLOCK_START:-19}"
HARD_ATTN_BLOCK_END="${HARD_ATTN_BLOCK_END:-37}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-0}"
PROMPT="${PROMPT:-A high-resolution remote sensing satellite image with buildings, roads, water, barren land, forest, and agriculture.}"

if [[ "${BOOTSTRAP_SEG2ANY}" == "1" || "${BOOTSTRAP_SEG2ANY}" == "true" || "${BOOTSTRAP_SEG2ANY}" == "yes" ]]; then
  SEG2ANY_ROOT="${SEG2ANY_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_seg2any.sh"
fi

if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" ]]; then
  echo "[seg2any_loveda_gen] Missing condition folder: ${VISTAR_EVAL_DIR}/cond_mask" >&2
  echo "[seg2any_loveda_gen] First run Vistar LoveDA gen eval, or set VISTAR_EVAL_DIR to an eval output with cond_mask and gt_rgb." >&2
  exit 1
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[seg2any_loveda_gen] Missing GT folder: ${VISTAR_EVAL_DIR}/gt_rgb" >&2
  exit 1
fi
if [[ ! -d "${SEG2ANY_LORA_CKPT}" ]]; then
  echo "[seg2any_loveda_gen] Missing Seg2Any LoRA checkpoint: ${SEG2ANY_LORA_CKPT}" >&2
  echo "[seg2any_loveda_gen] Suggested download:" >&2
  echo "  huggingface-cli download 0xLDF/Seg2Any --local-dir /root/data/weight/seg2any" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[seg2any_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[seg2any_loveda_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[seg2any_loveda_gen] SEG2ANY_ROOT=${SEG2ANY_ROOT}"
echo "[seg2any_loveda_gen] SEG2ANY_FLUX1_MODEL=${SEG2ANY_FLUX1_MODEL}"
echo "[seg2any_loveda_gen] SEG2ANY_LORA_CKPT=${SEG2ANY_LORA_CKPT}"
echo "[seg2any_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[seg2any_loveda_gen] MANIFEST=${MANIFEST}"
echo "[seg2any_loveda_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} cond_scale_factor=${COND_SCALE_FACTOR}"
echo "[seg2any_loveda_gen] steps=${NUM_INFERENCE_STEPS} cfg=${GUIDANCE_SCALE} cond2image_attention_weight=${COND2IMAGE_ATTENTION_WEIGHT}"
echo "[seg2any_loveda_gen] attention_mask_method=${ATTENTION_MASK_METHOD} hard_attn_block_range=${HARD_ATTN_BLOCK_START},${HARD_ATTN_BLOCK_END}"
echo "[seg2any_loveda_gen] dtype=${DTYPE} seed=${SEED} max_samples=${MAX_SAMPLES} overwrite=${OVERWRITE}"
echo "[seg2any_loveda_gen] prompt=${PROMPT}"

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
if [[ "${PIPELINE_PROGRESS}" == "1" || "${PIPELINE_PROGRESS}" == "true" || "${PIPELINE_PROGRESS}" == "yes" ]]; then
  EXTRA_ARGS+=(--pipeline_progress)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/seg2any/run_seg2any_manifest.py" \
  --seg2any_root "${SEG2ANY_ROOT}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained_model_name_or_path "${SEG2ANY_FLUX1_MODEL}" \
  --lora_ckpt_path "${SEG2ANY_LORA_CKPT}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --cond_scale_factor "${COND_SCALE_FACTOR}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --cond2image_attention_weight "${COND2IMAGE_ATTENTION_WEIGHT}" \
  --attention_mask_method "${ATTENTION_MASK_METHOD}" \
  --hard_attn_block_start "${HARD_ATTN_BLOCK_START}" \
  --hard_attn_block_end "${HARD_ATTN_BLOCK_END}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --global_caption "${PROMPT}" \
  "${EXTRA_ARGS[@]}"

echo "[seg2any_loveda_gen] done: ${OUTPUT_DIR}"
