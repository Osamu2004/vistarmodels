#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

SECOND_ROOT="${SECOND_ROOT:-/root/data/SECOND}"
SPLIT="${SPLIT:-test}"
DIRECTION="${DIRECTION:-t1_to_t2}"
RCDGEN_MODEL_ID="${RCDGEN_MODEL_ID:-yilmazkorkmaz/RCDGen}"
RCDGEN_WEIGHT_ROOT="${RCDGEN_WEIGHT_ROOT:-/root/data/weight/rcdgen}"
RCDGEN_MODEL_DIR="${RCDGEN_MODEL_DIR:-${RCDGEN_WEIGHT_ROOT}/RCDGen}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rcdgen_second_${SPLIT}_${DIRECTION}_oneclass_random_source_text_resize512_eval256_steps100_seed42}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.jsonl}"
BOOTSTRAP_RCDGEN="${BOOTSTRAP_RCDGEN:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CATEGORY_POLICY="${CATEGORY_POLICY:-random}"
CATEGORY="${CATEGORY:-auto}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "${BOOTSTRAP_RCDGEN}" == "1" ]]; then
  RCDGEN_MODEL_ID="${RCDGEN_MODEL_ID}" RCDGEN_MODEL_DIR="${RCDGEN_MODEL_DIR}" \
    bash "${ROOT_DIR}/scripts/bootstrap_rcdgen.sh"
fi
mkdir -p "${OUTPUT_DIR}"

BUILD_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then BUILD_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_rcdgen_second_manifest.py" \
  --second_root "${SECOND_ROOT}" --split "${SPLIT}" --direction "${DIRECTION}" \
  --output "${MANIFEST}" "${BUILD_ARGS[@]}"

RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then RUN_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${OVERWRITE}" == "1" ]]; then RUN_ARGS+=(--overwrite); fi
"${PYTHON_BIN}" "${ROOT_DIR}/baselines/rcdgen/run_rcdgen_manifest.py" \
  --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}" --model "${RCDGEN_MODEL_DIR}" \
  --resolution "${RESOLUTION}" --eval_size "${EVAL_SIZE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --image_guidance_scale "${IMAGE_GUIDANCE_SCALE}" --guidance_scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" --category_policy "${CATEGORY_POLICY}" --category "${CATEGORY}" "${RUN_ARGS[@]}"

echo "[rcdgen_second_gen] done: ${OUTPUT_DIR}"
