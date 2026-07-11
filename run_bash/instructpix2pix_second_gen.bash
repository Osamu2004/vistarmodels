#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
SPLIT="${SPLIT:-test}"
DIRECTION="${DIRECTION:-both}"
SELECTION_SEED="${SELECTION_SEED:-42}"
LABEL_PAIR_MODE="${LABEL_PAIR_MODE:-auto}"
CLASS_SELECTION_DIR="${CLASS_SELECTION_DIR:-/root/data/experiment/protocols}"
CLASS_SELECTION_FILE="${CLASS_SELECTION_FILE:-${CLASS_SELECTION_DIR}/second_${SPLIT}_oneclass_targetmask_both_resize256_seed${SELECTION_SEED}_labelpair${LABEL_PAIR_MODE}.jsonl}"
INSTRUCTPIX2PIX_MODEL_ID="${INSTRUCTPIX2PIX_MODEL_ID:-timbrooks/instruct-pix2pix}"
INSTRUCTPIX2PIX_WEIGHT_ROOT="${INSTRUCTPIX2PIX_WEIGHT_ROOT:-/root/data/weight/instructpix2pix}"
INSTRUCTPIX2PIX_MODEL_DIR="${INSTRUCTPIX2PIX_MODEL_DIR:-${INSTRUCTPIX2PIX_WEIGHT_ROOT}/instruct-pix2pix}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/instructpix2pix_second_${SPLIT}_${DIRECTION}_oneclass_targetmask_source_text_resize512_eval256_steps100_seed42_selectionseed${SELECTION_SEED}}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.jsonl}"
BOOTSTRAP_INSTRUCTPIX2PIX="${BOOTSTRAP_INSTRUCTPIX2PIX:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "${CLASS_SELECTION_FILE}" ]]; then
  echo "[instructpix2pix_second_gen] missing shared CLASS_SELECTION_FILE: ${CLASS_SELECTION_FILE}" >&2
  echo "Create it once with Vistar run_bash/seg/eval_flux2_second_oneclass_binarymask_gen.bash." >&2
  exit 2
fi
if [[ "${BOOTSTRAP_INSTRUCTPIX2PIX}" == "1" ]]; then
  INSTRUCTPIX2PIX_MODEL_ID="${INSTRUCTPIX2PIX_MODEL_ID}" \
  INSTRUCTPIX2PIX_MODEL_DIR="${INSTRUCTPIX2PIX_MODEL_DIR}" \
    bash "${ROOT_DIR}/scripts/bootstrap_instructpix2pix.sh"
fi
mkdir -p "${OUTPUT_DIR}"

BUILD_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then BUILD_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_rcdgen_second_manifest.py" \
  --second_root "${SECOND_ROOT}" --split "${SPLIT}" --direction "${DIRECTION}" \
  --class_selection_file "${CLASS_SELECTION_FILE}" --output "${MANIFEST}" "${BUILD_ARGS[@]}"

RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then RUN_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${OVERWRITE}" == "1" ]]; then RUN_ARGS+=(--overwrite); fi
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --no-python \
  "${PYTHON_BIN}" -u "${ROOT_DIR}/baselines/instructpix2pix/run_instructpix2pix_manifest.py" \
  --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}" --model "${INSTRUCTPIX2PIX_MODEL_DIR}" \
  --resolution "${RESOLUTION}" --eval_size "${EVAL_SIZE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --image_guidance_scale "${IMAGE_GUIDANCE_SCALE}" --guidance_scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" "${RUN_ARGS[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/merge_ranked_jsonl.py" --directory "${OUTPUT_DIR}"
echo "[instructpix2pix_second_gen] done: ${OUTPUT_DIR}"
