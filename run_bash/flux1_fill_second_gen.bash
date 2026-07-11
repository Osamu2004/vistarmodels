#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"
export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
SPLIT="${SPLIT:-test}"
DIRECTION="${DIRECTION:-both}"
SELECTION_SEED="${SELECTION_SEED:-42}"
LABEL_PAIR_MODE="${LABEL_PAIR_MODE:-auto}"
CLASS_SELECTION_DIR="${CLASS_SELECTION_DIR:-/root/data/experiment/protocols}"
CLASS_SELECTION_FILE="${CLASS_SELECTION_FILE:-${CLASS_SELECTION_DIR}/second_${SPLIT}_oneclass_targetmask_both_resize256_seed${SELECTION_SEED}_labelpair${LABEL_PAIR_MODE}.jsonl}"
FLUX1_FILL_MODEL_ID="${FLUX1_FILL_MODEL_ID:-black-forest-labs/FLUX.1-Fill-dev}"
FLUX1_FILL_WEIGHT_ROOT="${FLUX1_FILL_WEIGHT_ROOT:-/root/data/weight/flux1_fill}"
FLUX1_FILL_MODEL_DIR="${FLUX1_FILL_MODEL_DIR:-${FLUX1_FILL_WEIGHT_ROOT}/FLUX.1-Fill-dev}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-30.0}"
GUIDANCE_TAG="${GUIDANCE_SCALE/./p}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
VAE_TILING="${VAE_TILING:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_FLUX1_FILL="${BOOTSTRAP_FLUX1_FILL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/flux1_fill_second_${SPLIT}_${DIRECTION}_oneclass_targetmask_source_binarymask_text_resize${RESOLUTION}_eval${EVAL_SIZE}_steps${NUM_INFERENCE_STEPS}_cfg${GUIDANCE_TAG}_seed${SEED}_selectionseed${SELECTION_SEED}_1gpu}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.jsonl}"

if [[ ! -f "${CLASS_SELECTION_FILE}" ]]; then
  echo "[flux1_fill_second_gen] missing shared CLASS_SELECTION_FILE: ${CLASS_SELECTION_FILE}" >&2
  echo "Create it once with Vistar run_bash/seg/eval_flux2_second_oneclass_binarymask_gen.bash." >&2
  exit 2
fi
if [[ "${BOOTSTRAP_FLUX1_FILL}" == "1" ]]; then
  FLUX1_FILL_MODEL_ID="${FLUX1_FILL_MODEL_ID}" FLUX1_FILL_MODEL_DIR="${FLUX1_FILL_MODEL_DIR}" \
    bash "${ROOT_DIR}/scripts/bootstrap_flux1_fill.sh"
fi
mkdir -p "${OUTPUT_DIR}"

BUILD_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then BUILD_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_rcdgen_second_manifest.py" \
  --second_root "${SECOND_ROOT}" --split "${SPLIT}" --direction "${DIRECTION}" \
  --consumer flux1_fill --class_selection_file "${CLASS_SELECTION_FILE}" --output "${MANIFEST}" "${BUILD_ARGS[@]}"

RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then RUN_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${OVERWRITE}" == "1" ]]; then RUN_ARGS+=(--overwrite); fi
case "${CPU_OFFLOAD,,}" in
  0|false|no|n|off) RUN_ARGS+=(--no-cpu_offload) ;;
  *) RUN_ARGS+=(--cpu_offload) ;;
esac
if [[ "${VAE_TILING}" == "1" || "${VAE_TILING}" == "true" || "${VAE_TILING}" == "yes" ]]; then
  RUN_ARGS+=(--vae_tiling)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/baselines/flux1_fill/run_flux1_fill_manifest.py" \
  --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}" --model "${FLUX1_FILL_MODEL_DIR}" \
  --resolution "${RESOLUTION}" --eval_size "${EVAL_SIZE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" --guidance_scale "${GUIDANCE_SCALE}" \
  --max_sequence_length "${MAX_SEQUENCE_LENGTH}" --seed "${SEED}" --dtype "${DTYPE}" "${RUN_ARGS[@]}"

echo "[flux1_fill_second_gen] done: ${OUTPUT_DIR}"
