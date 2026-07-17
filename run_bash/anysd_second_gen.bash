#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
SPLIT="${SPLIT:-test}"
DIRECTION="${DIRECTION:-both}"
MASK_MODE="${MASK_MODE:-full_multiclass}"
SELECTION_SEED="${SELECTION_SEED:-42}"
LABEL_PAIR_MODE="${LABEL_PAIR_MODE:-auto}"
SEMANTIC_ZERO_IS_CLASS="${SEMANTIC_ZERO_IS_CLASS:-0}"
CLASS_SELECTION_DIR="${CLASS_SELECTION_DIR:-/root/data/experiment/protocols}"
CLASS_SELECTION_FILE="${CLASS_SELECTION_FILE:-${CLASS_SELECTION_DIR}/second_${SPLIT}_oneclass_targetmask_both_resize256_seed${SELECTION_SEED}_labelpair${LABEL_PAIR_MODE}.jsonl}"
ANYSD_ROOT="${ANYSD_ROOT:-${ROOT_DIR}/third_party/AnySD}"
ANYSD_MODEL_ID="${ANYSD_MODEL_ID:-WeiChow/AnySD}"
ANYSD_WEIGHT_ROOT="${ANYSD_WEIGHT_ROOT:-/root/data/weight/anysd}"
ANYSD_MODEL_DIR="${ANYSD_MODEL_DIR:-${ANYSD_WEIGHT_ROOT}/AnySD}"
ANYSD_BASE_MODEL_ID="${ANYSD_BASE_MODEL_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
ANYSD_BASE_MODEL_DIR="${ANYSD_BASE_MODEL_DIR:-${ANYSD_WEIGHT_ROOT}/stable-diffusion-v1-5}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-256}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-2.0}"
REFERENCE_IMAGE_GUIDANCE_SCALE="${REFERENCE_IMAGE_GUIDANCE_SCALE:-0.8}"
PROMPT_MODE="${PROMPT_MODE:-official_visual_segment}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ONLY_CHANGED="${ONLY_CHANGED:-0}"
XFORMERS="${XFORMERS:-0}"
VAE_TILING="${VAE_TILING:-0}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_ANYSD="${BOOTSTRAP_ANYSD:-0}"
CHECK_DEPS="${CHECK_DEPS:-1}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
COMPUTE_METRICS="${COMPUTE_METRICS:-0}"
VISTAR_ROOT="${VISTAR_ROOT:-/root/code/vistar}"
METRIC_GPU_ID="${METRIC_GPU_ID:-${GPU_IDS%%,*}}"
GUIDANCE_TAG="${GUIDANCE_SCALE/./p}"
IMAGE_GUIDANCE_TAG="${IMAGE_GUIDANCE_SCALE/./p}"
REFERENCE_GUIDANCE_TAG="${REFERENCE_IMAGE_GUIDANCE_SCALE/./p}"
PROTOCOL_TAG="multiclass_targetmask"
if [[ "${MASK_MODE}" == "oneclass" ]]; then PROTOCOL_TAG="oneclass_targetmask"; fi
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/anysd_second_${SPLIT}_${DIRECTION}_${PROTOCOL_TAG}_visualseg_resize${RESOLUTION}_eval${EVAL_SIZE}_steps${NUM_INFERENCE_STEPS}_cfg${GUIDANCE_TAG}_imgcfg${IMAGE_GUIDANCE_TAG}_ref${REFERENCE_GUIDANCE_TAG}_seed${SEED}}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.jsonl}"

if [[ "${MASK_MODE}" != "full_multiclass" && "${MASK_MODE}" != "oneclass" ]]; then
  echo "[anysd_second_gen] MASK_MODE must be full_multiclass or oneclass, got: ${MASK_MODE}" >&2
  exit 2
fi

if [[ "${MASK_MODE}" == "oneclass" && ! -f "${CLASS_SELECTION_FILE}" ]]; then
  echo "[anysd_second_gen] missing shared CLASS_SELECTION_FILE: ${CLASS_SELECTION_FILE}" >&2
  echo "Create it once with Vistar run_bash/seg/eval_flux2_second_oneclass_binarymask_gen.bash." >&2
  exit 2
fi

if [[ "${BOOTSTRAP_ANYSD}" == "1" ]]; then
  ANYSD_ROOT="${ANYSD_ROOT}" ANYSD_MODEL_ID="${ANYSD_MODEL_ID}" \
  ANYSD_WEIGHT_ROOT="${ANYSD_WEIGHT_ROOT}" ANYSD_MODEL_DIR="${ANYSD_MODEL_DIR}" \
  ANYSD_BASE_MODEL_ID="${ANYSD_BASE_MODEL_ID}" ANYSD_BASE_MODEL_DIR="${ANYSD_BASE_MODEL_DIR}" \
    bash "${ROOT_DIR}/scripts/bootstrap_anysd.sh"
fi
if [[ "${CHECK_DEPS}" == "1" ]]; then
  ANYSD_ROOT="${ANYSD_ROOT}" ANYSD_MODEL_DIR="${ANYSD_MODEL_DIR}" \
  ANYSD_BASE_MODEL_DIR="${ANYSD_BASE_MODEL_DIR}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_anysd_deps.py"
fi

mkdir -p "${OUTPUT_DIR}"
BUILD_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then BUILD_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${MASK_MODE}" == "full_multiclass" ]]; then
  SEMANTIC_ZERO_ARG="--no-semantic_zero_is_class"
  if [[ "${SEMANTIC_ZERO_IS_CLASS}" == "1" ]]; then SEMANTIC_ZERO_ARG="--semantic_zero_is_class"; fi
  REUSE_ARGS=()
  if [[ "${REBUILD_MANIFEST}" != "1" ]]; then REUSE_ARGS+=(--reuse_if_valid); fi
  echo "[anysd_second_gen] validating/building the full-multiclass manifest"
  echo "[anysd_second_gen] first build reads every SECOND label pair and writes directional masks"
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/build_anysd_second_manifest.py" \
    --second_root "${SECOND_ROOT}" --split "${SPLIT}" --direction "${DIRECTION}" \
    --label_pair_mode "${LABEL_PAIR_MODE}" "${SEMANTIC_ZERO_ARG}" \
    --output "${MANIFEST}" "${REUSE_ARGS[@]}" "${BUILD_ARGS[@]}"
else
  if [[ -s "${MANIFEST}" && "${REBUILD_MANIFEST}" != "1" ]]; then
    echo "[anysd_second_gen] reusing existing one-class manifest: ${MANIFEST}"
  else
    echo "[anysd_second_gen] building the one-class manifest"
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/build_rcdgen_second_manifest.py" \
      --second_root "${SECOND_ROOT}" --split "${SPLIT}" --direction "${DIRECTION}" \
      --consumer anysd --class_selection_file "${CLASS_SELECTION_FILE}" \
      --output "${MANIFEST}" "${BUILD_ARGS[@]}"
  fi
fi

RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then RUN_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${ONLY_CHANGED}" == "1" ]]; then RUN_ARGS+=(--only_changed); fi
if [[ "${OVERWRITE}" == "1" ]]; then RUN_ARGS+=(--overwrite); fi
if [[ "${XFORMERS}" == "1" ]]; then RUN_ARGS+=(--xformers); else RUN_ARGS+=(--no-xformers); fi
if [[ "${VAE_TILING}" == "1" ]]; then RUN_ARGS+=(--vae_tiling); else RUN_ARGS+=(--no-vae_tiling); fi

echo "[anysd_second_gen] SECOND_ROOT=${SECOND_ROOT} split=${SPLIT} direction=${DIRECTION}"
echo "[anysd_second_gen] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE}"
echo "[anysd_second_gen] expert=visual_segment steps=${NUM_INFERENCE_STEPS} cfg=${GUIDANCE_SCALE} image_cfg=${IMAGE_GUIDANCE_SCALE} reference_scale=${REFERENCE_IMAGE_GUIDANCE_SCALE}"
echo "[anysd_second_gen] mask_mode=${MASK_MODE} label_pair_mode=${LABEL_PAIR_MODE} semantic_zero_is_class=${SEMANTIC_ZERO_IS_CLASS}"
echo "[anysd_second_gen] model inputs=source RGB + full multi-class color change mask + instruction"
if [[ "${MASK_MODE}" == "oneclass" ]]; then
  echo "[anysd_second_gen] compatibility inputs=source RGB + selected one-class semantic mask + instruction"
fi
echo "[anysd_second_gen] output=${OUTPUT_DIR}"

LAUNCH_ARGS=(
  "${ROOT_DIR}/baselines/anysd/run_anysd_manifest.py"
  --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}"
  --anysd_root "${ANYSD_ROOT}" --model "${ANYSD_MODEL_DIR}" --base_model "${ANYSD_BASE_MODEL_DIR}"
  --resolution "${RESOLUTION}" --eval_size "${EVAL_SIZE}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}" --image_guidance_scale "${IMAGE_GUIDANCE_SCALE}"
  --reference_image_guidance_scale "${REFERENCE_IMAGE_GUIDANCE_SCALE}"
  --prompt_mode "${PROMPT_MODE}" --mask_mode "${MASK_MODE}" --seed "${SEED}"
  "${RUN_ARGS[@]}"
)
if command -v torchrun >/dev/null 2>&1; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${LAUNCH_ARGS[@]}"
else
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${LAUNCH_ARGS[@]}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/merge_ranked_jsonl.py" --directory "${OUTPUT_DIR}"

if [[ "${COMPUTE_METRICS}" == "1" ]]; then
  if [[ ! -f "${VISTAR_ROOT}/run_bash/seg/compute_saved_second_gen_metrics.bash" ]]; then
    echo "[anysd_second_gen] Missing VISTAR metric launcher under ${VISTAR_ROOT}" >&2
    exit 2
  fi
  CDGEN_METRICS_VALUE="${CDGEN_METRICS:-1}"
  if [[ "${DIRECTION}" != "both" ]]; then CDGEN_METRICS_VALUE=0; fi
  CODE_DIR="${VISTAR_ROOT}" GPU_IDS="${METRIC_GPU_ID}" SPLIT="${SPLIT}" \
  DIRECTIONS="${DIRECTION}" INPUT_SIZE="${EVAL_SIZE}" CONDITION_MASK_MODE=semantic \
  LABEL_MODE=semantic_pair LABEL_PAIR_MODE="${LABEL_PAIR_MODE}" NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  CDGEN_METRICS="${CDGEN_METRICS_VALUE}" \
    bash "${VISTAR_ROOT}/run_bash/seg/compute_saved_second_gen_metrics.bash" "${OUTPUT_DIR}"
fi

echo "[anysd_second_gen] done: ${OUTPUT_DIR}"
