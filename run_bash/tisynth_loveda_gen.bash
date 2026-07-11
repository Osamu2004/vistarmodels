#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TQDM_DISABLE=0

# The unified 512x512 protocol reads the exact condition and GT files produced
# by Vistar. REFERENCE_DIR must be an independent training/external RGB pool;
# TISynth is a mask+text+reference model, so this input cannot be omitted.
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_loveda_gen_gen_only_step300000}"
REFERENCE_DIR="${REFERENCE_DIR:-}"

TISYNTH_ROOT="${TISYNTH_ROOT:-${ROOT_DIR}/third_party/TISynth}"
TISYNTH_WEIGHT_DIR="${TISYNTH_WEIGHT_DIR:-/root/data/weight/TISynth}"
TISYNTH_CKPT="${TISYNTH_CKPT:-${TISYNTH_WEIGHT_DIR}/GID_model.ckpt}"
TISYNTH_CONFIG="${TISYNTH_CONFIG:-${TISYNTH_ROOT}/models/cldm_ssl_v15_aia_v0_augmentation.yaml}"
TISYNTH_CLIP_VERSION="${TISYNTH_CLIP_VERSION:-openai/clip-vit-large-patch14}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/tisynth_gid_zeroshot_loveda_mask_to_rgb_gen_resize512_steps50_cfg9p0_seed0_refseed0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_tisynth.jsonl}"

BOOTSTRAP_TISYNTH="${BOOTSTRAP_TISYNTH:-1}"
RESOLUTION="${RESOLUTION:-512}"
EVAL_SIZE="${EVAL_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DDIM_STEPS="${DDIM_STEPS:-50}"
SCALE="${SCALE:-9.0}"
STRENGTH="${STRENGTH:-1.0}"
ETA="${ETA:-0.0}"
SEED="${SEED:-0}"
REFERENCE_SEED="${REFERENCE_SEED:-0}"
PRECISION="${PRECISION:-fp16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
STRICT_PALETTE="${STRICT_PALETTE:-1}"
ALLOW_EVAL_GT_REFERENCE_POOL="${ALLOW_EVAL_GT_REFERENCE_POOL:-0}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a high-resolution remote sensing satellite image}"

# Metrics run in the main Vistar environment. Leave disabled during TISynth
# generation if the two models use separate conda environments.
RUN_METRICS="${RUN_METRICS:-0}"
METRIC_CODE_DIR="${METRIC_CODE_DIR:-/root/code/vistar}"
METRIC_GPU_ID="${METRIC_GPU_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-16}"

_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if _truthy "${BOOTSTRAP_TISYNTH}"; then
  TISYNTH_ROOT="${TISYNTH_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_tisynth.sh"
fi
if [[ ! -d "${VISTAR_EVAL_DIR}/cond_mask" || ! -d "${VISTAR_EVAL_DIR}/gt_rgb" ]]; then
  echo "[tisynth_loveda_gen] VISTAR_EVAL_DIR must contain cond_mask and gt_rgb: ${VISTAR_EVAL_DIR}" >&2
  exit 1
fi
if [[ -z "${REFERENCE_DIR}" ]]; then
  echo "[tisynth_loveda_gen] REFERENCE_DIR is required." >&2
  echo "Use an independent LoveDA training/external RGB pool; never use the paired GT itself as reference." >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "[tisynth_loveda_gen] Missing REFERENCE_DIR: ${REFERENCE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TISYNTH_CKPT}" ]]; then
  echo "[tisynth_loveda_gen] Missing TISynth inference checkpoint: ${TISYNTH_CKPT}" >&2
  echo "The default zero-shot protocol expects ${TISYNTH_WEIGHT_DIR}/GID_model.ckpt." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "[tisynth_loveda_gen] VISTAR_EVAL_DIR=${VISTAR_EVAL_DIR}"
echo "[tisynth_loveda_gen] REFERENCE_DIR=${REFERENCE_DIR} reference_seed=${REFERENCE_SEED}"
echo "[tisynth_loveda_gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[tisynth_loveda_gen] TISYNTH_ROOT=${TISYNTH_ROOT}"
echo "[tisynth_loveda_gen] TISYNTH_WEIGHT_DIR=${TISYNTH_WEIGHT_DIR}"
echo "[tisynth_loveda_gen] TISYNTH_CKPT=${TISYNTH_CKPT}"
echo "[tisynth_loveda_gen] TISYNTH_CONFIG=${TISYNTH_CONFIG}"
echo "[tisynth_loveda_gen] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[tisynth_loveda_gen] resolution=${RESOLUTION} eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE}"
echo "[tisynth_loveda_gen] steps=${DDIM_STEPS} cfg=${SCALE} strength=${STRENGTH} eta=${ETA} seed=${SEED} precision=${PRECISION}"
echo "[tisynth_loveda_gen] resume=$((1-OVERWRITE)) max_samples=${MAX_SAMPLES} strict_palette=${STRICT_PALETTE}"

MANIFEST_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  MANIFEST_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if ! _truthy "${STRICT_PALETTE}"; then
  MANIFEST_ARGS+=(--no-strict_palette)
fi
if _truthy "${ALLOW_EVAL_GT_REFERENCE_POOL}"; then
  MANIFEST_ARGS+=(--allow_eval_gt_reference_pool)
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_tisynth_loveda_manifest.py" \
  --eval_dir "${VISTAR_EVAL_DIR}" \
  --reference_dir "${REFERENCE_DIR}" \
  --output "${MANIFEST}" \
  --seed "${REFERENCE_SEED}" \
  --prompt_prefix "${PROMPT_PREFIX}" \
  "${MANIFEST_ARGS[@]}"

RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  RUN_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if _truthy "${OVERWRITE}"; then
  RUN_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" "${ROOT_DIR}/baselines/tisynth/run_tisynth_manifest.py" \
  --tisynth_root "${TISYNTH_ROOT}" \
  --config "${TISYNTH_CONFIG}" \
  --ckpt "${TISYNTH_CKPT}" \
  --clip_version "${TISYNTH_CLIP_VERSION}" \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution "${RESOLUTION}" \
  --eval_size "${EVAL_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --ddim_steps "${DDIM_STEPS}" \
  --scale "${SCALE}" \
  --strength "${STRENGTH}" \
  --eta "${ETA}" \
  --seed "${SEED}" \
  --precision "${PRECISION}" \
  "${RUN_ARGS[@]}"

if _truthy "${RUN_METRICS}"; then
  if [[ ! -f "${METRIC_CODE_DIR}/run_bash/seg/compute_saved_loveda_gen_metrics.bash" ]]; then
    echo "[tisynth_loveda_gen] Missing Vistar metric script under METRIC_CODE_DIR=${METRIC_CODE_DIR}" >&2
    exit 1
  fi
  CODE_DIR="${METRIC_CODE_DIR}" GPU_ID="${METRIC_GPU_ID}" \
    METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE}" INPUT_SIZE="${EVAL_SIZE}" \
    bash "${METRIC_CODE_DIR}/run_bash/seg/compute_saved_loveda_gen_metrics.bash" "${OUTPUT_DIR}"
fi

echo "[tisynth_loveda_gen] done: ${OUTPUT_DIR}"
