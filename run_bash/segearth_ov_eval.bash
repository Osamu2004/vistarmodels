#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?Set DATASET to loveda, flair, xbd_pre, or chn6_cug}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29660}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

case "${DATASET}" in
  loveda)
    DEFAULT_DATA_ROOT="/root/data/LoveDA"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/segearth_ov/configs/loveda.txt"
    DEFAULT_PROBABILITY_THRESHOLD="0.3"
    DEFAULT_CLS_TOKEN_LAMBDA="-0.3"
    DEFAULT_OUTPUT_TAG="loveda_val"
    ;;
  flair)
    DEFAULT_DATA_ROOT="/root/data/FLAIR-1-2/data/flair#1-test"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/segearth_ov/configs/flair_12.txt"
    DEFAULT_PROBABILITY_THRESHOLD="0.0"
    DEFAULT_CLS_TOKEN_LAMBDA="-0.3"
    DEFAULT_OUTPUT_TAG="flair1_test_12class"
    ;;
  xbd_pre)
    DEFAULT_DATA_ROOT="/root/data/xview2/test"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/segearth_ov/configs/xbd_pre.txt"
    DEFAULT_PROBABILITY_THRESHOLD="0.0"
    DEFAULT_CLS_TOKEN_LAMBDA="0.0"
    DEFAULT_OUTPUT_TAG="xbd_pre_test"
    ;;
  chn6_cug)
    DEFAULT_DATA_ROOT="/root/data/CHN6-CUG/val"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/segearth_ov/configs/chn6_cug.txt"
    DEFAULT_PROBABILITY_THRESHOLD="0.8"
    DEFAULT_CLS_TOKEN_LAMBDA="-0.3"
    DEFAULT_OUTPUT_TAG="chn6_cug_val"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; expected loveda, flair, xbd_pre, or chn6_cug." >&2
    exit 2
    ;;
esac

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT:-${ROOT_DIR}/third_party/SegEarth-OV}"
SIMFEATUP_ROOT="${SIMFEATUP_ROOT:-${ROOT_DIR}/third_party/SimFeatUp}"
SEGEARTH_OV_WEIGHT_ROOT="${SEGEARTH_OV_WEIGHT_ROOT:-/root/data/weight/segearth_ov}"
SEGEARTH_OV_CLIP_VITB="${SEGEARTH_OV_CLIP_VITB:-${SEGEARTH_OV_WEIGHT_ROOT}/pretrained/ViT-B-16.pt}"
SEGEARTH_OV_SIMFEATUP="${SEGEARTH_OV_SIMFEATUP:-${SEGEARTH_OV_ROOT}/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt}"
CLASS_FILE="${CLASS_FILE:-${DEFAULT_CLASS_FILE}}"
INPUT_SIZE="${INPUT_SIZE:-448}"
SLIDE_CROP="${SLIDE_CROP:-224}"
SLIDE_STRIDE="${SLIDE_STRIDE:-112}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-${DEFAULT_PROBABILITY_THRESHOLD}}"
CLS_TOKEN_LAMBDA="${CLS_TOKEN_LAMBDA:-${DEFAULT_CLS_TOKEN_LAMBDA}}"
FEATURE_UP="${FEATURE_UP:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_SEGEARTH_OV="${BOOTSTRAP_SEGEARTH_OV:-1}"
SEGEARTH_OV_DOWNLOAD_WEIGHTS="${SEGEARTH_OV_DOWNLOAD_WEIGHTS:-1}"
SEGEARTH_OV_INSTALL_SIMFEATUP="${SEGEARTH_OV_INSTALL_SIMFEATUP:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/segearth_ov_${DEFAULT_OUTPUT_TAG}_clip_vitb16_input${INPUT_SIZE}_slide${SLIDE_CROP}_stride${SLIDE_STRIDE}_${NPROC_PER_NODE}gpu}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if ! [[ "${INPUT_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${SLIDE_CROP}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${SLIDE_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INPUT_SIZE, SLIDE_CROP, and SLIDE_STRIDE must be positive integers." >&2
  exit 2
fi
if (( SLIDE_CROP > INPUT_SIZE )); then
  echo "SLIDE_CROP=${SLIDE_CROP} cannot exceed INPUT_SIZE=${INPUT_SIZE}." >&2
  exit 2
fi
if [[ "${MAX_SAMPLES}" != "0" ]] && ! [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be zero or a positive integer, got ${MAX_SAMPLES}." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if is_truthy "${FEATURE_UP}"; then EXTRA_ARGS+=(--feature_up); else EXTRA_ARGS+=(--no-feature_up); fi
if is_truthy "${STRICT_PROTOCOL}"; then EXTRA_ARGS+=(--strict_protocol); else EXTRA_ARGS+=(--no-strict_protocol); fi
if is_truthy "${SAVE_IMAGES}"; then EXTRA_ARGS+=(--save_images); else EXTRA_ARGS+=(--no-save_images); fi
if is_truthy "${OVERWRITE}"; then EXTRA_ARGS+=(--overwrite); fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/segearth_ov/eval_segearth_ov.py"
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --segearth_root "${SEGEARTH_OV_ROOT}"
  --simfeatup_root "${SIMFEATUP_ROOT}"
  --class_file "${CLASS_FILE}"
  --simfeatup_checkpoint "${SEGEARTH_OV_SIMFEATUP}"
  --clip_vitb "${SEGEARTH_OV_CLIP_VITB}"
  --input_size "${INPUT_SIZE}"
  --slide_crop "${SLIDE_CROP}"
  --slide_stride "${SLIDE_STRIDE}"
  --probability_threshold "${PROBABILITY_THRESHOLD}"
  --cls_token_lambda "${CLS_TOKEN_LAMBDA}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] standalone SegEarth-OV evaluation"
echo "[$(date)] dataset=${DATASET} | data_root=${DATA_ROOT}"
echo "[$(date)] source=${SEGEARTH_OV_ROOT} | model=CLIP ViT-B/16 + SimFeatUp"
echo "[$(date)] simfeatup_source=${SIMFEATUP_ROOT} | checkpoint=${SEGEARTH_OV_SIMFEATUP}"
echo "[$(date)] class_file=${CLASS_FILE}"
echo "[$(date)] input=keep-ratio-${INPUT_SIZE} | internal_slide=${SLIDE_CROP}/${SLIDE_STRIDE} | metric_size=original"
echo "[$(date)] probability_threshold=${PROBABILITY_THRESHOLD} | cls_token_lambda=${CLS_TOKEN_LAMBDA} | feature_up=${FEATURE_UP}"
echo "[$(date)] metrics=IoU/mIoU,mF1,mAcc,pixel_accuracy,wfm_3px_percent"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE} | synchronization=gloo"
echo "[$(date)] strict_protocol=${STRICT_PROTOCOL} | max_samples=${MAX_SAMPLES} | overwrite=${OVERWRITE}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if is_truthy "${BOOTSTRAP_SEGEARTH_OV}"; then
  SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT}" \
  SIMFEATUP_ROOT="${SIMFEATUP_ROOT}" \
  SEGEARTH_OV_WEIGHT_ROOT="${SEGEARTH_OV_WEIGHT_ROOT}" \
  SEGEARTH_OV_CLIP_VITB="${SEGEARTH_OV_CLIP_VITB}" \
  SEGEARTH_OV_SIMFEATUP="${SEGEARTH_OV_SIMFEATUP}" \
  SEGEARTH_OV_DOWNLOAD_WEIGHTS="${SEGEARTH_OV_DOWNLOAD_WEIGHTS}" \
  SEGEARTH_OV_INSTALL_SIMFEATUP="${SEGEARTH_OV_INSTALL_SIMFEATUP}" \
    bash "${ROOT_DIR}/scripts/bootstrap_segearth_ov.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT}" \
  SIMFEATUP_ROOT="${SIMFEATUP_ROOT}" \
  SEGEARTH_OV_CLIP_VITB="${SEGEARTH_OV_CLIP_VITB}" \
  SEGEARTH_OV_SIMFEATUP="${SEGEARTH_OV_SIMFEATUP}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_segearth_ov_deps.py"
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
