#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?Set DATASET to loveda, flair, uavid, xbd_pre, or chn6_cug}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29720}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

case "${DATASET}" in
  loveda)
    DEFAULT_DATA_ROOT="/root/data/LoveDA"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/vip/configs/loveda.txt"
    DEFAULT_OUTPUT_TAG="loveda_val"
    ;;
  flair)
    DEFAULT_DATA_ROOT="/root/data/FLAIR-1-2/data/flair#1-test"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/vip/configs/flair_12.txt"
    DEFAULT_OUTPUT_TAG="flair1_test_12class"
    ;;
  uavid)
    DEFAULT_DATA_ROOT="/root/data/OVSISBenchDataset/uavid"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/vip/configs/uavid_8.txt"
    DEFAULT_OUTPUT_TAG="uavid_8class_all"
    ;;
  xbd_pre)
    DEFAULT_DATA_ROOT="/root/data/xview2/test"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/vip/configs/xbd_pre.txt"
    DEFAULT_OUTPUT_TAG="xbd_pre_test"
    ;;
  chn6_cug)
    DEFAULT_DATA_ROOT="/root/data/CHN6-CUG/val"
    DEFAULT_CLASS_FILE="${ROOT_DIR}/baselines/vip/configs/chn6_cug.txt"
    DEFAULT_OUTPUT_TAG="chn6_cug_val"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; expected loveda, flair, uavid, xbd_pre, or chn6_cug." >&2
    exit 2
    ;;
esac

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
VIP_ROOT="${VIP_ROOT:-${ROOT_DIR}/third_party/VIP}"
VIP_WEIGHT_ROOT="${VIP_WEIGHT_ROOT:-/root/data/weight/vip}"
VIP_BACKBONE="${VIP_BACKBONE:-${VIP_WEIGHT_ROOT}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
VIP_DINOTXT="${VIP_DINOTXT:-${VIP_WEIGHT_ROOT}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth}"
VIP_BPE="${VIP_BPE:-${VIP_WEIGHT_ROOT}/bpe_simple_vocab_16e6.txt.gz}"
CLASS_FILE="${CLASS_FILE:-${DEFAULT_CLASS_FILE}}"
RESIZE_POLICY="${RESIZE_POLICY:-release_max_side}"
INPUT_SIZE="${INPUT_SIZE:-448}"
SLIDE_CROP="${SLIDE_CROP:-336}"
SLIDE_STRIDE="${SLIDE_STRIDE:-112}"
LOGIT_SCALE="${LOGIT_SCALE:-40}"
TAU="${TAU:-4.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.0}"
LOW_CONFIDENCE_ACTION="${LOW_CONFIDENCE_ACTION:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MASK_ID_BASE="${MASK_ID_BASE:-auto}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_VIP="${BOOTSTRAP_VIP:-1}"
VIP_DOWNLOAD_WEIGHTS="${VIP_DOWNLOAD_WEIGHTS:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/vip_${DEFAULT_OUTPUT_TAG}_${RESIZE_POLICY}${INPUT_SIZE}_crop${SLIDE_CROP}_stride${SLIDE_STRIDE}_${NPROC_PER_NODE}gpu}"

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
if [[ "${SLIDE_CROP}" != "336" ]]; then
  echo "The released VIP head requires SLIDE_CROP=336, got ${SLIDE_CROP}." >&2
  exit 2
fi
if [[ "${MAX_SAMPLES}" != "0" ]] && ! [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be zero or a positive integer, got ${MAX_SAMPLES}." >&2
  exit 2
fi
if [[ "${RESIZE_POLICY}" != "release_max_side" ]] && \
   [[ "${RESIZE_POLICY}" != "paper_short_side" ]]; then
  echo "RESIZE_POLICY must be release_max_side or paper_short_side." >&2
  exit 2
fi
if [[ "${LOW_CONFIDENCE_ACTION}" != "auto" ]] && \
   [[ "${LOW_CONFIDENCE_ACTION}" != "background" ]] && \
   [[ "${LOW_CONFIDENCE_ACTION}" != "ignore" ]]; then
  echo "LOW_CONFIDENCE_ACTION must be auto, background, or ignore." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if is_truthy "${STRICT_PROTOCOL}"; then EXTRA_ARGS+=(--strict_protocol); else EXTRA_ARGS+=(--no-strict_protocol); fi
if is_truthy "${SAVE_IMAGES}"; then EXTRA_ARGS+=(--save_images); else EXTRA_ARGS+=(--no-save_images); fi
if is_truthy "${OVERWRITE}"; then EXTRA_ARGS+=(--overwrite); fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/vip/eval_vip.py"
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --vip_root "${VIP_ROOT}"
  --class_file "${CLASS_FILE}"
  --backbone_checkpoint "${VIP_BACKBONE}"
  --dinotxt_checkpoint "${VIP_DINOTXT}"
  --bpe_vocabulary "${VIP_BPE}"
  --resize_policy "${RESIZE_POLICY}"
  --input_size "${INPUT_SIZE}"
  --slide_crop "${SLIDE_CROP}"
  --slide_stride "${SLIDE_STRIDE}"
  --logit_scale "${LOGIT_SCALE}"
  --tau "${TAU}"
  --temperature "${TEMPERATURE}"
  --probability_threshold "${PROBABILITY_THRESHOLD}"
  --low_confidence_action "${LOW_CONFIDENCE_ACTION}"
  --mask_id_base "${MASK_ID_BASE}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] VIP training-free open-vocabulary segmentation"
echo "[$(date)] dataset=${DATASET} | data_root=${DATA_ROOT}"
echo "[$(date)] source=${VIP_ROOT} | model=frozen DINOv3 ViT-L/16 + dino.txt"
echo "[$(date)] class_file=${CLASS_FILE}"
echo "[$(date)] resize=${RESIZE_POLICY}:${INPUT_SIZE} | slide=${SLIDE_CROP}/${SLIDE_STRIDE} | metric_size=original"
echo "[$(date)] logit_scale=${LOGIT_SCALE} | tau=${TAU} | temperature=${TEMPERATURE} | probability_threshold=${PROBABILITY_THRESHOLD} | low_confidence_action=${LOW_CONFIDENCE_ACTION}"
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

if is_truthy "${BOOTSTRAP_VIP}"; then
  VIP_ROOT="${VIP_ROOT}" \
  VIP_WEIGHT_ROOT="${VIP_WEIGHT_ROOT}" \
  VIP_BACKBONE="${VIP_BACKBONE}" \
  VIP_DINOTXT="${VIP_DINOTXT}" \
  VIP_BPE="${VIP_BPE}" \
  VIP_DOWNLOAD_WEIGHTS="${VIP_DOWNLOAD_WEIGHTS}" \
    bash "${ROOT_DIR}/scripts/bootstrap_vip.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  VIP_ROOT="${VIP_ROOT}" \
  VIP_BACKBONE="${VIP_BACKBONE}" \
  VIP_DINOTXT="${VIP_DINOTXT}" \
  VIP_BPE="${VIP_BPE}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_vip_deps.py"
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
