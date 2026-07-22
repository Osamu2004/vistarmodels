#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29632}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-/root/data/xview2/test}"
RSKT_ROOT="${RSKT_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg}"
RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT:-/root/data/weight/rskt_seg}"
RSKT_CONFIG="${RSKT_CONFIG:-${RSKT_ROOT}/configs/vitl_336_DLRSD.yaml}"
RSKT_CHECKPOINT="${RSKT_CHECKPOINT:-/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth}"
RSKT_CLASS_JSON="${RSKT_CLASS_JSON:-${ROOT_DIR}/baselines/rskt_seg/configs/xbd_pre_classes.json}"
RSKT_CLIP_VITL="${RSKT_CLIP_VITL:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-L-14-336px.pt}"
RSKT_CLIP_VITB="${RSKT_CLIP_VITB:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-B-32.pt}"
RSKT_REMOTE_CLIP="${RSKT_REMOTE_CLIP:-${RSKT_WEIGHT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"
RSKT_RSIB="${RSKT_RSIB:-/root/data/weight/rsib/RSIB.pth}"

INPUT_SIZE="${INPUT_SIZE:-512}"
TILE_SIZE="${TILE_SIZE:-512}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PROMPT_ENSEMBLE="${PROMPT_ENSEMBLE:-single}"
AMP="${AMP:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_RSKT_SEG="${BOOTSTRAP_RSKT_SEG:-1}"
RSKT_DOWNLOAD_AUX_WEIGHTS="${RSKT_DOWNLOAD_AUX_WEIGHTS:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

CHECKPOINT_TAG="$(basename "$(dirname "${RSKT_CHECKPOINT}")")"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rskt_seg_xbd_pre_${CHECKPOINT_TAG}_crossdomain_tile${TILE_SIZE}_resize${INPUT_SIZE}_${NPROC_PER_NODE}gpu}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if ! [[ "${TILE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TILE_SIZE must be a positive integer, got ${TILE_SIZE}." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

if is_truthy "${BOOTSTRAP_RSKT_SEG}"; then
  RSKT_ROOT="${RSKT_ROOT}" \
  RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT}" \
  RSKT_RSIB="${RSKT_RSIB}" \
  RSKT_CHECKPOINT="${RSKT_CHECKPOINT}" \
  RSKT_DOWNLOAD_AUX_WEIGHTS="${RSKT_DOWNLOAD_AUX_WEIGHTS}" \
    bash "${ROOT_DIR}/scripts/bootstrap_rskt_seg.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  RSKT_ROOT="${RSKT_ROOT}" \
  RSKT_CHECKPOINT="${RSKT_CHECKPOINT}" \
  RSKT_CLIP_VITL="${RSKT_CLIP_VITL}" \
  RSKT_CLIP_VITB="${RSKT_CLIP_VITB}" \
  RSKT_REMOTE_CLIP="${RSKT_REMOTE_CLIP}" \
  RSKT_RSIB="${RSKT_RSIB}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_rskt_seg_deps.py"
fi

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if is_truthy "${SAVE_IMAGES}"; then
  EXTRA_ARGS+=(--save_images)
else
  EXTRA_ARGS+=(--no-save_images)
fi
if is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/rskt_seg/eval_rskt_seg_xbd_pre.py"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --rskt_root "${RSKT_ROOT}"
  --config "${RSKT_CONFIG}"
  --checkpoint "${RSKT_CHECKPOINT}"
  --class_json "${RSKT_CLASS_JSON}"
  --clip_vitl "${RSKT_CLIP_VITL}"
  --clip_vitb "${RSKT_CLIP_VITB}"
  --remote_clip "${RSKT_REMOTE_CLIP}"
  --rsib "${RSKT_RSIB}"
  --input_size "${INPUT_SIZE}"
  --tile_size "${TILE_SIZE}"
  --num_layers "${NUM_LAYERS}"
  --prompt_ensemble "${PROMPT_ENSEMBLE}"
  --amp "${AMP}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] RSKT-Seg xBD-pre building evaluation"
echo "[$(date)] setting=DLRSD-trained cross-dataset/out-of-domain"
echo "[$(date)] data_root=${DATA_ROOT}"
echo "[$(date)] expected_layout=test/images/*_pre_disaster.png + test/labels/*_pre_disaster.json"
echo "[$(date)] checkpoint=${RSKT_CHECKPOINT}"
echo "[$(date)] classes=background,building | primary_metric=building_iou"
echo "[$(date)] boundary_metric=wfm_3px_percent (IDGBR Sobel + 3x3 dilation + Margolin WFm)"
echo "[$(date)] labels=features.xy WKT rounded and rasterized with cv2.fillPoly"
echo "[$(date)] inference=native_nonoverlap_tiled | source_tile=${TILE_SIZE} | model_input=${INPUT_SIZE}"
echo "[$(date)] tile_resize=$([[ "${TILE_SIZE}" == "${INPUT_SIZE}" ]] && echo off || echo on) | padding=zero_right_bottom | metric_size=original"
echo "[$(date)] prompt_ensemble=${PROMPT_ENSEMBLE} | amp=${AMP}"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE} | synchronization=gloo"
echo "[$(date)] bootstrap=${BOOTSTRAP_RSKT_SEG} | auto_download_aux_weights=${RSKT_DOWNLOAD_AUX_WEIGHTS}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
