#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29634}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-/root/data/FLAIR-1-2/data/flair#1-test}"
RSKT_ROOT="${RSKT_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg}"
RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT:-/root/data/weight/rskt_seg}"
RSKT_CONFIG="${RSKT_CONFIG:-${RSKT_ROOT}/configs/vitl_336_DLRSD.yaml}"
RSKT_CHECKPOINT="${RSKT_CHECKPOINT:-/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth}"
RSKT_CLASS_JSON="${RSKT_CLASS_JSON:-${ROOT_DIR}/baselines/rskt_seg/configs/flair_12_classes.json}"
RSKT_CLIP_VITL="${RSKT_CLIP_VITL:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-L-14-336px.pt}"
RSKT_CLIP_VITB="${RSKT_CLIP_VITB:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-B-32.pt}"
RSKT_REMOTE_CLIP="${RSKT_REMOTE_CLIP:-${RSKT_WEIGHT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"
RSKT_RSIB="${RSKT_RSIB:-/root/data/weight/rsib/RSIB.pth}"

INPUT_SIZE="${INPUT_SIZE:-640}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PROMPT_ENSEMBLE="${PROMPT_ENSEMBLE:-single}"
AMP="${AMP:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_PRED_RGB="${SAVE_PRED_RGB:-1}"
SAVE_GT_RGB="${SAVE_GT_RGB:-1}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
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
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rskt_seg_flair1_${CHECKPOINT_TAG}_crossdomain_resize${INPUT_SIZE}_${NPROC_PER_NODE}gpu}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if ! [[ "${INPUT_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INPUT_SIZE must be a positive integer, got ${INPUT_SIZE}." >&2
  exit 2
fi
if [[ "${NUM_LAYERS}" != "5" ]]; then
  echo "The released DLRSD+ViT-L checkpoint requires NUM_LAYERS=5." >&2
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
if is_truthy "${SAVE_PRED_RGB}"; then
  EXTRA_ARGS+=(--save_pred_rgb)
else
  EXTRA_ARGS+=(--no-save_pred_rgb)
fi
if is_truthy "${SAVE_GT_RGB}"; then
  EXTRA_ARGS+=(--save_gt_rgb)
else
  EXTRA_ARGS+=(--no-save_gt_rgb)
fi
if ! is_truthy "${STRICT_PROTOCOL}"; then
  EXTRA_ARGS+=(--no_strict_protocol)
fi
if is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/rskt_seg/eval_rskt_seg_flair.py"
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
  --num_layers "${NUM_LAYERS}"
  --prompt_ensemble "${PROMPT_ENSEMBLE}"
  --amp "${AMP}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] RSKT-Seg FLAIR#1 evaluation"
echo "[$(date)] setting=DLRSD-trained cross-dataset/out-of-domain"
echo "[$(date)] official_rskt_flair_reproduction=false"
echo "[$(date)] paper_table_ovrsisbenchv2_comparable=false"
echo "[$(date)] data_root=${DATA_ROOT}"
echo "[$(date)] expected_samples=15700 | expected_zones=193 | strict=${STRICT_PROTOCOL}"
echo "[$(date)] checkpoint=${RSKT_CHECKPOINT}"
echo "[$(date)] classes=GSNet FLAIR 12-class protocol | raw_ids=1..12->0..11 | other=ignore"
echo "[$(date)] source_bands=RGB 1-3 of official 5-band GeoTIFF"
echo "[$(date)] inference=whole_patch_shortest_edge_resize | source=512x512 | model_input=${INPUT_SIZE}x${INPUT_SIZE} | output=512x512"
echo "[$(date)] metrics=mIoU,mACC,mF1,pixel_accuracy,wfm_3px_percent"
echo "[$(date)] save=pred_mask,gt_mask,pred_rgb:${SAVE_PRED_RGB},gt_rgb:${SAVE_GT_RGB}"
echo "[$(date)] prompt_ensemble=${PROMPT_ENSEMBLE} | num_layers=${NUM_LAYERS} | amp=${AMP}"
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
