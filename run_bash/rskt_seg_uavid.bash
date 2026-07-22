#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29636}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-/root/data/OVSISBenchDataset/uavid}"
RSKT_ROOT="${RSKT_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg}"
RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT:-/root/data/weight/rskt_seg}"
RSKT_CONFIG="${RSKT_CONFIG:-${RSKT_ROOT}/configs/vitl_336_DLRSD.yaml}"
RSKT_CHECKPOINT="${RSKT_CHECKPOINT:-/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth}"
RSKT_CLASS_JSON="${RSKT_CLASS_JSON:-${ROOT_DIR}/baselines/rskt_seg/configs/uavid_8_classes.json}"
RSKT_CLIP_VITL="${RSKT_CLIP_VITL:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-L-14-336px.pt}"
RSKT_CLIP_VITB="${RSKT_CLIP_VITB:-${RSKT_WEIGHT_ROOT}/pretrained/ViT-B-32.pt}"
RSKT_REMOTE_CLIP="${RSKT_REMOTE_CLIP:-${RSKT_WEIGHT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"
RSKT_RSIB="${RSKT_RSIB:-/root/data/weight/rsib/RSIB.pth}"

MIN_SIZE_TEST="${MIN_SIZE_TEST:-640}"
MAX_SIZE_TEST="${MAX_SIZE_TEST:-2560}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PROMPT_ENSEMBLE="${PROMPT_ENSEMBLE:-single}"
AMP="${AMP:-fp32}"
MASK_ID_BASE="${MASK_ID_BASE:-auto}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-270}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
COMPUTE_WFM="${COMPUTE_WFM:-1}"
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
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rskt_seg_uavid_${CHECKPOINT_TAG}_8class_all_whole_min${MIN_SIZE_TEST}_max${MAX_SIZE_TEST}_gpu${GPU_IDS//,/_}}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
if ! [[ "${MIN_SIZE_TEST}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${MAX_SIZE_TEST}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIN_SIZE_TEST and MAX_SIZE_TEST must be positive integers." >&2
  exit 2
fi
if (( MAX_SIZE_TEST < MIN_SIZE_TEST )); then
  echo "MAX_SIZE_TEST must be greater than or equal to MIN_SIZE_TEST." >&2
  exit 2
fi
if [[ "${NUM_LAYERS}" != "5" ]]; then
  echo "The released DLRSD+ViT-L checkpoint requires NUM_LAYERS=5." >&2
  exit 2
fi
if [[ "${EXPECTED_SAMPLES}" != "270" ]] && is_truthy "${STRICT_PROTOCOL}"; then
  echo "Strict UAVid evaluation requires EXPECTED_SAMPLES=270." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if is_truthy "${COMPUTE_WFM}"; then
  EXTRA_ARGS+=(--compute_wfm)
else
  EXTRA_ARGS+=(--no-compute_wfm)
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
  "${ROOT_DIR}/baselines/rskt_seg/eval_rskt_seg_uavid.py"
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
  --min_size_test "${MIN_SIZE_TEST}"
  --max_size_test "${MAX_SIZE_TEST}"
  --num_layers "${NUM_LAYERS}"
  --prompt_ensemble "${PROMPT_ENSEMBLE}"
  --amp "${AMP}"
  --mask_id_base "${MASK_ID_BASE}"
  --expected_samples "${EXPECTED_SAMPLES}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] RSKT-Seg UAVid evaluation"
echo "[$(date)] data_root=${DATA_ROOT} | expected_samples=270 | strict=${STRICT_PROTOCOL}"
echo "[$(date)] classes=8 | class0=Background clutter | extra_negative_class=false"
echo "[$(date)] palette=VISTAR UAVid | moving/static car kept separate"
echo "[$(date)] inference=whole_image_shortest_edge | min=${MIN_SIZE_TEST} | max=${MAX_SIZE_TEST} | output=original"
echo "[$(date)] sliding_window=false | pooling=[1,1] | prompt=${PROMPT_ENSEMBLE} | amp=${AMP}"
echo "[$(date)] metrics=mIoU(all 8 primary),mIoU(foreground 7 auxiliary),mACC,mF1,pixel_accuracy,WFm"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE} | output=${OUTPUT_DIR}"

# Dry-run is intentionally dependency- and network-free.
if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
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

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
