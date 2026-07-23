#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29636}"
RSKT_DATASET_KEY="${RSKT_DATASET_KEY:?RSKT_DATASET_KEY is required}"
RSKT_DATASET_NAME="${RSKT_DATASET_NAME:?RSKT_DATASET_NAME is required}"
RSKT_DEFAULT_DATA_ROOT="${RSKT_DEFAULT_DATA_ROOT:?RSKT_DEFAULT_DATA_ROOT is required}"
RSKT_EVALUATOR="${RSKT_EVALUATOR:?RSKT_EVALUATOR is required}"
RSKT_CLASS_FILENAME="${RSKT_CLASS_FILENAME:?RSKT_CLASS_FILENAME is required}"
RSKT_EXPECTED_SAMPLES="${RSKT_EXPECTED_SAMPLES:?RSKT_EXPECTED_SAMPLES is required}"
RSKT_OUTPUT_SLUG="${RSKT_OUTPUT_SLUG:?RSKT_OUTPUT_SLUG is required}"
RSKT_CLASS_DESCRIPTION="${RSKT_CLASS_DESCRIPTION:?RSKT_CLASS_DESCRIPTION is required}"
RSKT_METRIC_DESCRIPTION="${RSKT_METRIC_DESCRIPTION:?RSKT_METRIC_DESCRIPTION is required}"
RSKT_TAXONOMY_DESCRIPTION="${RSKT_TAXONOMY_DESCRIPTION:?RSKT_TAXONOMY_DESCRIPTION is required}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-${RSKT_DEFAULT_DATA_ROOT}}"
RSKT_ROOT="${RSKT_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg}"
RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT:-/root/data/weight/rskt_seg}"
RSKT_CONFIG="${RSKT_CONFIG:-${RSKT_ROOT}/configs/vitl_336_DLRSD.yaml}"
RSKT_CHECKPOINT="${RSKT_CHECKPOINT:-/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth}"
RSKT_CLASS_JSON="${RSKT_CLASS_JSON:-${ROOT_DIR}/baselines/rskt_seg/configs/${RSKT_CLASS_FILENAME}}"
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
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-${RSKT_EXPECTED_SAMPLES}}"
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
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rskt_seg_${RSKT_OUTPUT_SLUG}_${CHECKPOINT_TAG}_whole_min${MIN_SIZE_TEST}_max${MAX_SIZE_TEST}_gpu${GPU_IDS//,/_}}"

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
if [[ "${EXPECTED_SAMPLES}" != "${RSKT_EXPECTED_SAMPLES}" ]] && is_truthy "${STRICT_PROTOCOL}"; then
  echo "Strict ${RSKT_DATASET_NAME} evaluation requires EXPECTED_SAMPLES=${RSKT_EXPECTED_SAMPLES}." >&2
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
  "${ROOT_DIR}/baselines/rskt_seg/${RSKT_EVALUATOR}"
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
  --expected_samples "${EXPECTED_SAMPLES}"
  --mask_id_base "${MASK_ID_BASE}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] RSKT-Seg ${RSKT_DATASET_NAME} evaluation"
echo "[$(date)] dataset_key=${RSKT_DATASET_KEY} | data_root=${DATA_ROOT} | expected_samples=${RSKT_EXPECTED_SAMPLES} | strict=${STRICT_PROTOCOL}"
echo "[$(date)] classes=${RSKT_CLASS_DESCRIPTION} | extra_support_class=false"
echo "[$(date)] taxonomy=${RSKT_TAXONOMY_DESCRIPTION}"
echo "[$(date)] inference=whole_image_shortest_edge | min=${MIN_SIZE_TEST} | max=${MAX_SIZE_TEST} | output=original"
echo "[$(date)] sliding_window=false | pooling=[1,1] | prompt=${PROMPT_ENSEMBLE} | amp=${AMP}"
echo "[$(date)] metrics=${RSKT_METRIC_DESCRIPTION}"
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
