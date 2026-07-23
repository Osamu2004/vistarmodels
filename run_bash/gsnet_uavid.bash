#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29644}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-/root/data/OVSISBenchDataset/uavid}"
GSNET_ROOT="${GSNET_ROOT:-${ROOT_DIR}/third_party/GSNet}"
GSNET_WEIGHT_ROOT="${GSNET_WEIGHT_ROOT:-/root/data/weight/gsnet}"
GSNET_CONFIG="${GSNET_CONFIG:-${GSNET_ROOT}/configs/vitb_384.yaml}"
GSNET_CHECKPOINT="${GSNET_CHECKPOINT:-${GSNET_WEIGHT_ROOT}/GSNet_base.pth}"
GSNET_CLASS_JSON="${GSNET_CLASS_JSON:-${ROOT_DIR}/baselines/gsnet/configs/uavid_8_classes.json}"
GSNET_CLIP_VITB="${GSNET_CLIP_VITB:-${GSNET_WEIGHT_ROOT}/pretrained/ViT-B-16.pt}"
GSNET_RSIB="${GSNET_RSIB:-/root/data/weight/rsib/RSIB.pth}"

INPUT_SIZE="${INPUT_SIZE:-512}"
TILE_SIZE="${TILE_SIZE:-512}"
NUM_LAYERS="${NUM_LAYERS:-2}"
PROMPT_ENSEMBLE="${PROMPT_ENSEMBLE:-single}"
AMP="${AMP:-fp32}"
MASK_ID_BASE="${MASK_ID_BASE:-auto}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-270}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
COMPUTE_WFM="${COMPUTE_WFM:-1}"
OVERWRITE="${OVERWRITE:-0}"
STRICT_PROTOCOL="${STRICT_PROTOCOL:-1}"
BOOTSTRAP_GSNET="${BOOTSTRAP_GSNET:-1}"
GSNET_DOWNLOAD_WEIGHTS="${GSNET_DOWNLOAD_WEIGHTS:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/gsnet_uavid_ld50k_vistar8_tile${TILE_SIZE}_resize${INPUT_SIZE}_gpu${GPU_IDS//,/_}}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be positive, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi
for value_name in INPUT_SIZE TILE_SIZE NUM_LAYERS EXPECTED_SAMPLES; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be positive, got ${value}." >&2
    exit 2
  fi
done
if [[ "${MAX_SAMPLES}" != "0" ]] && ! [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be zero or positive, got ${MAX_SAMPLES}." >&2
  exit 2
fi
case "${MASK_ID_BASE}" in
  auto|zero|one) ;;
  *)
    echo "MASK_ID_BASE must be auto, zero, or one; got ${MASK_ID_BASE}." >&2
    exit 2
    ;;
esac

if is_truthy "${BOOTSTRAP_GSNET}"; then
  GSNET_ROOT="${GSNET_ROOT}" \
  GSNET_WEIGHT_ROOT="${GSNET_WEIGHT_ROOT}" \
  GSNET_CHECKPOINT="${GSNET_CHECKPOINT}" \
  GSNET_CLIP_VITB="${GSNET_CLIP_VITB}" \
  GSNET_RSIB="${GSNET_RSIB}" \
  GSNET_DOWNLOAD_WEIGHTS="${GSNET_DOWNLOAD_WEIGHTS}" \
    bash "${ROOT_DIR}/scripts/bootstrap_gsnet.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  GSNET_ROOT="${GSNET_ROOT}" \
  GSNET_CHECKPOINT="${GSNET_CHECKPOINT}" \
  GSNET_CLIP_VITB="${GSNET_CLIP_VITB}" \
  GSNET_RSIB="${GSNET_RSIB}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_gsnet_deps.py"
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
if is_truthy "${COMPUTE_WFM}"; then
  EXTRA_ARGS+=(--compute_wfm)
else
  EXTRA_ARGS+=(--no-compute_wfm)
fi
if is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi
if ! is_truthy "${STRICT_PROTOCOL}"; then
  EXTRA_ARGS+=(--no_strict_protocol)
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/gsnet/eval_gsnet_uavid.py"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --gsnet_root "${GSNET_ROOT}"
  --config "${GSNET_CONFIG}"
  --checkpoint "${GSNET_CHECKPOINT}"
  --class_json "${GSNET_CLASS_JSON}"
  --clip_vitb "${GSNET_CLIP_VITB}"
  --rsib "${GSNET_RSIB}"
  --input_size "${INPUT_SIZE}"
  --tile_size "${TILE_SIZE}"
  --num_layers "${NUM_LAYERS}"
  --prompt_ensemble "${PROMPT_ENSEMBLE}"
  --amp "${AMP}"
  --mask_id_base "${MASK_ID_BASE}"
  --expected_samples "${EXPECTED_SAMPLES}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] GSNet UAVid eight-class evaluation"
echo "[$(date)] setting=released LandDiscover50K checkpoint, cross-dataset/out-of-domain"
echo "[$(date)] paper_table_ovrsisbenchv2_comparable=false"
echo "[$(date)] data_root=${DATA_ROOT}"
echo "[$(date)] checkpoint=${GSNET_CHECKPOINT}"
echo "[$(date)] protocol=VISTAR UAVid all 270 images | classes=8 | background_clutter=evaluated class 0"
echo "[$(date)] taxonomy=no extra negative/background/unknown/support channel"
echo "[$(date)] inference=native non-overlap ${TILE_SIZE} tiles | model_input=${INPUT_SIZE} | encoder_internal=384"
echo "[$(date)] padding=zero_right_bottom | tile_overlap=0 | metric_size=original reassembled image"
echo "[$(date)] metrics=mIoU,mAcc,mF1,pixel_accuracy,foreground7,wfm_3px_percent"
echo "[$(date)] outputs=pred_mask,gt_mask,pred_rgb,gt_rgb,metrics.json,per_image_metrics.csv"
echo "[$(date)] prompt_ensemble=${PROMPT_ENSEMBLE} | num_layers=${NUM_LAYERS} | amp=${AMP}"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE} | synchronization=gloo"
echo "[$(date)] strict_protocol=${STRICT_PROTOCOL} | max_samples=${MAX_SAMPLES} | overwrite=${OVERWRITE}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
