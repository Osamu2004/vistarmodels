#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29641}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

DATA_ROOT="${DATA_ROOT:-/root/data/CHN6-CUG/val}"
GSNET_ROOT="${GSNET_ROOT:-${ROOT_DIR}/third_party/GSNet}"
GSNET_WEIGHT_ROOT="${GSNET_WEIGHT_ROOT:-/root/data/weight/gsnet}"
GSNET_CONFIG="${GSNET_CONFIG:-${GSNET_ROOT}/configs/vitb_384.yaml}"
GSNET_CHECKPOINT="${GSNET_CHECKPOINT:-${GSNET_WEIGHT_ROOT}/GSNet_base.pth}"
GSNET_CLASS_JSON="${GSNET_CLASS_JSON:-${ROOT_DIR}/baselines/gsnet/configs/chn6_cug_classes.json}"
GSNET_CLIP_VITB="${GSNET_CLIP_VITB:-${GSNET_WEIGHT_ROOT}/pretrained/ViT-B-16.pt}"
GSNET_RSIB="${GSNET_RSIB:-/root/data/weight/rsib/RSIB.pth}"

INPUT_SIZE="${INPUT_SIZE:-512}"
TILE_SIZE="${TILE_SIZE:-512}"
NUM_LAYERS="${NUM_LAYERS:-2}"
PROMPT_ENSEMBLE="${PROMPT_ENSEMBLE:-single}"
AMP="${AMP:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_GSNET="${BOOTSTRAP_GSNET:-1}"
GSNET_DOWNLOAD_WEIGHTS="${GSNET_DOWNLOAD_WEIGHTS:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/gsnet_chn6_cug_ld50k_crossdomain_tile${TILE_SIZE}_resize${INPUT_SIZE}_${NPROC_PER_NODE}gpu}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be positive, got ${NPROC_PER_NODE}." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NPROC_PER_NODE )); then
  echo "GPU_IDS=${GPU_IDS} exposes fewer devices than NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

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
if is_truthy "${OVERWRITE}"; then
  EXTRA_ARGS+=(--overwrite)
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/gsnet/eval_gsnet_binary.py"
  --dataset chn6_cug
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
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] GSNet CHN6-CUG road evaluation"
echo "[$(date)] setting=LandDiscover50K-trained cross-dataset/out-of-domain"
echo "[$(date)] data_root=${DATA_ROOT}"
echo "[$(date)] checkpoint=${GSNET_CHECKPOINT}"
echo "[$(date)] classes=background,road | primary_metric=road_iou"
echo "[$(date)] inference=native_nonoverlap_tiled | source_tile=${TILE_SIZE} | model_input=${INPUT_SIZE}"
echo "[$(date)] encoder_internal=384 | padding=zero_right_bottom | metric_size=original"
echo "[$(date)] prompt_ensemble=${PROMPT_ENSEMBLE} | amp=${AMP}"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE} | synchronization=gloo"
echo "[$(date)] bootstrap=${BOOTSTRAP_GSNET} | auto_download_weights=${GSNET_DOWNLOAD_WEIGHTS}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
