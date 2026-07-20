#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29652}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/root/data/LEVIR-CD}"
SPLIT="${SPLIT:-test}"
DYNAMIC_EARTH_ROOT="${DYNAMIC_EARTH_ROOT:-${ROOT_DIR}/third_party/DynamicEarth}"
DYNAMIC_EARTH_WEIGHT_ROOT="${DYNAMIC_EARTH_WEIGHT_ROOT:-/root/data/weight/dynamicearth}"
DYNAMIC_EARTH_SAM_CHECKPOINT="${DYNAMIC_EARTH_SAM_CHECKPOINT:-${DYNAMIC_EARTH_WEIGHT_ROOT}/sam_vit_h_4b8939.pth}"
DYNAMIC_EARTH_SEGEARTH_WEIGHT="${DYNAMIC_EARTH_SEGEARTH_WEIGHT:-${DYNAMIC_EARTH_WEIGHT_ROOT}/xclip_jbu_one_million_aid.ckpt}"
CHANGE_THRESHOLD="${CHANGE_THRESHOLD:-145}"
FEATURE_UP="${FEATURE_UP:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_DYNAMIC_EARTH="${BOOTSTRAP_DYNAMIC_EARTH:-1}"
DYNAMIC_EARTH_DOWNLOAD_WEIGHTS="${DYNAMIC_EARTH_DOWNLOAD_WEIGHTS:-1}"
DYNAMIC_EARTH_BUILD_EXTENSIONS="${DYNAMIC_EARTH_BUILD_EXTENSIONS:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dynamicearth_mci_sam_dino_segearth_levircd_${SPLIT}_fullres_thr${CHANGE_THRESHOLD}}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if is_truthy "${SAVE_IMAGES}"; then EXTRA_ARGS+=(--save_images); else EXTRA_ARGS+=(--no-save_images); fi
if is_truthy "${OVERWRITE}"; then EXTRA_ARGS+=(--overwrite); fi
if is_truthy "${FEATURE_UP}"; then EXTRA_ARGS+=(--feature_up); else EXTRA_ARGS+=(--no-feature_up); fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/dynamicearth/eval_dynamicearth_levircd.py"
  --data_root "${DATA_ROOT}"
  --split "${SPLIT}"
  --output_dir "${OUTPUT_DIR}"
  --dynamic_root "${DYNAMIC_EARTH_ROOT}"
  --sam_checkpoint "${DYNAMIC_EARTH_SAM_CHECKPOINT}"
  --segearth_weight "${DYNAMIC_EARTH_SEGEARTH_WEIGHT}"
  --change_threshold "${CHANGE_THRESHOLD}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] DynamicEarth LEVIR-CD inference"
echo "[$(date)] variant=M-C-I SAM-ViT-H + DINO-ViT-B/16 + SegEarth-OV CLIP-ViT-B/16"
echo "[$(date)] data_root=${DATA_ROOT} | split=${SPLIT} | native_full_resolution=1"
echo "[$(date)] change_angle_threshold=${CHANGE_THRESHOLD} | feature_up=${FEATURE_UP}"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if is_truthy "${BOOTSTRAP_DYNAMIC_EARTH}"; then
  DYNAMIC_EARTH_ROOT="${DYNAMIC_EARTH_ROOT}" \
  DYNAMIC_EARTH_WEIGHT_ROOT="${DYNAMIC_EARTH_WEIGHT_ROOT}" \
  DYNAMIC_EARTH_SAM_CHECKPOINT="${DYNAMIC_EARTH_SAM_CHECKPOINT}" \
  DYNAMIC_EARTH_SEGEARTH_WEIGHT="${DYNAMIC_EARTH_SEGEARTH_WEIGHT}" \
  DYNAMIC_EARTH_DOWNLOAD_WEIGHTS="${DYNAMIC_EARTH_DOWNLOAD_WEIGHTS}" \
  DYNAMIC_EARTH_BUILD_EXTENSIONS="${DYNAMIC_EARTH_BUILD_EXTENSIONS}" \
    bash "${ROOT_DIR}/scripts/bootstrap_dynamicearth.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  DYNAMIC_EARTH_ROOT="${DYNAMIC_EARTH_ROOT}" \
  DYNAMIC_EARTH_SAM_CHECKPOINT="${DYNAMIC_EARTH_SAM_CHECKPOINT}" \
  DYNAMIC_EARTH_SEGEARTH_WEIGHT="${DYNAMIC_EARTH_SEGEARTH_WEIGHT}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_dynamicearth_deps.py"
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
