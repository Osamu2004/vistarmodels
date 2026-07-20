#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29651}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/root/data/LEVIR-CD}"
SPLIT="${SPLIT:-test}"
RCDNET_ROOT="${RCDNET_ROOT:-${ROOT_DIR}/third_party/referring_change_detection}"
RCDNET_WEIGHT_ROOT="${RCDNET_WEIGHT_ROOT:-/root/data/weight/rcdnet}"
RCDNET_CHECKPOINT="${RCDNET_CHECKPOINT:-${RCDNET_WEIGHT_ROOT}/SECOND-model.safetensors}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
PROMPT="${PROMPT:-building}"
THRESHOLD="${THRESHOLD:-0.5}"
TILE_SIZE="${TILE_SIZE:-512}"
MODEL_INPUT_SIZE="${MODEL_INPUT_SIZE:-512}"
AMP="${AMP:-fp16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_RCDNET="${BOOTSTRAP_RCDNET:-1}"
RCDNET_DOWNLOAD_WEIGHTS="${RCDNET_DOWNLOAD_WEIGHTS:-1}"
RCDNET_BUILD_SELECTIVE_SCAN="${RCDNET_BUILD_SELECTIVE_SCAN:-1}"
CHECK_DEPS="${CHECK_DEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rcdnet_second_real_levircd_${SPLIT}_building_native_tile${TILE_SIZE}_input${MODEL_INPUT_SIZE}_thr${THRESHOLD}}"

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

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT_DIR}/baselines/rcdnet/eval_rcdnet_levircd.py"
  --data_root "${DATA_ROOT}"
  --split "${SPLIT}"
  --output_dir "${OUTPUT_DIR}"
  --rcdnet_root "${RCDNET_ROOT}"
  --checkpoint "${RCDNET_CHECKPOINT}"
  --clip_model "${CLIP_MODEL}"
  --prompt "${PROMPT}"
  --threshold "${THRESHOLD}"
  --tile_size "${TILE_SIZE}"
  --model_input_size "${MODEL_INPUT_SIZE}"
  --amp "${AMP}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "[$(date)] RCDNet LEVIR-CD inference"
echo "[$(date)] source=official referring_change_detection | checkpoint=SECOND real-data release"
echo "[$(date)] caveat=paper synthetic-pretrained checkpoint for 60.21 cross-domain IoU is not public"
echo "[$(date)] data_root=${DATA_ROOT} | split=${SPLIT}"
echo "[$(date)] prompt=${PROMPT} | threshold=${THRESHOLD}"
echo "[$(date)] native_tile=${TILE_SIZE} | model_input=${MODEL_INPUT_SIZE} | amp=${AMP}"
echo "[$(date)] GPUs=${GPU_IDS} | nproc=${NPROC_PER_NODE}"
echo "[$(date)] output=${OUTPUT_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[%s] command:' "$(date)"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if is_truthy "${BOOTSTRAP_RCDNET}"; then
  RCDNET_ROOT="${RCDNET_ROOT}" \
  RCDNET_WEIGHT_ROOT="${RCDNET_WEIGHT_ROOT}" \
  RCDNET_CHECKPOINT="${RCDNET_CHECKPOINT}" \
  RCDNET_DOWNLOAD_WEIGHTS="${RCDNET_DOWNLOAD_WEIGHTS}" \
  RCDNET_BUILD_SELECTIVE_SCAN="${RCDNET_BUILD_SELECTIVE_SCAN}" \
    bash "${ROOT_DIR}/scripts/bootstrap_rcdnet.sh"
fi

if is_truthy "${CHECK_DEPS}"; then
  RCDNET_ROOT="${RCDNET_ROOT}" \
  RCDNET_CHECKPOINT="${RCDNET_CHECKPOINT}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_rcdnet_deps.py"
fi

mkdir -p "${OUTPUT_DIR}"
exec "${CMD[@]}"
