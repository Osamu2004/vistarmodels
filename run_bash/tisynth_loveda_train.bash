#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TISYNTH_ROOT="${TISYNTH_ROOT:-${ROOT_DIR}/third_party/TISynth}"
TISYNTH_PRETRAIN="${TISYNTH_PRETRAIN:-/root/data/weight/tisynth/controlnet1.5.ckpt}"
TISYNTH_CLIP_VERSION="${TISYNTH_CLIP_VERSION:-openai/clip-vit-large-patch14}"

# These directories must have the same cond_mask/gt_rgb contract as Vistar.
# Keep training and validation sources disjoint.
TRAIN_SOURCE_DIR="${TRAIN_SOURCE_DIR:-}"
VAL_SOURCE_DIR="${VAL_SOURCE_DIR:-}"
LOG_DIR="${LOG_DIR:-/root/data/experiment/tisynth_loveda_train}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${LOG_DIR}/loveda_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${LOG_DIR}/loveda_val.jsonl}"
PATCHED_MODEL_CONFIG="${PATCHED_MODEL_CONFIG:-${LOG_DIR}/cldm_ssl_v15_aia_loveda.yaml}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/baselines/tisynth/tisynth_loveda_train.yaml}"

GPU_IDS="${GPU_IDS:-0,}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
SEED="${SEED:-23}"
RUN_NAME="${RUN_NAME:-loveda_512}"
RESUME="${RESUME:-}"
STRICT_PALETTE="${STRICT_PALETTE:-1}"
BOOTSTRAP_TISYNTH="${BOOTSTRAP_TISYNTH:-1}"

_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if _truthy "${BOOTSTRAP_TISYNTH}"; then
  TISYNTH_ROOT="${TISYNTH_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_tisynth.sh"
fi
if [[ -z "${TRAIN_SOURCE_DIR}" || -z "${VAL_SOURCE_DIR}" ]]; then
  echo "[tisynth_loveda_train] TRAIN_SOURCE_DIR and VAL_SOURCE_DIR are required and must be disjoint." >&2
  exit 1
fi
if [[ ! -d "${TRAIN_SOURCE_DIR}" || ! -d "${VAL_SOURCE_DIR}" ]]; then
  echo "[tisynth_loveda_train] Training or validation source directory does not exist." >&2
  exit 1
fi
if [[ "$(cd "${TRAIN_SOURCE_DIR}" && pwd)" == "$(cd "${VAL_SOURCE_DIR}" && pwd)" ]]; then
  echo "[tisynth_loveda_train] Refusing identical training and validation directories." >&2
  exit 1
fi
if [[ ! -f "${TISYNTH_PRETRAIN}" ]]; then
  echo "[tisynth_loveda_train] Missing official initialization checkpoint: ${TISYNTH_PRETRAIN}" >&2
  echo "Download TISynth_models.zip from the official Google Drive archive." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
MANIFEST_PALETTE_ARGS=()
if ! _truthy "${STRICT_PALETTE}"; then
  MANIFEST_PALETTE_ARGS+=(--no-strict_palette)
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_tisynth_loveda_train_manifest.py" \
  --source_dir "${TRAIN_SOURCE_DIR}" --output "${TRAIN_MANIFEST}" "${MANIFEST_PALETTE_ARGS[@]}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_tisynth_loveda_train_manifest.py" \
  --source_dir "${VAL_SOURCE_DIR}" --output "${VAL_MANIFEST}" "${MANIFEST_PALETTE_ARGS[@]}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/patch_tisynth_model_config.py" \
  --input "${TISYNTH_ROOT}/models/cldm_ssl_v15_aia_v0_augmentation.yaml" \
  --output "${PATCHED_MODEL_CONFIG}" \
  --clip_version "${TISYNTH_CLIP_VERSION}"

echo "[tisynth_loveda_train] train=${TRAIN_SOURCE_DIR} val=${VAL_SOURCE_DIR}"
echo "[tisynth_loveda_train] pretrain=${TISYNTH_PRETRAIN}"
echo "[tisynth_loveda_train] GPUs=${GPU_IDS} batch_size=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[tisynth_loveda_train] max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} seed=${SEED}"
echo "[tisynth_loveda_train] logs=${LOG_DIR}"

TRAIN_ARGS=(
  --base "${BASE_CONFIG}"
  -t
  --pretrain_path "${TISYNTH_PRETRAIN}"
  --config_model "${PATCHED_MODEL_CONFIG}"
  -l "${LOG_DIR}"
  --gpus "${GPU_IDS}"
  --max_steps "${MAX_STEPS}"
  --seed "${SEED}"
  --data_root /
  --train_txt_file "${TRAIN_MANIFEST}"
  --val_txt_file "${VAL_MANIFEST}"
  "data.params.batch_size=${BATCH_SIZE}"
  "data.params.num_workers=${NUM_WORKERS}"
  "lightning.modelcheckpoint.params.every_n_train_steps=${SAVE_EVERY}"
)
if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
else
  TRAIN_ARGS+=(-n "${RUN_NAME}")
fi

cd "${TISYNTH_ROOT}"
"${PYTHON_BIN}" main.py "${TRAIN_ARGS[@]}"
