#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[controlnet_second_oneclick] no Python interpreter found" >&2
    exit 2
  fi
fi

SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
BASE_MODEL="${BASE_MODEL:-/root/data/weight/stable-diffusion-v1-5}"
DATA_ROOT="${DATA_ROOT:-/root/data/experiment/controlnet_second_data}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/second/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42_smoke}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PER_GPU_BATCH="${PER_GPU_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-100000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"
RESOLUTION="${RESOLUTION:-256}"
SEED="${SEED:-42}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
PREPARE_DATA="${PREPARE_DATA:-auto}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "[controlnet_second_oneclick] repo=${ROOT_DIR}"
echo "[controlnet_second_oneclick] python=${PYTHON_BIN}"
echo "[controlnet_second_oneclick] second_root=${SECOND_ROOT}"
echo "[controlnet_second_oneclick] base_model=${BASE_MODEL}"
echo "[controlnet_second_oneclick] data_root=${DATA_ROOT}"
echo "[controlnet_second_oneclick] output=${OUTPUT_DIR}"
echo "[controlnet_second_oneclick] gpu_ids=${GPU_IDS} nproc=${NPROC_PER_NODE} per_gpu_batch=${PER_GPU_BATCH} grad_accum=${GRAD_ACCUM}"
echo "[controlnet_second_oneclick] steps=${MAX_STEPS} lr=${LEARNING_RATE} precision=${MIXED_PRECISION} backend=${DIST_BACKEND}"
echo "[controlnet_second_oneclick] condition=target-side directional mask + class-aware pre/post text"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -d "${SECOND_ROOT}" ]] || { echo "missing SECOND_ROOT: ${SECOND_ROOT}" >&2; exit 2; }
  [[ -d "${BASE_MODEL}" ]] || { echo "missing BASE_MODEL: ${BASE_MODEL}" >&2; exit 2; }
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[controlnet_second_oneclick] dry_run: ${PYTHON_BIN} -m pip install -r ${ROOT_DIR}/requirements-controlnet.txt"
  else
    "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-controlnet.txt"
  fi
fi

if [[ "${PREPARE_DATA}" == "1" || "${PREPARE_DATA}" == "true" || \
      ( "${PREPARE_DATA}" == "auto" && ! -f "${MANIFEST}" ) || "${REBUILD_MANIFEST}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[controlnet_second_oneclick] dry_run: prepare train,test online manifests"
  else
    SECOND_ROOT="${SECOND_ROOT}" DATA_ROOT="${DATA_ROOT}" SECOND_SPLITS=train,test \
      RESOLUTION="${RESOLUTION}" SECOND_STORAGE=online REBUILD_MANIFEST="${REBUILD_MANIFEST}" \
      PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/run_bash/controlnet_second_prepare.bash"
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_controlnet_deps.py" \
    --base_model "${BASE_MODEL}" --manifest "${MANIFEST}" --require_cuda
fi

COMMON_ENV=(
  PYTHON_BIN="${PYTHON_BIN}"
  BASE_MODEL="${BASE_MODEL}"
  MANIFEST="${MANIFEST}"
  GPU_IDS="${GPU_IDS}"
  NPROC_PER_NODE="${NPROC_PER_NODE}"
  PER_GPU_BATCH="${PER_GPU_BATCH}"
  GRAD_ACCUM="${GRAD_ACCUM}"
  LEARNING_RATE="${LEARNING_RATE}"
  MIXED_PRECISION="${MIXED_PRECISION}"
  DIST_BACKEND="${DIST_BACKEND}"
  RESOLUTION="${RESOLUTION}"
  SEED="${SEED}"
  DRY_RUN="${DRY_RUN}"
)

if [[ "${RUN_SMOKE}" == "1" ]]; then
  if [[ -f "${SMOKE_OUTPUT_DIR}/completed.json" ]] && \
     grep -q '"global_step": 3' "${SMOKE_OUTPUT_DIR}/completed.json"; then
    echo "[controlnet_second_oneclick] smoke/resume already complete: ${SMOKE_OUTPUT_DIR}"
  else
    echo "[controlnet_second_oneclick] smoke stage: optimizer steps 1--2"
    env "${COMMON_ENV[@]}" OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" MAX_STEPS=2 MAX_TRAIN_SAMPLES=32 \
      NUM_WORKERS=0 SAVE_EVERY=1 CHECKPOINT_LIMIT=3 LOG_EVERY=1 RESUME=none \
      bash "${ROOT_DIR}/run_bash/controlnet_second_train.bash"
    echo "[controlnet_second_oneclick] resume stage: restore step 2 and advance to step 3"
    env "${COMMON_ENV[@]}" OUTPUT_DIR="${SMOKE_OUTPUT_DIR}" MAX_STEPS=3 MAX_TRAIN_SAMPLES=32 \
      NUM_WORKERS=0 SAVE_EVERY=1 CHECKPOINT_LIMIT=3 LOG_EVERY=1 RESUME=auto \
      bash "${ROOT_DIR}/run_bash/controlnet_second_train.bash"
  fi
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  echo "[controlnet_second_oneclick] full stage: train/resume to ${MAX_STEPS} optimizer steps"
  env "${COMMON_ENV[@]}" OUTPUT_DIR="${OUTPUT_DIR}" MAX_STEPS="${MAX_STEPS}" \
    NUM_WORKERS="${NUM_WORKERS}" SAVE_EVERY="${SAVE_EVERY:-5000}" \
    CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-3}" LOG_EVERY="${LOG_EVERY:-20}" RESUME="${RESUME:-auto}" \
    bash "${ROOT_DIR}/run_bash/controlnet_second_train.bash"
fi
