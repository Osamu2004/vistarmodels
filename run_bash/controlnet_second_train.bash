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
    echo "[controlnet_second_train] no Python interpreter found" >&2
    exit 2
  fi
fi
BASE_MODEL="${BASE_MODEL:-/root/data/weight/stable-diffusion-v1-5}"
MANIFEST="${MANIFEST:-/root/data/experiment/controlnet_second_data/second/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29641}"
PER_GPU_BATCH="${PER_GPU_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
RESOLUTION="${RESOLUTION:-256}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-3}"
LOG_EVERY="${LOG_EVERY:-20}"
RESUME="${RESUME:-auto}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

ARGS=(
  --manifest "${MANIFEST}"
  --base_model "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --resolution "${RESOLUTION}"
  --max_train_steps "${MAX_STEPS}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --batch_size "${PER_GPU_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --num_workers "${NUM_WORKERS}"
  --learning_rate "${LEARNING_RATE}"
  --mixed_precision "${MIXED_PRECISION}"
  --seed "${SEED}"
  --save_every "${SAVE_EVERY}"
  --checkpoint_limit "${CHECKPOINT_LIMIT}"
  --log_every "${LOG_EVERY}"
  --resume "${RESUME}"
  --dist_backend "${DIST_BACKEND}"
)
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then ARGS+=(--gradient_checkpointing); fi
if [[ "${ENABLE_XFORMERS:-0}" == "1" ]]; then ARGS+=(--enable_xformers); fi
if [[ "${ALLOW_TF32:-1}" == "1" ]]; then ARGS+=(--allow_tf32); fi
if [[ "${RANDOM_FLIP:-1}" != "1" ]]; then ARGS+=(--no_random_flip); fi
if [[ -n "${CONTROLNET_INIT:-}" ]]; then ARGS+=(--controlnet_init "${CONTROLNET_INIT}"); fi
if (( $# > 0 )); then
  ARGS+=("$@")
fi

echo "[controlnet_second_train] python=${PYTHON_BIN}"
echo "[controlnet_second_train] base_model=${BASE_MODEL}"
echo "[controlnet_second_train] manifest=${MANIFEST}"
echo "[controlnet_second_train] output=${OUTPUT_DIR}"
echo "[controlnet_second_train] gpu_ids=${GPU_IDS} nproc=${NPROC_PER_NODE} per_gpu_batch=${PER_GPU_BATCH} grad_accum=${GRAD_ACCUM}"
echo "[controlnet_second_train] steps=${MAX_STEPS} lr=${LEARNING_RATE} resolution=${RESOLUTION} precision=${MIXED_PRECISION}"
echo "[controlnet_second_train] resume=${RESUME} dist_backend=${DIST_BACKEND}"

TRAIN_SCRIPT="${ROOT_DIR}/baselines/controlnet/train_controlnet_second.py"
if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  COMMAND=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}")
else
  COMMAND=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" "${TRAIN_SCRIPT}" "${ARGS[@]}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[controlnet_second_train] dry_run:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/check_controlnet_deps.py" \
  --base_model "${BASE_MODEL}" --manifest "${MANIFEST}" --require_cuda
"${COMMAND[@]}"
