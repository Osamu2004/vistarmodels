#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-/data/vistar/runs/paper_baselines/data/loveda/train/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/weights/ddpm_loveda_512}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
RESUME_ARGS=()
if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" ]]; then
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null; then
    RESUME_ARGS+=(--resume_from_checkpoint latest)
  fi
elif [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "none" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
cd "${ROOT_DIR}/third_party/diffusers"
export PYTHONPATH="${ROOT_DIR}/third_party/diffusers/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m accelerate.commands.launch examples/unconditional_image_generation/train_unconditional.py \
  --train_data_dir "${DATA_DIR}" --output_dir "${OUTPUT_DIR}" --resolution 512 \
  --train_batch_size "${BATCH_SIZE:-2}" --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --num_epochs "${NUM_EPOCHS:-200}" --learning_rate "${LEARNING_RATE:-1e-4}" \
  --mixed_precision bf16 --checkpointing_steps "${CHECKPOINTING_STEPS:-5000}" \
  --ddpm_num_inference_steps "${DDPM_INFERENCE_STEPS:-50}" --save_images_epochs "${SAVE_IMAGES_EPOCHS:-10}" \
  "${RESUME_ARGS[@]}" "${@}"
