#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:?set local SD 1.5 or SD 2.1 diffusers snapshot}"
OUTPUT_DIR="${OUTPUT_DIR:?set SSD checkpoint directory}"
DATA_DIR="${DATA_DIR:-/data/vistar/runs/paper_baselines/controlnet_second_train}"
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
"${PYTHON_BIN}" -m accelerate.commands.launch examples/controlnet/train_controlnet.py \
  --pretrained_model_name_or_path "${BASE_MODEL}" --dataset_name imagefolder --train_data_dir "${DATA_DIR}" \
  --image_column image --conditioning_image_column conditioning_image --caption_column text \
  --resolution 256 --learning_rate "${LEARNING_RATE:-1e-5}" --train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" --mixed_precision bf16 \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-5000}" --output_dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" "${@}"
