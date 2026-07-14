#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RSEDIT_MODEL_ID="${RSEDIT_MODEL_ID:-BiliSakura/RSEdit-UNet-text-ablation}"
RSEDIT_WEIGHT_ROOT="${RSEDIT_WEIGHT_ROOT:-/root/data/weight/rsedit}"
RSEDIT_REPO_DIR="${RSEDIT_REPO_DIR:-${RSEDIT_WEIGHT_ROOT}/RSEdit-UNet-text-ablation}"
RSEDIT_VARIANT="${RSEDIT_VARIANT:-DGTRS-CLIP-ViT-L-14}"
RSEDIT_MODEL_DIR="${RSEDIT_MODEL_DIR:-${RSEDIT_REPO_DIR}/${RSEDIT_VARIANT}}"
RSEDIT_DOWNLOAD_WEIGHTS="${RSEDIT_DOWNLOAD_WEIGHTS:-1}"

if [[ "${RSEDIT_DOWNLOAD_WEIGHTS}" == "1" ]]; then
  mkdir -p "${RSEDIT_REPO_DIR}"
  RSEDIT_MODEL_ID="${RSEDIT_MODEL_ID}" RSEDIT_REPO_DIR="${RSEDIT_REPO_DIR}" RSEDIT_VARIANT="${RSEDIT_VARIANT}" \
    "${PYTHON_BIN}" -c 'import os; from huggingface_hub import snapshot_download; v=os.environ["RSEDIT_VARIANT"]; patterns=[f"{v}/model_index.json", f"{v}/feature_extractor/*", f"{v}/scheduler/*", f"{v}/text_encoder/*", f"{v}/tokenizer/*", f"{v}/unet/*", f"{v}/vae/*"]; snapshot_download(repo_id=os.environ["RSEDIT_MODEL_ID"], local_dir=os.environ["RSEDIT_REPO_DIR"], allow_patterns=patterns)'
fi
REQUIRED_FILES=(
  model_index.json
  scheduler/scheduler_config.json
  tokenizer/tokenizer_config.json
  text_encoder/config.json
  text_encoder/model.safetensors
  unet/config.json
  unet/diffusion_pytorch_model.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
)
for relative in "${REQUIRED_FILES[@]}"; do
  if [[ ! -s "${RSEDIT_MODEL_DIR}/${relative}" ]]; then
    echo "[bootstrap_rsedit] missing/empty variant file: ${RSEDIT_MODEL_DIR}/${relative}" >&2
    exit 2
  fi
done
echo "[bootstrap_rsedit] model=${RSEDIT_MODEL_DIR}"
