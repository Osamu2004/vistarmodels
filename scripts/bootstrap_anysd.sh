#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANYSD_REPO_URL="${ANYSD_REPO_URL:-https://github.com/weichow23/AnySD.git}"
ANYSD_ROOT="${ANYSD_ROOT:-${ROOT_DIR}/third_party/AnySD}"
ANYSD_MODEL_ID="${ANYSD_MODEL_ID:-WeiChow/AnySD}"
ANYSD_WEIGHT_ROOT="${ANYSD_WEIGHT_ROOT:-/root/data/weight/anysd}"
ANYSD_MODEL_DIR="${ANYSD_MODEL_DIR:-${ANYSD_WEIGHT_ROOT}/AnySD}"
ANYSD_BASE_MODEL_ID="${ANYSD_BASE_MODEL_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
ANYSD_BASE_MODEL_DIR="${ANYSD_BASE_MODEL_DIR:-${ANYSD_WEIGHT_ROOT}/stable-diffusion-v1-5}"
ANYSD_DOWNLOAD_WEIGHTS="${ANYSD_DOWNLOAD_WEIGHTS:-1}"
ANYSD_DOWNLOAD_ALL_EXPERTS="${ANYSD_DOWNLOAD_ALL_EXPERTS:-0}"

if [[ ! -f "${ANYSD_ROOT}/anysd/src/model.py" ]]; then
  mkdir -p "$(dirname "${ANYSD_ROOT}")"
  echo "[bootstrap_anysd] cloning official source ${ANYSD_REPO_URL} -> ${ANYSD_ROOT}"
  git clone --depth 1 "${ANYSD_REPO_URL}" "${ANYSD_ROOT}"
else
  echo "[bootstrap_anysd] using official source: ${ANYSD_ROOT}"
fi

case "$(printf '%s' "${ANYSD_DOWNLOAD_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "[bootstrap_anysd] downloading/resuming ${ANYSD_MODEL_ID} -> ${ANYSD_MODEL_DIR}"
    ANYSD_MODEL_ID="${ANYSD_MODEL_ID}" ANYSD_MODEL_DIR="${ANYSD_MODEL_DIR}" \
    ANYSD_DOWNLOAD_ALL_EXPERTS="${ANYSD_DOWNLOAD_ALL_EXPERTS}" \
      "${PYTHON_BIN}" -c '
import os
from huggingface_hub import snapshot_download

patterns = [
    "README.md", "unet/*", "image_encoder/*",
    "experts/task_embs.bin", "experts/visual_seg.bin",
]
if os.environ.get("ANYSD_DOWNLOAD_ALL_EXPERTS", "0").lower() in {"1", "true", "yes", "on"}:
    patterns = ["README.md", "unet/*", "image_encoder/*", "experts/*"]
snapshot_download(
    repo_id=os.environ["ANYSD_MODEL_ID"],
    local_dir=os.environ["ANYSD_MODEL_DIR"],
    allow_patterns=patterns,
)
'
    echo "[bootstrap_anysd] downloading/resuming ${ANYSD_BASE_MODEL_ID} -> ${ANYSD_BASE_MODEL_DIR}"
    ANYSD_BASE_MODEL_ID="${ANYSD_BASE_MODEL_ID}" ANYSD_BASE_MODEL_DIR="${ANYSD_BASE_MODEL_DIR}" \
      "${PYTHON_BIN}" -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["ANYSD_BASE_MODEL_ID"],
    local_dir=os.environ["ANYSD_BASE_MODEL_DIR"],
    allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*",
        "text_encoder/*", "vae/*", "feature_extractor/*",
    ],
)
'
    ;;
  *) echo "[bootstrap_anysd] weight download disabled" ;;
esac

echo "[bootstrap_anysd] done"
