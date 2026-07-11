#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLUX1_FILL_MODEL_ID="${FLUX1_FILL_MODEL_ID:-black-forest-labs/FLUX.1-Fill-dev}"
FLUX1_FILL_WEIGHT_ROOT="${FLUX1_FILL_WEIGHT_ROOT:-/root/data/weight/flux1_fill}"
FLUX1_FILL_MODEL_DIR="${FLUX1_FILL_MODEL_DIR:-${FLUX1_FILL_WEIGHT_ROOT}/FLUX.1-Fill-dev}"
FLUX1_FILL_DOWNLOAD_WEIGHTS="${FLUX1_FILL_DOWNLOAD_WEIGHTS:-1}"

case "$(printf '%s' "${FLUX1_FILL_DOWNLOAD_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "[bootstrap_flux1_fill] downloading/resuming ${FLUX1_FILL_MODEL_ID} -> ${FLUX1_FILL_MODEL_DIR}"
    echo "[bootstrap_flux1_fill] accept the FLUX.1 Fill-dev Hugging Face license first; set HF_TOKEN or run huggingface-cli login."
    FLUX1_FILL_MODEL_ID="${FLUX1_FILL_MODEL_ID}" FLUX1_FILL_MODEL_DIR="${FLUX1_FILL_MODEL_DIR}" \
      "${PYTHON_BIN}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["FLUX1_FILL_MODEL_ID"], local_dir=os.environ["FLUX1_FILL_MODEL_DIR"], token=os.environ.get("HF_TOKEN"))'
    ;;
  *) echo "[bootstrap_flux1_fill] weight download disabled" ;;
esac

echo "[bootstrap_flux1_fill] done"
