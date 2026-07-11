#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTRUCTPIX2PIX_MODEL_ID="${INSTRUCTPIX2PIX_MODEL_ID:-timbrooks/instruct-pix2pix}"
INSTRUCTPIX2PIX_WEIGHT_ROOT="${INSTRUCTPIX2PIX_WEIGHT_ROOT:-/root/data/weight/instructpix2pix}"
INSTRUCTPIX2PIX_MODEL_DIR="${INSTRUCTPIX2PIX_MODEL_DIR:-${INSTRUCTPIX2PIX_WEIGHT_ROOT}/instruct-pix2pix}"
INSTRUCTPIX2PIX_DOWNLOAD_WEIGHTS="${INSTRUCTPIX2PIX_DOWNLOAD_WEIGHTS:-1}"

case "$(printf '%s' "${INSTRUCTPIX2PIX_DOWNLOAD_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "[bootstrap_instructpix2pix] downloading/resuming ${INSTRUCTPIX2PIX_MODEL_ID} -> ${INSTRUCTPIX2PIX_MODEL_DIR}"
    INSTRUCTPIX2PIX_MODEL_ID="${INSTRUCTPIX2PIX_MODEL_ID}" \
    INSTRUCTPIX2PIX_MODEL_DIR="${INSTRUCTPIX2PIX_MODEL_DIR}" \
      "${PYTHON_BIN}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["INSTRUCTPIX2PIX_MODEL_ID"], local_dir=os.environ["INSTRUCTPIX2PIX_MODEL_DIR"])'
    ;;
  *) echo "[bootstrap_instructpix2pix] weight download disabled" ;;
esac

echo "[bootstrap_instructpix2pix] done"
