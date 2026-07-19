#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GSNET_REPO_URL="${GSNET_REPO_URL:-https://github.com/yecy749/GSNet.git}"
GSNET_COMMIT="${GSNET_COMMIT:-61da3017529a99f8ae1bad5d423e62e2c7484e36}"
GSNET_ROOT="${GSNET_ROOT:-${ROOT_DIR}/third_party/GSNet}"
GSNET_WEIGHT_ROOT="${GSNET_WEIGHT_ROOT:-/root/data/weight/gsnet}"
GSNET_CHECKPOINT="${GSNET_CHECKPOINT:-${GSNET_WEIGHT_ROOT}/GSNet_base.pth}"
GSNET_CLIP_VITB="${GSNET_CLIP_VITB:-${GSNET_WEIGHT_ROOT}/pretrained/ViT-B-16.pt}"
GSNET_RSIB="${GSNET_RSIB:-/root/data/weight/rsib/RSIB.pth}"
GSNET_DOWNLOAD_WEIGHTS="${GSNET_DOWNLOAD_WEIGHTS:-1}"
GSNET_INSTALL_DETECTRON2="${GSNET_INSTALL_DETECTRON2:-0}"
GSNET_DETECTRON2_ROOT="${GSNET_DETECTRON2_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg/detectron2}"

GSNET_GOOGLE_DRIVE_ID="1YMAZj5fMUI3uSCvUmGHzyf4LthXdji0Y"
RSIB_GOOGLE_DRIVE_ID="1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV"
CLIP_VITB_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_VITB_URL="https://openaipublic.azureedge.net/clip/models/${CLIP_VITB_SHA256}/ViT-B-16.pt"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

validate_torch_checkpoint() {
  local checkpoint="$1"
  "${PYTHON_BIN}" - "${checkpoint}" >/dev/null 2>&1 <<'PY'
import sys

import torch

path = sys.argv[1]
try:
    value = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    value = torch.load(path, map_location="cpu")
if value is None:
    raise RuntimeError("checkpoint decoded to None")
PY
}

download_curl_sha256() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local partial="${destination}.part"
  if [[ -s "${destination}" ]] && \
    printf '%s  %s\n' "${expected_sha}" "${destination}" | shasum -a 256 -c - >/dev/null 2>&1; then
    echo "[bootstrap_gsnet] exists and checksum matches: ${destination}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_gsnet] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${partial}" "${url}"
  printf '%s  %s\n' "${expected_sha}" "${partial}" | shasum -a 256 -c -
  mv -f "${partial}" "${destination}"
}

download_gdrive_checkpoint() {
  local file_id="$1"
  local destination="$2"
  local partial="${destination}.part"
  local attempt

  if [[ -s "${destination}" ]]; then
    echo "[bootstrap_gsnet] using existing checkpoint: ${destination}"
    return
  fi
  if ! "${PYTHON_BIN}" -c "import gdown" >/dev/null 2>&1; then
    echo "[bootstrap_gsnet] gdown is required for Google Drive weights." >&2
    echo "Install requirements-gsnet.txt and rerun." >&2
    return 1
  fi

  mkdir -p "$(dirname "${destination}")"
  for attempt in 1 2 3; do
    echo "[bootstrap_gsnet] downloading checkpoint (attempt ${attempt}/3) -> ${destination}"
    if "${PYTHON_BIN}" -m gdown \
      --continue \
      "https://drive.google.com/uc?id=${file_id}" \
      --output "${partial}"; then
      if validate_torch_checkpoint "${partial}"; then
        mv -f "${partial}" "${destination}"
        echo "[bootstrap_gsnet] downloaded and validated: ${destination}"
        return
      fi
      echo "[bootstrap_gsnet] downloaded file is not a valid torch checkpoint" >&2
      rm -f "${partial}"
    fi
  done
  echo "[bootstrap_gsnet] automatic download failed: ${destination}" >&2
  return 1
}

if [[ ! -f "${GSNET_ROOT}/gs_net/GSNet.py" ]]; then
  mkdir -p "$(dirname "${GSNET_ROOT}")"
  echo "[bootstrap_gsnet] cloning official source -> ${GSNET_ROOT}"
  git clone --depth 1 "${GSNET_REPO_URL}" "${GSNET_ROOT}"
  if ! git -C "${GSNET_ROOT}" cat-file -e "${GSNET_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${GSNET_ROOT}" fetch --depth 1 origin "${GSNET_COMMIT}"
  fi
  git -C "${GSNET_ROOT}" checkout --detach "${GSNET_COMMIT}"
else
  echo "[bootstrap_gsnet] using official source: ${GSNET_ROOT}"
  if [[ -d "${GSNET_ROOT}/.git" ]]; then
    CURRENT_COMMIT="$(git -C "${GSNET_ROOT}" rev-parse HEAD)"
    if [[ "${CURRENT_COMMIT}" != "${GSNET_COMMIT}" ]]; then
      echo "[bootstrap_gsnet] warning: source revision ${CURRENT_COMMIT} differs from pinned ${GSNET_COMMIT}" >&2
    fi
  fi
fi

if is_truthy "${GSNET_DOWNLOAD_WEIGHTS}"; then
  download_curl_sha256 \
    "${CLIP_VITB_URL}" \
    "${GSNET_CLIP_VITB}" \
    "${CLIP_VITB_SHA256}"

  if [[ ! -s "${GSNET_CHECKPOINT}" ]]; then
    if ! download_gdrive_checkpoint \
      "${GSNET_GOOGLE_DRIVE_ID}" \
      "${GSNET_CHECKPOINT}"; then
      echo "[bootstrap_gsnet] manually download GSNet_base.pth from:" >&2
      echo "  https://drive.google.com/file/d/${GSNET_GOOGLE_DRIVE_ID}/view" >&2
      echo "and place it at ${GSNET_CHECKPOINT}" >&2
      exit 1
    fi
  else
    echo "[bootstrap_gsnet] using existing GSNet checkpoint: ${GSNET_CHECKPOINT}"
  fi

  if [[ ! -s "${GSNET_RSIB}" ]]; then
    if ! download_gdrive_checkpoint \
      "${RSIB_GOOGLE_DRIVE_ID}" \
      "${GSNET_RSIB}"; then
      echo "[bootstrap_gsnet] manually download RSIB.pth from:" >&2
      echo "  https://drive.google.com/file/d/${RSIB_GOOGLE_DRIVE_ID}/view" >&2
      echo "and place it at ${GSNET_RSIB}" >&2
      exit 1
    fi
  else
    echo "[bootstrap_gsnet] using existing RSIB checkpoint: ${GSNET_RSIB}"
  fi
else
  echo "[bootstrap_gsnet] weight download disabled by GSNET_DOWNLOAD_WEIGHTS=0"
fi

if is_truthy "${GSNET_INSTALL_DETECTRON2}"; then
  if [[ ! -f "${GSNET_DETECTRON2_ROOT}/setup.py" ]]; then
    echo "[bootstrap_gsnet] Detectron2 source not found: ${GSNET_DETECTRON2_ROOT}" >&2
    echo "Run scripts/bootstrap_rskt_seg.sh first or set GSNET_DETECTRON2_ROOT." >&2
    exit 1
  fi
  echo "[bootstrap_gsnet] installing Detectron2 without build isolation"
  MAX_JOBS="${MAX_JOBS:-8}" "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    -e "${GSNET_DETECTRON2_ROOT}"
else
  echo "[bootstrap_gsnet] Detectron2 installation disabled"
fi

echo "[bootstrap_gsnet] done"
