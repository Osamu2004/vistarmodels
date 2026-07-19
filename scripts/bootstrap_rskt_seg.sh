#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RSKT_REPO_URL="${RSKT_REPO_URL:-https://github.com/LiBingyu01/RSKT-Seg.git}"
RSKT_ROOT="${RSKT_ROOT:-${ROOT_DIR}/third_party/RSKT-Seg}"
RSKT_WEIGHT_ROOT="${RSKT_WEIGHT_ROOT:-/root/data/weight/rskt_seg}"
RSKT_PRETRAINED_DIR="${RSKT_PRETRAINED_DIR:-${RSKT_WEIGHT_ROOT}/pretrained}"
RSKT_CHECKPOINT="${RSKT_CHECKPOINT:-/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth}"
RSKT_DOWNLOAD_AUX_WEIGHTS="${RSKT_DOWNLOAD_AUX_WEIGHTS:-1}"
RSKT_INSTALL_DETECTRON2="${RSKT_INSTALL_DETECTRON2:-0}"
RSKT_CHECKPOINT_URL="${RSKT_CHECKPOINT_URL:-}"

CLIP_VITL_URL="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
CLIP_VITB_URL="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
RSIB_GOOGLE_DRIVE_ID="1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_curl() {
  local url="$1"
  local destination="$2"
  if [[ -s "${destination}" ]]; then
    echo "[bootstrap_rskt_seg] exists: ${destination}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_rskt_seg] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${destination}" "${url}"
}

if [[ ! -f "${RSKT_ROOT}/RSKT_Seg/RSKT_Seg.py" ]]; then
  mkdir -p "$(dirname "${RSKT_ROOT}")"
  echo "[bootstrap_rskt_seg] cloning official source ${RSKT_REPO_URL} -> ${RSKT_ROOT}"
  git clone --depth 1 "${RSKT_REPO_URL}" "${RSKT_ROOT}"
else
  echo "[bootstrap_rskt_seg] using official source: ${RSKT_ROOT}"
fi

if [[ ! -f "${RSKT_ROOT}/detectron2/setup.py" ]]; then
  if [[ ! -f "${RSKT_ROOT}/detectron2.zip" ]]; then
    echo "[bootstrap_rskt_seg] official detectron2.zip is missing under ${RSKT_ROOT}" >&2
    exit 1
  fi
  echo "[bootstrap_rskt_seg] extracting official detectron2 source"
  (
    cd "${RSKT_ROOT}"
    unzip -q detectron2.zip
  )
fi

if is_truthy "${RSKT_DOWNLOAD_AUX_WEIGHTS}"; then
  mkdir -p "${RSKT_PRETRAINED_DIR}"
  download_curl "${CLIP_VITL_URL}" "${RSKT_PRETRAINED_DIR}/ViT-L-14-336px.pt"
  download_curl "${CLIP_VITB_URL}" "${RSKT_PRETRAINED_DIR}/ViT-B-32.pt"

  if [[ ! -s "${RSKT_PRETRAINED_DIR}/RemoteCLIP-ViT-B-32.pt" ]]; then
    RSKT_PRETRAINED_DIR="${RSKT_PRETRAINED_DIR}" "${PYTHON_BIN}" -c '
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="chendelong/RemoteCLIP",
    filename="RemoteCLIP-ViT-B-32.pt",
    local_dir=os.environ["RSKT_PRETRAINED_DIR"],
)
'
  fi

  if [[ ! -s "${RSKT_PRETRAINED_DIR}/RSIB.pth" ]]; then
    if ! "${PYTHON_BIN}" -c "import gdown" >/dev/null 2>&1; then
      echo "[bootstrap_rskt_seg] gdown is required for RSIB.pth." >&2
      echo "Install requirements-rskt-seg.txt, then rerun with RSKT_DOWNLOAD_AUX_WEIGHTS=1." >&2
      exit 1
    fi
    "${PYTHON_BIN}" -m gdown \
      "https://drive.google.com/uc?id=${RSIB_GOOGLE_DRIVE_ID}" \
      --output "${RSKT_PRETRAINED_DIR}/RSIB.pth"
  fi
else
  echo "[bootstrap_rskt_seg] auxiliary weight download disabled by RSKT_DOWNLOAD_AUX_WEIGHTS=0"
fi

if [[ -n "${RSKT_CHECKPOINT_URL}" && ! -s "${RSKT_CHECKPOINT}" ]]; then
  download_curl "${RSKT_CHECKPOINT_URL}" "${RSKT_CHECKPOINT}"
fi

if is_truthy "${RSKT_INSTALL_DETECTRON2}"; then
  echo "[bootstrap_rskt_seg] installing official bundled Detectron2"
  "${PYTHON_BIN}" -m pip install -e "${RSKT_ROOT}/detectron2"
else
  echo "[bootstrap_rskt_seg] Detectron2 installation disabled"
fi

if [[ ! -s "${RSKT_CHECKPOINT}" ]]; then
  echo "[bootstrap_rskt_seg] MANUAL DOWNLOAD REQUIRED: official DLRSD+ViT-L checkpoint"
  echo "[bootstrap_rskt_seg] Hugging Face does not currently host an official copy."
  echo "[bootstrap_rskt_seg] download model_final.pth from either official folder:"
  echo "  Baidu (password USTC):"
  echo "    https://pan.baidu.com/s/1xX6TBLAn3Xypsq-IZI3azw?pwd=USTC"
  echo "  OneDrive:"
  echo "    https://1drv.ms/f/c/69a773fee5342110/EnsFZEJptAlHgHmyPUkdoksBN-SUP9JPdu-VC_ePsCLEtg?e=cEX1oC"
  echo "[bootstrap_rskt_seg] place it under /root/data/weight/RSKT-Seg-ckpt"
  echo "[bootstrap_rskt_seg] or at the resolved path:"
  echo "  ${RSKT_CHECKPOINT}"
  echo "[bootstrap_rskt_seg] or set RSKT_CHECKPOINT_URL to a direct file URL."
fi

echo "[bootstrap_rskt_seg] done"
