#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEGEARTH_OV_ROOT="${SEGEARTH_OV_ROOT:-${ROOT_DIR}/third_party/SegEarth-OV}"
SEGEARTH_OV_REPO_URL="${SEGEARTH_OV_REPO_URL:-https://github.com/likyoo/SegEarth-OV.git}"
SEGEARTH_OV_COMMIT="${SEGEARTH_OV_COMMIT:-3e22a969b32c6d751bdbba64a88a0b670e630f55}"
SIMFEATUP_ROOT="${SIMFEATUP_ROOT:-${ROOT_DIR}/third_party/SimFeatUp}"
SIMFEATUP_REPO_URL="${SIMFEATUP_REPO_URL:-https://github.com/likyoo/SimFeatUp.git}"
SIMFEATUP_COMMIT="${SIMFEATUP_COMMIT:-78a0ba70b1d6ea7283684a88c98ce338af4593ca}"
SEGEARTH_OV_WEIGHT_ROOT="${SEGEARTH_OV_WEIGHT_ROOT:-/root/data/weight/segearth_ov}"
SEGEARTH_OV_CLIP_VITB="${SEGEARTH_OV_CLIP_VITB:-${SEGEARTH_OV_WEIGHT_ROOT}/pretrained/ViT-B-16.pt}"
SEGEARTH_OV_SIMFEATUP="${SEGEARTH_OV_SIMFEATUP:-${SEGEARTH_OV_ROOT}/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt}"
SEGEARTH_OV_DOWNLOAD_WEIGHTS="${SEGEARTH_OV_DOWNLOAD_WEIGHTS:-1}"
SEGEARTH_OV_INSTALL_SIMFEATUP="${SEGEARTH_OV_INSTALL_SIMFEATUP:-1}"

CLIP_VITB_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_VITB_URL="https://openaipublic.azureedge.net/clip/models/${CLIP_VITB_SHA256}/ViT-B-16.pt"
SIMFEATUP_SHA256="cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

clone_and_pin() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  local sentinel="$4"
  if [[ ! -f "${destination}/${sentinel}" ]]; then
    if [[ -e "${destination}" ]]; then
      echo "[bootstrap_segearth_ov] incomplete source directory exists: ${destination}" >&2
      echo "Move it aside or set an alternate source-root variable." >&2
      exit 1
    fi
    mkdir -p "$(dirname "${destination}")"
    echo "[bootstrap_segearth_ov] cloning ${url} -> ${destination}"
    git clone --depth 1 "${url}" "${destination}"
  fi
  if [[ ! -d "${destination}/.git" ]]; then
    echo "[bootstrap_segearth_ov] source lacks Git metadata: ${destination}" >&2
    exit 1
  fi
  if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${destination}" fetch --depth 1 origin "${commit}"
  fi
  git -C "${destination}" checkout --detach "${commit}"
}

download_curl_sha256() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local partial="${destination}.part"
  if [[ -s "${destination}" ]] && \
    printf '%s  %s\n' "${expected_sha}" "${destination}" | shasum -a 256 -c - >/dev/null 2>&1; then
    echo "[bootstrap_segearth_ov] exists and checksum matches: ${destination}"
    return
  fi
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_segearth_ov] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${partial}" "${url}"
  printf '%s  %s\n' "${expected_sha}" "${partial}" | shasum -a 256 -c -
  mv -f "${partial}" "${destination}"
}

clone_and_pin \
  "${SEGEARTH_OV_REPO_URL}" \
  "${SEGEARTH_OV_ROOT}" \
  "${SEGEARTH_OV_COMMIT}" \
  "segearth_segmentor.py"
clone_and_pin \
  "${SIMFEATUP_REPO_URL}" \
  "${SIMFEATUP_ROOT}" \
  "${SIMFEATUP_COMMIT}" \
  "setup.py"

if [[ ! -s "${SEGEARTH_OV_SIMFEATUP}" ]]; then
  echo "[bootstrap_segearth_ov] missing official bundled SimFeatUp checkpoint: ${SEGEARTH_OV_SIMFEATUP}" >&2
  exit 1
fi
printf '%s  %s\n' "${SIMFEATUP_SHA256}" "${SEGEARTH_OV_SIMFEATUP}" | shasum -a 256 -c -

if is_truthy "${SEGEARTH_OV_DOWNLOAD_WEIGHTS}"; then
  download_curl_sha256 \
    "${CLIP_VITB_URL}" \
    "${SEGEARTH_OV_CLIP_VITB}" \
    "${CLIP_VITB_SHA256}"
else
  echo "[bootstrap_segearth_ov] CLIP download disabled"
fi

if is_truthy "${SEGEARTH_OV_INSTALL_SIMFEATUP}"; then
  echo "[bootstrap_segearth_ov] building SimFeatUp CUDA/C++ extensions"
  MAX_JOBS="${MAX_JOBS:-8}" "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    -e "${SIMFEATUP_ROOT}"
else
  echo "[bootstrap_segearth_ov] SimFeatUp installation disabled"
fi

echo "[bootstrap_segearth_ov] source=${SEGEARTH_OV_ROOT}"
echo "[bootstrap_segearth_ov] source_revision=${SEGEARTH_OV_COMMIT}"
echo "[bootstrap_segearth_ov] simfeatup_source=${SIMFEATUP_ROOT}"
echo "[bootstrap_segearth_ov] simfeatup_revision=${SIMFEATUP_COMMIT}"
echo "[bootstrap_segearth_ov] simfeatup_checkpoint=${SEGEARTH_OV_SIMFEATUP}"
echo "[bootstrap_segearth_ov] clip=${SEGEARTH_OV_CLIP_VITB}"
echo "[bootstrap_segearth_ov] done"
