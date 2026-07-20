#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DYNAMIC_EARTH_ROOT="${DYNAMIC_EARTH_ROOT:-${ROOT_DIR}/third_party/DynamicEarth}"
DYNAMIC_EARTH_REPO_URL="${DYNAMIC_EARTH_REPO_URL:-https://github.com/likyoo/DynamicEarth.git}"
DYNAMIC_EARTH_COMMIT="${DYNAMIC_EARTH_COMMIT:-c9ffd90cafbd791cd75a48a5717a902966c2436c}"
DYNAMIC_EARTH_WEIGHT_ROOT="${DYNAMIC_EARTH_WEIGHT_ROOT:-/root/data/weight/dynamicearth}"
DYNAMIC_EARTH_SAM_CHECKPOINT="${DYNAMIC_EARTH_SAM_CHECKPOINT:-${DYNAMIC_EARTH_WEIGHT_ROOT}/sam_vit_h_4b8939.pth}"
DYNAMIC_EARTH_SEGEARTH_WEIGHT="${DYNAMIC_EARTH_SEGEARTH_WEIGHT:-${DYNAMIC_EARTH_WEIGHT_ROOT}/xclip_jbu_one_million_aid.ckpt}"
DYNAMIC_EARTH_DOWNLOAD_WEIGHTS="${DYNAMIC_EARTH_DOWNLOAD_WEIGHTS:-1}"
DYNAMIC_EARTH_BUILD_EXTENSIONS="${DYNAMIC_EARTH_BUILD_EXTENSIONS:-1}"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SEGEARTH_URL="https://raw.githubusercontent.com/likyoo/SegEarth-OV/main/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt"
SEGEARTH_SHA256="cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_resumable() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_dynamicearth] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${partial}" "${url}"
  mv -f "${partial}" "${destination}"
}

if [[ ! -f "${DYNAMIC_EARTH_ROOT}/dynamic_earth/comparator/bi_match.py" ]]; then
  mkdir -p "$(dirname "${DYNAMIC_EARTH_ROOT}")"
  echo "[bootstrap_dynamicearth] cloning official source -> ${DYNAMIC_EARTH_ROOT}"
  git clone --depth 1 "${DYNAMIC_EARTH_REPO_URL}" "${DYNAMIC_EARTH_ROOT}"
fi

if [[ -d "${DYNAMIC_EARTH_ROOT}/.git" ]]; then
  if ! git -C "${DYNAMIC_EARTH_ROOT}" cat-file -e "${DYNAMIC_EARTH_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${DYNAMIC_EARTH_ROOT}" fetch --depth 1 origin "${DYNAMIC_EARTH_COMMIT}"
  fi
  git -C "${DYNAMIC_EARTH_ROOT}" checkout --detach "${DYNAMIC_EARTH_COMMIT}"
else
  echo "[bootstrap_dynamicearth] using source without Git metadata: ${DYNAMIC_EARTH_ROOT}"
fi

if is_truthy "${DYNAMIC_EARTH_DOWNLOAD_WEIGHTS}"; then
  if [[ ! -s "${DYNAMIC_EARTH_SAM_CHECKPOINT}" ]]; then
    download_resumable "${SAM_URL}" "${DYNAMIC_EARTH_SAM_CHECKPOINT}"
  else
    echo "[bootstrap_dynamicearth] using SAM checkpoint: ${DYNAMIC_EARTH_SAM_CHECKPOINT}"
  fi
  SAM_SIZE="$(wc -c < "${DYNAMIC_EARTH_SAM_CHECKPOINT}")"
  if (( SAM_SIZE < 1000000000 )); then
    echo "[bootstrap_dynamicearth] SAM checkpoint is unexpectedly small: ${SAM_SIZE} bytes" >&2
    exit 1
  fi

  if [[ ! -s "${DYNAMIC_EARTH_SEGEARTH_WEIGHT}" ]] || ! \
    printf '%s  %s\n' "${SEGEARTH_SHA256}" "${DYNAMIC_EARTH_SEGEARTH_WEIGHT}" | \
      shasum -a 256 -c - >/dev/null 2>&1; then
    download_resumable "${SEGEARTH_URL}" "${DYNAMIC_EARTH_SEGEARTH_WEIGHT}"
  fi
  printf '%s  %s\n' "${SEGEARTH_SHA256}" "${DYNAMIC_EARTH_SEGEARTH_WEIGHT}" | \
    shasum -a 256 -c -
else
  echo "[bootstrap_dynamicearth] weight download disabled"
fi

if is_truthy "${DYNAMIC_EARTH_BUILD_EXTENSIONS}"; then
  echo "[bootstrap_dynamicearth] installing official SAM package"
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e \
    "${DYNAMIC_EARTH_ROOT}/third_party/segment_anything"
  echo "[bootstrap_dynamicearth] building SimFeatUp adaptive-convolution extensions"
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e \
    "${DYNAMIC_EARTH_ROOT}/third_party/SimFeatUp"
else
  echo "[bootstrap_dynamicearth] extension builds disabled"
fi

echo "[bootstrap_dynamicearth] source=${DYNAMIC_EARTH_ROOT}"
echo "[bootstrap_dynamicearth] sam=${DYNAMIC_EARTH_SAM_CHECKPOINT}"
echo "[bootstrap_dynamicearth] segearth=${DYNAMIC_EARTH_SEGEARTH_WEIGHT}"
echo "[bootstrap_dynamicearth] done"
