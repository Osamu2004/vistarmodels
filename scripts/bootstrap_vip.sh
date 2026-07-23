#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIP_ROOT="${VIP_ROOT:-${ROOT_DIR}/third_party/VIP}"
VIP_REPO_URL="${VIP_REPO_URL:-https://github.com/MiSsU-HH/VIP.git}"
VIP_COMMIT="${VIP_COMMIT:-5bd25ee03ec25c1538622cf7da661e8c0461e769}"
VIP_WEIGHT_ROOT="${VIP_WEIGHT_ROOT:-/root/data/weight/vip}"
VIP_BACKBONE="${VIP_BACKBONE:-${VIP_WEIGHT_ROOT}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
VIP_DINOTXT="${VIP_DINOTXT:-${VIP_WEIGHT_ROOT}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth}"
VIP_BPE="${VIP_BPE:-${VIP_WEIGHT_ROOT}/bpe_simple_vocab_16e6.txt.gz}"
VIP_DOWNLOAD_WEIGHTS="${VIP_DOWNLOAD_WEIGHTS:-1}"

BACKBONE_URL="${VIP_BACKBONE_URL:-https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
DINOTXT_URL="${VIP_DINOTXT_URL:-https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth}"
BPE_URL="https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
BPE_SHA256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

clone_and_pin() {
  if [[ ! -f "${VIP_ROOT}/dinosegmentor.py" ]]; then
    if [[ -e "${VIP_ROOT}" ]]; then
      echo "[bootstrap_vip] incomplete source directory exists: ${VIP_ROOT}" >&2
      echo "Move it aside or set VIP_ROOT to another path." >&2
      exit 1
    fi
    mkdir -p "$(dirname "${VIP_ROOT}")"
    echo "[bootstrap_vip] cloning ${VIP_REPO_URL} -> ${VIP_ROOT}"
    git clone --depth 1 "${VIP_REPO_URL}" "${VIP_ROOT}"
  fi
  if [[ ! -d "${VIP_ROOT}/.git" ]]; then
    echo "[bootstrap_vip] VIP source lacks Git metadata: ${VIP_ROOT}" >&2
    exit 1
  fi
  if ! git -C "${VIP_ROOT}" cat-file -e "${VIP_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${VIP_ROOT}" fetch --depth 1 origin "${VIP_COMMIT}"
  fi
  git -C "${VIP_ROOT}" checkout --detach "${VIP_COMMIT}"
}

sha256_of() {
  shasum -a 256 "$1" | awk '{print $1}'
}

validate_hash_prefix() {
  local path="$1"
  local expected_prefix="$2"
  local actual
  [[ -s "${path}" ]] || return 1
  actual="$(sha256_of "${path}")"
  case "${actual}" in
    "${expected_prefix}"*) return 0 ;;
    *)
      echo "[bootstrap_vip] checksum mismatch: ${path}" >&2
      echo "[bootstrap_vip] expected sha256 prefix=${expected_prefix}, actual=${actual}" >&2
      return 1
      ;;
  esac
}

download_hash_prefix() {
  local url="$1"
  local destination="$2"
  local expected_prefix="$3"
  local partial="${destination}.part"
  if [[ -e "${destination}" ]]; then
    if validate_hash_prefix "${destination}" "${expected_prefix}"; then
      echo "[bootstrap_vip] exists and checksum matches: ${destination}"
      return
    fi
    echo "[bootstrap_vip] refusing to replace an invalid existing file: ${destination}" >&2
    echo "Move it aside manually, then rerun the bootstrap." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_vip] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${partial}" "${url}"
  validate_hash_prefix "${partial}" "${expected_prefix}"
  mv "${partial}" "${destination}"
}

download_exact_sha256() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local partial="${destination}.part"
  if [[ -e "${destination}" ]]; then
    if [[ -s "${destination}" ]] && \
      printf '%s  %s\n' "${expected_sha}" "${destination}" | shasum -a 256 -c - >/dev/null 2>&1; then
      echo "[bootstrap_vip] exists and checksum matches: ${destination}"
      return
    fi
    echo "[bootstrap_vip] refusing to replace an invalid existing file: ${destination}" >&2
    echo "Move it aside manually, then rerun the bootstrap." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${destination}")"
  echo "[bootstrap_vip] downloading ${url} -> ${destination}"
  curl -L --fail --retry 5 --continue-at - --output "${partial}" "${url}"
  printf '%s  %s\n' "${expected_sha}" "${partial}" | shasum -a 256 -c -
  gzip -t "${partial}"
  mv "${partial}" "${destination}"
}

clone_and_pin

if is_truthy "${VIP_DOWNLOAD_WEIGHTS}"; then
  echo "[bootstrap_vip] Meta may require approved DINOv3 access."
  echo "[bootstrap_vip] If a weight URL is rejected, place the approved files at the managed paths"
  echo "[bootstrap_vip] or pass VIP_BACKBONE_URL and VIP_DINOTXT_URL from Meta's access email."
  download_hash_prefix "${BACKBONE_URL}" "${VIP_BACKBONE}" "8aa4cbdd"
  download_hash_prefix "${DINOTXT_URL}" "${VIP_DINOTXT}" "a442d8f5"
  download_exact_sha256 "${BPE_URL}" "${VIP_BPE}" "${BPE_SHA256}"
else
  echo "[bootstrap_vip] managed-weight download disabled"
fi

echo "[bootstrap_vip] source=${VIP_ROOT}"
echo "[bootstrap_vip] source_revision=${VIP_COMMIT}"
echo "[bootstrap_vip] backbone=${VIP_BACKBONE}"
echo "[bootstrap_vip] dinotxt=${VIP_DINOTXT}"
echo "[bootstrap_vip] bpe=${VIP_BPE}"
echo "[bootstrap_vip] done"
