#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RCDNET_ROOT="${RCDNET_ROOT:-${ROOT_DIR}/third_party/referring_change_detection}"
RCDNET_REPO_URL="${RCDNET_REPO_URL:-https://github.com/yilmazkorkmaz1/referring_change_detection.git}"
RCDNET_COMMIT="${RCDNET_COMMIT:-0966e96ff7075476d77442bbf6623ed5086d52da}"
RCDNET_WEIGHT_ROOT="${RCDNET_WEIGHT_ROOT:-/root/data/weight/rcdnet}"
RCDNET_CHECKPOINT="${RCDNET_CHECKPOINT:-${RCDNET_WEIGHT_ROOT}/SECOND-model.safetensors}"
RCDNET_DOWNLOAD_WEIGHTS="${RCDNET_DOWNLOAD_WEIGHTS:-1}"
RCDNET_BUILD_SELECTIVE_SCAN="${RCDNET_BUILD_SELECTIVE_SCAN:-1}"
RCDNET_SECOND_GDRIVE_ID="1yAs4tB3ScrH5oUcvQLlouBXRjz9CmiQ1"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

validate_safetensors() {
  local checkpoint="$1"
  "${PYTHON_BIN}" - "${checkpoint}" <<'PY'
import sys
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
if not keys:
    raise RuntimeError("checkpoint has no tensor keys")
required = {"decode_head.output.weight"}
missing = sorted(required - set(keys))
if missing:
    raise RuntimeError(f"checkpoint is missing expected RCDNet keys: {missing}")
print(f"[bootstrap_rcdnet] validated {len(keys)} tensors: {path}")
PY
}

if [[ ! -f "${RCDNET_ROOT}/RCDNet/models/builder.py" ]]; then
  mkdir -p "$(dirname "${RCDNET_ROOT}")"
  echo "[bootstrap_rcdnet] cloning official source -> ${RCDNET_ROOT}"
  git clone --depth 1 "${RCDNET_REPO_URL}" "${RCDNET_ROOT}"
fi

if [[ -d "${RCDNET_ROOT}/.git" ]]; then
  if ! git -C "${RCDNET_ROOT}" cat-file -e "${RCDNET_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${RCDNET_ROOT}" fetch --depth 1 origin "${RCDNET_COMMIT}"
  fi
  git -C "${RCDNET_ROOT}" checkout --detach "${RCDNET_COMMIT}"
else
  echo "[bootstrap_rcdnet] using source without Git metadata: ${RCDNET_ROOT}"
fi

if is_truthy "${RCDNET_DOWNLOAD_WEIGHTS}"; then
  if [[ ! -s "${RCDNET_CHECKPOINT}" ]]; then
    if ! "${PYTHON_BIN}" -c "import gdown" >/dev/null 2>&1; then
      echo "[bootstrap_rcdnet] gdown is required; install requirements-rcdnet.txt." >&2
      exit 1
    fi
    mkdir -p "$(dirname "${RCDNET_CHECKPOINT}")"
    PARTIAL="${RCDNET_CHECKPOINT}.part"
    echo "[bootstrap_rcdnet] downloading official SECOND real-data weights"
    "${PYTHON_BIN}" -m gdown \
      --continue \
      "https://drive.google.com/uc?id=${RCDNET_SECOND_GDRIVE_ID}" \
      --output "${PARTIAL}"
    validate_safetensors "${PARTIAL}"
    mv -f "${PARTIAL}" "${RCDNET_CHECKPOINT}"
  else
    validate_safetensors "${RCDNET_CHECKPOINT}"
  fi
else
  echo "[bootstrap_rcdnet] weight download disabled"
fi

if is_truthy "${RCDNET_BUILD_SELECTIVE_SCAN}"; then
  SELECTIVE_SCAN_ROOT="${RCDNET_ROOT}/RCDNet/models/encoders/selective_scan"
  echo "[bootstrap_rcdnet] building official selective-scan CUDA extension"
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e "${SELECTIVE_SCAN_ROOT}"
else
  echo "[bootstrap_rcdnet] selective-scan build disabled"
fi

echo "[bootstrap_rcdnet] source=${RCDNET_ROOT}"
echo "[bootstrap_rcdnet] checkpoint=${RCDNET_CHECKPOINT}"
echo "[bootstrap_rcdnet] done"
