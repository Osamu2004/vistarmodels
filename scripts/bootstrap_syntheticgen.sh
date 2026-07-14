#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${SYNTHETICGEN_ROOT:-${ROOT_DIR}/third_party/SyntheticGen}"
WEIGHT_DIR="${SYNTHETICGEN_WEIGHT_DIR:-/data/vistar/weights/syntheticgen}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
XET_HELPER="${ROOT_DIR}/../tools/download_hf_xet_mirror.py"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/Buddhi19/SyntheticGen.git "${REPO_DIR}"
fi
for name in layout controlnet; do
  zip="${WEIGHT_DIR}/${name}.zip"
  if [[ ! -f "${zip}" ]]; then
    [[ -f "${XET_HELPER}" ]] || { echo "Missing downloader: ${XET_HELPER}" >&2; exit 2; }
    "${PYTHON_BIN}" "${XET_HELPER}" --repo buddhi19/SyntheticGen --filename "${name}.zip" --output "${zip}"
  fi
  mkdir -p "${WEIGHT_DIR}/${name}"
  unzip -q -o "${zip}" -d "${WEIGHT_DIR}/${name}"
done
test -f "${WEIGHT_DIR}/layout/checkpoint-79000/model.safetensors"
test -f "${WEIGHT_DIR}/controlnet/checkpoint-112000/model.safetensors"
echo "[bootstrap_syntheticgen] ready: ${REPO_DIR} ${WEIGHT_DIR}"
