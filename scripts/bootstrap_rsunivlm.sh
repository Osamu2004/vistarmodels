#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RSUNIVLM_ROOT="${RSUNIVLM_ROOT:-${ROOT_DIR}/third_party/RSUniVLM}"
RSUNIVLM_MODEL_PATH="${RSUNIVLM_MODEL_PATH:-/root/data/weight/rsunivlm/llava-qwen-gmoe}"
PIN="66dce1dc68b81a6c62f02fb8e5b306571092a864"
OFFICIAL_WEIGHTS="https://drive.google.com/drive/folders/1TtaoOPmh167gpgHHWRNBMCaA7t_XZ4Vg"

if [[ ! -e "${RSUNIVLM_ROOT}" ]]; then
  mkdir -p "$(dirname "${RSUNIVLM_ROOT}")"
  git clone --depth 1 https://github.com/xuliu-cyber/RSUniVLM.git "${RSUNIVLM_ROOT}"
fi
if [[ ! -d "${RSUNIVLM_ROOT}/.git" ]]; then
  echo "[RSUniVLM] Expected official Git checkout at ${RSUNIVLM_ROOT}; existing content preserved." >&2
  exit 1
fi
if [[ "$(git -C "${RSUNIVLM_ROOT}" rev-parse HEAD)" != "${PIN}" ]]; then
  git -C "${RSUNIVLM_ROOT}" fetch --depth 1 origin "${PIN}"
  git -C "${RSUNIVLM_ROOT}" checkout --detach "${PIN}"
fi

# Source is imported explicitly by the wrapper. Do not pip-install upstream's
# training extras: they pin torch 2.1.2 and downgrade an RTX 50-series runtime.
if [[ "${RSUNIVLM_DOWNLOAD_WEIGHTS:-1}" == "1" ]]; then
  if ! PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -c 'import sys; from pathlib import Path; from baselines.rsunivlm.protocols import checkpoint_identity; checkpoint_identity(Path(sys.argv[1]).expanduser().resolve())' \
      "${RSUNIVLM_MODEL_PATH}" >/dev/null 2>&1; then
    mkdir -p "${RSUNIVLM_MODEL_PATH}"
    "${PYTHON_BIN}" -m gdown --folder "${OFFICIAL_WEIGHTS}" -O "${RSUNIVLM_MODEL_PATH}"
  fi
fi
echo "[RSUniVLM] source=${RSUNIVLM_ROOT} revision=${PIN}"
echo "[RSUniVLM] checkpoint=${RSUNIVLM_MODEL_PATH}"
echo "[RSUniVLM] SigLIP is loaded from the checkpoint mm_vision_tower config; set RSUNIVLM_VISION_TOWER for a local copy."
