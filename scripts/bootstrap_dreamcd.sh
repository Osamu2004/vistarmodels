#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
DREAMCD_ROOT="${DREAMCD_ROOT:-${THIRD_PARTY_DIR}/DreamCD}"
DREAMCD_REPO="${DREAMCD_REPO:-https://github.com/tangkai-RS/DreamCD.git}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${DREAMCD_ROOT}/.git" ]]; then
  echo "[bootstrap_dreamcd] DreamCD already exists: ${DREAMCD_ROOT}"
  git -C "${DREAMCD_ROOT}" pull --ff-only
else
  echo "[bootstrap_dreamcd] cloning ${DREAMCD_REPO} -> ${DREAMCD_ROOT}"
  git clone "${DREAMCD_REPO}" "${DREAMCD_ROOT}"
fi

echo "[bootstrap_dreamcd] done"
echo "[bootstrap_dreamcd] checkpoints expected at:"
echo "  ${DREAMCD_ROOT}/checkpoints/second/vqvae.ckpt"
echo "  ${DREAMCD_ROOT}/checkpoints/second/ldm.ckpt"
echo "[bootstrap_dreamcd] official weights: https://huggingface.co/tangkaii/DreamCD"
echo "[bootstrap_dreamcd] dependency check:"
echo "  python tools/check_dreamcd_deps.py"
