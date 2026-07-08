#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CRSDIFF_ROOT="${CRSDIFF_ROOT:-${THIRD_PARTY_DIR}/CRS-Diff}"
CRSDIFF_REPO="${CRSDIFF_REPO:-https://github.com/Sonettoo/CRS-Diff.git}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${CRSDIFF_ROOT}/.git" ]]; then
  echo "[bootstrap_crsdiff] CRS-Diff already exists: ${CRSDIFF_ROOT}"
  git -C "${CRSDIFF_ROOT}" pull --ff-only
else
  echo "[bootstrap_crsdiff] cloning ${CRSDIFF_REPO} -> ${CRSDIFF_ROOT}"
  git clone "${CRSDIFF_REPO}" "${CRSDIFF_ROOT}"
fi

echo "[bootstrap_crsdiff] done"
echo "[bootstrap_crsdiff] checkpoint expected at /root/data/weight/crsdiff/last.ckpt"
echo "[bootstrap_crsdiff] official weights: https://huggingface.co/Sonetto702/AeroGen/tree/main"

