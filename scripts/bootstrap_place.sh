#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
PLACE_ROOT="${PLACE_ROOT:-${THIRD_PARTY_DIR}/PLACE}"
PLACE_REPO="${PLACE_REPO:-https://github.com/cszy98/PLACE.git}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${PLACE_ROOT}/.git" ]]; then
  echo "[bootstrap_place] PLACE already exists: ${PLACE_ROOT}"
  git -C "${PLACE_ROOT}" pull --ff-only
else
  echo "[bootstrap_place] cloning ${PLACE_REPO} -> ${PLACE_ROOT}"
  git clone "${PLACE_REPO}" "${PLACE_ROOT}"
fi

echo "[bootstrap_place] done"
echo "[bootstrap_place] expected default checkpoint: /root/data/weight/place/coco_best.ckpt"
echo "[bootstrap_place] official pretrained weights:"
echo "  https://drive.google.com/drive/folders/1b5pC52hasLwm1gOkc9LmdIyxZjrdlNWC"
echo "[bootstrap_place] run dependency check with:"
echo "  python tools/check_place_deps.py"
