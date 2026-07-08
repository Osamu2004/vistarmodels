#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEG2ANY_ROOT="${SEG2ANY_ROOT:-${ROOT_DIR}/third_party/Seg2Any}"
SEG2ANY_REPO="${SEG2ANY_REPO:-https://github.com/0xLDF/Seg2Any.git}"

mkdir -p "$(dirname "${SEG2ANY_ROOT}")"

if [[ -d "${SEG2ANY_ROOT}/.git" ]]; then
  echo "[bootstrap_seg2any] Seg2Any already exists: ${SEG2ANY_ROOT}"
  git -C "${SEG2ANY_ROOT}" pull --ff-only
else
  echo "[bootstrap_seg2any] cloning ${SEG2ANY_REPO} -> ${SEG2ANY_ROOT}"
  git clone "${SEG2ANY_REPO}" "${SEG2ANY_ROOT}"
fi

echo "[bootstrap_seg2any] done"
echo "[bootstrap_seg2any] expected LoRA default: /root/data/weight/seg2any/sacap_1m/seg2any/checkpoint-20000"
echo "[bootstrap_seg2any] download LoRA with:"
echo "  huggingface-cli download 0xLDF/Seg2Any --local-dir /root/data/weight/seg2any"
echo "[bootstrap_seg2any] FLUX.1-dev is gated; use HuggingFace login or set SEG2ANY_FLUX1_MODEL=/path/to/FLUX.1-dev"
