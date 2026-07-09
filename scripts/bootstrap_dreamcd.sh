#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
PYTHON_BIN="${PYTHON_BIN:-python}"
DREAMCD_ROOT="${DREAMCD_ROOT:-${THIRD_PARTY_DIR}/DreamCD}"
DREAMCD_REPO="${DREAMCD_REPO:-https://github.com/tangkai-RS/DreamCD.git}"
DREAMCD_HF_REPO="${DREAMCD_HF_REPO:-tangkaii/DreamCD}"
DREAMCD_HF_REVISION="${DREAMCD_HF_REVISION:-main}"
DREAMCD_WEIGHT_SET="${DREAMCD_WEIGHT_SET:-second}"
DREAMCD_CKPT="${DREAMCD_CKPT:-${DREAMCD_ROOT}/checkpoints/${DREAMCD_WEIGHT_SET}/ldm.ckpt}"
DREAMCD_VQVAE_CKPT="${DREAMCD_VQVAE_CKPT:-${DREAMCD_ROOT}/checkpoints/${DREAMCD_WEIGHT_SET}/vqvae.ckpt}"
DREAMCD_DOWNLOAD_WEIGHTS="${DREAMCD_DOWNLOAD_WEIGHTS:-1}"

_is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_download_hf_file() {
  local hf_filename="$1"
  local dst_path="$2"

  if [[ -s "${dst_path}" ]]; then
    echo "[bootstrap_dreamcd] weight exists: ${dst_path}"
    return 0
  fi

  echo "[bootstrap_dreamcd] downloading ${DREAMCD_HF_REPO}/${hf_filename} -> ${dst_path}"
  "${PYTHON_BIN}" - "${DREAMCD_HF_REPO}" "${hf_filename}" "${dst_path}" "${DREAMCD_HF_REVISION}" <<'PY'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

repo_id, filename, dst_text, revision = sys.argv[1:5]
dst = Path(dst_text).expanduser().resolve()
dst.parent.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print(
        "[bootstrap_dreamcd] Missing Python package: huggingface_hub. "
        "Install it with `pip install huggingface-hub` or "
        "`pip install -r requirements-dreamcd.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(2)

downloaded = Path(
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision or None,
    )
).expanduser().resolve()

if downloaded != dst:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(downloaded, tmp)
    os.replace(tmp, dst)

if not dst.is_file() or dst.stat().st_size <= 0:
    raise RuntimeError(f"downloaded file is missing or empty: {dst}")

print(f"[bootstrap_dreamcd] downloaded weight: {dst} ({dst.stat().st_size} bytes)")
PY
}

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${DREAMCD_ROOT}/.git" ]]; then
  echo "[bootstrap_dreamcd] DreamCD already exists: ${DREAMCD_ROOT}"
  git -C "${DREAMCD_ROOT}" pull --ff-only
else
  echo "[bootstrap_dreamcd] cloning ${DREAMCD_REPO} -> ${DREAMCD_ROOT}"
  git clone "${DREAMCD_REPO}" "${DREAMCD_ROOT}"
fi

if _is_truthy "${DREAMCD_DOWNLOAD_WEIGHTS}"; then
  _download_hf_file "${DREAMCD_WEIGHT_SET}/vqvae.ckpt" "${DREAMCD_VQVAE_CKPT}"
  _download_hf_file "${DREAMCD_WEIGHT_SET}/ldm.ckpt" "${DREAMCD_CKPT}"
else
  echo "[bootstrap_dreamcd] skip weight download because DREAMCD_DOWNLOAD_WEIGHTS=${DREAMCD_DOWNLOAD_WEIGHTS}"
fi

echo "[bootstrap_dreamcd] done"
echo "[bootstrap_dreamcd] checkpoints:"
echo "  ${DREAMCD_VQVAE_CKPT}"
echo "  ${DREAMCD_CKPT}"
echo "[bootstrap_dreamcd] official weights: https://huggingface.co/tangkaii/DreamCD"
echo "[bootstrap_dreamcd] dependency check:"
echo "  python tools/check_dreamcd_deps.py"
