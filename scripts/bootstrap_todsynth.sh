#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
TODSYNTH_ROOT="${TODSYNTH_ROOT:-${ROOT_DIR}/third_party/crfm}"
TODSYNTH_REPO="${TODSYNTH_REPO:-https://github.com/Yunkai-Yang/crfm.git}"
TODSYNTH_REVISION="${TODSYNTH_REVISION:-6da6634046f6834d756b58256eabad398b1fe1d7}"
SD35_MODEL_ID="${SD35_MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
SD35_MODEL_DIR="${SD35_MODEL_DIR:-/data/vistar/weights/todsynth/sd3.5-medium}"
DOWNLOAD_SD35="${DOWNLOAD_SD35:-0}"
if [[ -f "${TODSYNTH_ROOT}/.vistar_revision" ]]; then
  test "$(cat "${TODSYNTH_ROOT}/.vistar_revision")" = "${TODSYNTH_REVISION}"
else
  if [[ ! -d "${TODSYNTH_ROOT}/.git" ]]; then git clone "${TODSYNTH_REPO}" "${TODSYNTH_ROOT}"; fi
  git -C "${TODSYNTH_ROOT}" fetch origin "${TODSYNTH_REVISION}"
  git -C "${TODSYNTH_ROOT}" checkout "${TODSYNTH_REVISION}"
  git -C "${TODSYNTH_ROOT}" rev-parse HEAD
fi
if [[ "${DOWNLOAD_SD35}" == "1" ]]; then
  SD35_MODEL_ID="${SD35_MODEL_ID}" SD35_MODEL_DIR="${SD35_MODEL_DIR}" \
    "${PYTHON_BIN}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["SD35_MODEL_ID"], local_dir=os.environ["SD35_MODEL_DIR"])'
fi
echo "[bootstrap_todsynth] source=${TODSYNTH_ROOT} base_model=${SD35_MODEL_DIR} (gated; DOWNLOAD_SD35=${DOWNLOAD_SD35})"
