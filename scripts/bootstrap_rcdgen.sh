#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
RCDGEN_ROOT="${RCDGEN_ROOT:-${THIRD_PARTY_DIR}/referring_change_detection}"
RCDGEN_REPO="${RCDGEN_REPO:-https://github.com/yilmazkorkmaz1/referring_change_detection.git}"
RCDGEN_REVISION="${RCDGEN_REVISION:-main}"
RCDGEN_MODEL_ID="${RCDGEN_MODEL_ID:-yilmazkorkmaz/RCDGen}"
RCDGEN_WEIGHT_ROOT="${RCDGEN_WEIGHT_ROOT:-/root/data/weight/rcdgen}"
RCDGEN_MODEL_DIR="${RCDGEN_MODEL_DIR:-${RCDGEN_WEIGHT_ROOT}/RCDGen}"
RCDGEN_DOWNLOAD_WEIGHTS="${RCDGEN_DOWNLOAD_WEIGHTS:-1}"

mkdir -p "${THIRD_PARTY_DIR}"
PIPELINE_SOURCE="${RCDGEN_ROOT}/RCDGen/RCDGenSDPipeline.py"
if [[ -f "${PIPELINE_SOURCE}" ]]; then
  echo "[bootstrap_rcdgen] using vendored official pipeline: ${PIPELINE_SOURCE}"
elif [[ -d "${RCDGEN_ROOT}/.git" ]]; then
  echo "[bootstrap_rcdgen] official source exists: ${RCDGEN_ROOT}"
else
  echo "[bootstrap_rcdgen] cloning official source -> ${RCDGEN_ROOT}"
  git clone --depth 1 --branch "${RCDGEN_REVISION}" "${RCDGEN_REPO}" "${RCDGEN_ROOT}"
fi

if [[ ! -f "${PIPELINE_SOURCE}" ]]; then
  echo "[bootstrap_rcdgen] missing official pipeline: ${PIPELINE_SOURCE}" >&2
  exit 1
fi

DIFFUSERS_STABLE_DIR="$(${PYTHON_BIN} -c 'import pathlib, diffusers; print(pathlib.Path(diffusers.__file__).resolve().parent / "pipelines" / "stable_diffusion")')"
PIPELINE_TARGET="${DIFFUSERS_STABLE_DIR}/RCDGenSDPipeline.py"
if [[ ! -f "${PIPELINE_TARGET}" ]] || ! cmp -s "${PIPELINE_SOURCE}" "${PIPELINE_TARGET}"; then
  echo "[bootstrap_rcdgen] installing official custom pipeline -> ${PIPELINE_TARGET}"
  cp "${PIPELINE_SOURCE}" "${PIPELINE_TARGET}"
fi

case "$(printf '%s' "${RCDGEN_DOWNLOAD_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    echo "[bootstrap_rcdgen] downloading/resuming ${RCDGEN_MODEL_ID} -> ${RCDGEN_MODEL_DIR}"
    RCDGEN_MODEL_ID="${RCDGEN_MODEL_ID}" RCDGEN_MODEL_DIR="${RCDGEN_MODEL_DIR}" \
      "${PYTHON_BIN}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["RCDGEN_MODEL_ID"], local_dir=os.environ["RCDGEN_MODEL_DIR"])'
    ;;
  *) echo "[bootstrap_rcdgen] weight download disabled" ;;
esac

echo "[bootstrap_rcdgen] done"
