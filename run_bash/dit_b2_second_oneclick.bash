#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_run() {
  if _truthy "${DRY_RUN}"; then
    printf '[dit_b2_second_oneclick] DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[dit_b2_second_oneclick] no Python interpreter found" >&2
    exit 2
  fi
fi

SECOND_ROOT="${SECOND_ROOT:-/root/data/second_dataset}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-/root/data/experiment/dit_b2_second_data}"
MANIFEST="${MANIFEST:-${DATA_OUTPUT_DIR}/second/train.jsonl}"
VAE_MODEL="${VAE_MODEL:-/root/data/weight/stable-diffusion-v1-5}"
VAE_SUBFOLDER="${VAE_SUBFOLDER:-auto}"
DIT_ROOT="${DIT_ROOT:-${ROOT_DIR}/third_party/DiT}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29631}"
PER_GPU_BATCH="${PER_GPU_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
FULL_NUM_WORKERS="${NUM_WORKERS:-2}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"
FULL_MAX_STEPS="${MAX_STEPS:-300000}"
FULL_OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/dit_b2_second_source_mask_256_seed42}"
FULL_RESUME="${RESUME:-auto}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/root/data/experiment/dit_b2_second_smoke}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_FULL="${RUN_FULL:-1}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

for numeric_name in NPROC_PER_NODE PER_GPU_BATCH GRAD_ACCUM FULL_MAX_STEPS; do
  numeric_value="${!numeric_name}"
  if ! [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[dit_b2_second_oneclick] ${numeric_name} must be a positive integer, got ${numeric_value}" >&2
    exit 2
  fi
done
if ! [[ "${FULL_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "[dit_b2_second_oneclick] FULL_NUM_WORKERS must be a non-negative integer, got ${FULL_NUM_WORKERS}" >&2
  exit 2
fi
if [[ "${DIST_BACKEND}" != "gloo" ]]; then
  echo "[dit_b2_second_oneclick] only the Gloo backend is supported, got: ${DIST_BACKEND}" >&2
  exit 2
fi

if ! _truthy "${DRY_RUN}"; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[dit_b2_second_oneclick] Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
  fi
  if [[ ! -d "${SECOND_ROOT}" ]]; then
    echo "[dit_b2_second_oneclick] SECOND root does not exist: ${SECOND_ROOT}" >&2
    exit 2
  fi
  if [[ ! -f "${VAE_MODEL}/vae/config.json" && ! -f "${VAE_MODEL}/config.json" ]]; then
    echo "[dit_b2_second_oneclick] cannot find a Diffusers VAE config under: ${VAE_MODEL}" >&2
    echo "Set VAE_MODEL to the local SD 1.5 snapshot (containing vae/config.json) or direct VAE folder." >&2
    exit 2
  fi
fi

if [[ "${VAE_SUBFOLDER}" == "auto" ]]; then
  if [[ -f "${VAE_MODEL}/vae/config.json" ]] || _truthy "${DRY_RUN}"; then
    VAE_SUBFOLDER="vae"
  else
    VAE_SUBFOLDER=""
  fi
fi

echo "[dit_b2_second_oneclick] repo=${ROOT_DIR}"
echo "[dit_b2_second_oneclick] python=${PYTHON_BIN}"
echo "[dit_b2_second_oneclick] SECOND=${SECOND_ROOT}"
echo "[dit_b2_second_oneclick] manifest=${MANIFEST}"
echo "[dit_b2_second_oneclick] VAE=${VAE_MODEL}"
echo "[dit_b2_second_oneclick] VAE_subfolder=${VAE_SUBFOLDER:-<direct>}"
echo "[dit_b2_second_oneclick] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE}"
echo "[dit_b2_second_oneclick] dist_backend=${DIST_BACKEND}"
echo "[dit_b2_second_oneclick] global_batch=$((PER_GPU_BATCH * NPROC_PER_NODE * GRAD_ACCUM))"
echo "[dit_b2_second_oneclick] num_workers=${FULL_NUM_WORKERS} resume=${FULL_RESUME}"
echo "[dit_b2_second_oneclick] smoke=${RUN_SMOKE} full=${RUN_FULL} full_steps=${FULL_MAX_STEPS}"

if _truthy "${INSTALL_DEPS}"; then
  _run "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-dit.txt"
fi

_run env DIT_ROOT="${DIT_ROOT}" bash "${ROOT_DIR}/scripts/bootstrap_dit.sh"

NEEDS_MANIFEST=0
if [[ ! -f "${MANIFEST}" ]]; then
  NEEDS_MANIFEST=1
elif ! grep -q '"target_mask_source"' "${MANIFEST}"; then
  echo "[dit_b2_second_oneclick] existing manifest is not online format; rebuilding"
  NEEDS_MANIFEST=1
elif _truthy "${REBUILD_MANIFEST}"; then
  NEEDS_MANIFEST=1
fi

if [[ "${NEEDS_MANIFEST}" == "1" ]]; then
  _run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    SECOND_ROOT="${SECOND_ROOT}" \
    SECOND_SPLITS=train \
    OUTPUT_DIR="${DATA_OUTPUT_DIR}" \
    bash "${ROOT_DIR}/run_bash/dit_b2_second_prepare.bash" --overwrite
else
  echo "[dit_b2_second_oneclick] reusing online manifest: ${MANIFEST}"
fi

if ! _truthy "${DRY_RUN}"; then
  if [[ ! -s "${MANIFEST}" ]]; then
    echo "[dit_b2_second_oneclick] online train manifest is missing or empty: ${MANIFEST}" >&2
    exit 2
  fi
  MANIFEST_ROWS="$("${PYTHON_BIN}" -c 'import sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))' "${MANIFEST}")"
  if ! [[ "${MANIFEST_ROWS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[dit_b2_second_oneclick] invalid manifest row count: ${MANIFEST_ROWS}" >&2
    exit 2
  fi
  echo "[dit_b2_second_oneclick] manifest_rows=${MANIFEST_ROWS}"
fi

_train() {
  local output_dir="$1"
  local max_steps="$2"
  local num_workers="$3"
  local resume="$4"
  shift 4
  _run env \
    PYTHON_BIN="${PYTHON_BIN}" \
    DIT_ROOT="${DIT_ROOT}" \
    VAE_MODEL="${VAE_MODEL}" \
    MANIFEST="${MANIFEST}" \
    GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_PORT="${MASTER_PORT}" \
    PER_GPU_BATCH="${PER_GPU_BATCH}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    NUM_WORKERS="${num_workers}" \
    DIST_BACKEND="${DIST_BACKEND}" \
    OUTPUT_DIR="${output_dir}" \
    MAX_STEPS="${max_steps}" \
    RESUME="${resume}" \
    bash "${ROOT_DIR}/run_bash/dit_b2_second_train.bash" --vae_subfolder "${VAE_SUBFOLDER}" "$@"
}

if _truthy "${RUN_SMOKE}"; then
  echo "[dit_b2_second_oneclick] stage=smoke_first two optimizer steps"
  _train "${SMOKE_OUTPUT_DIR}" 2 0 auto --save_every 1 --log_every 1 ${@+"$@"}
  echo "[dit_b2_second_oneclick] stage=smoke_resume resume to optimizer step three"
  _train "${SMOKE_OUTPUT_DIR}" 3 0 auto --save_every 1 --log_every 1 ${@+"$@"}
  echo "[dit_b2_second_oneclick] smoke/resume complete"
fi

if _truthy "${RUN_FULL}"; then
  echo "[dit_b2_second_oneclick] stage=full output=${FULL_OUTPUT_DIR}"
  _train "${FULL_OUTPUT_DIR}" "${FULL_MAX_STEPS}" "${FULL_NUM_WORKERS}" "${FULL_RESUME}" ${@+"$@"}
fi

echo "[dit_b2_second_oneclick] done"
