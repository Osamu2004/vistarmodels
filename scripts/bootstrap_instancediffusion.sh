#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
INSTANCEDIFFUSION_DIFFUSERS_ROOT="${INSTANCEDIFFUSION_DIFFUSERS_ROOT:-${THIRD_PARTY_DIR}/diffusers-instancediffusion}"
INSTANCEDIFFUSION_DIFFUSERS_REPO="${INSTANCEDIFFUSION_DIFFUSERS_REPO:-https://github.com/gokyeongryeol/diffusers.git}"
INSTANCEDIFFUSION_DIFFUSERS_BRANCH="${INSTANCEDIFFUSION_DIFFUSERS_BRANCH:-instancediffusion}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}/.git" ]]; then
  echo "[bootstrap_instancediffusion] diffusers fork already exists: ${INSTANCEDIFFUSION_DIFFUSERS_ROOT}"
  git -C "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}" fetch origin "${INSTANCEDIFFUSION_DIFFUSERS_BRANCH}"
  git -C "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}" checkout "${INSTANCEDIFFUSION_DIFFUSERS_BRANCH}"
  git -C "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}" pull --ff-only
else
  echo "[bootstrap_instancediffusion] cloning ${INSTANCEDIFFUSION_DIFFUSERS_REPO} (${INSTANCEDIFFUSION_DIFFUSERS_BRANCH}) -> ${INSTANCEDIFFUSION_DIFFUSERS_ROOT}"
  git clone --branch "${INSTANCEDIFFUSION_DIFFUSERS_BRANCH}" "${INSTANCEDIFFUSION_DIFFUSERS_REPO}" "${INSTANCEDIFFUSION_DIFFUSERS_ROOT}"
fi

echo "[bootstrap_instancediffusion] done"
echo "[bootstrap_instancediffusion] default model: kyeongry/instancediffusion_sd15"
echo "[bootstrap_instancediffusion] run dependency check with:"
echo "  python tools/check_instancediffusion_deps.py"
