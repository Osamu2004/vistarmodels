#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIT_ROOT="${DIT_ROOT:-${ROOT_DIR}/third_party/DiT}"
DIT_REPO="${DIT_REPO:-https://github.com/facebookresearch/DiT.git}"
DIT_REVISION="${DIT_REVISION:-ed81ce2229091fd4ecc9a223645f95cf379d582b}"
DIT_ALLOW_UNPINNED_SOURCE="${DIT_ALLOW_UNPINNED_SOURCE:-0}"

mkdir -p "$(dirname "${DIT_ROOT}")"
if [[ -d "${DIT_ROOT}/.git" ]]; then
  CURRENT_REVISION="$(git -C "${DIT_ROOT}" rev-parse HEAD)"
  if [[ "${CURRENT_REVISION}" != "${DIT_REVISION}" && "${DIT_ALLOW_UNPINNED_SOURCE}" != "1" ]]; then
    echo "[bootstrap_dit] revision mismatch" >&2
    echo "  expected: ${DIT_REVISION}" >&2
    echo "  current:  ${CURRENT_REVISION}" >&2
    echo "Set DIT_ALLOW_UNPINNED_SOURCE=1 only after auditing the source." >&2
    exit 2
  fi
else
  echo "[bootstrap_dit] cloning ${DIT_REPO} -> ${DIT_ROOT}"
  git clone "${DIT_REPO}" "${DIT_ROOT}"
  git -C "${DIT_ROOT}" checkout --detach "${DIT_REVISION}"
fi

for relative in models.py diffusion; do
  if [[ ! -e "${DIT_ROOT}/${relative}" ]]; then
    echo "[bootstrap_dit] missing official DiT component: ${DIT_ROOT}/${relative}" >&2
    exit 2
  fi
done

echo "[bootstrap_dit] source=${DIT_ROOT}"
echo "[bootstrap_dit] revision=$(git -C "${DIT_ROOT}" rev-parse HEAD)"
