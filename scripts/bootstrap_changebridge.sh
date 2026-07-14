#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGEBRIDGE_ROOT="${CHANGEBRIDGE_ROOT:-${ROOT_DIR}/third_party/ChangeBridge}"
CHANGEBRIDGE_REPO="${CHANGEBRIDGE_REPO:-https://github.com/zhenghuizhao/ChangeBridge.git}"
CHANGEBRIDGE_REVISION="${CHANGEBRIDGE_REVISION:-7fefb2f5102d9e2403ed02cafab143c65f4cf1bc}"
if [[ ! -d "${CHANGEBRIDGE_ROOT}/.git" ]]; then
  git clone "${CHANGEBRIDGE_REPO}" "${CHANGEBRIDGE_ROOT}"
fi
git -C "${CHANGEBRIDGE_ROOT}" fetch origin "${CHANGEBRIDGE_REVISION}"
git -C "${CHANGEBRIDGE_ROOT}" checkout "${CHANGEBRIDGE_REVISION}"
git -C "${CHANGEBRIDGE_ROOT}" rev-parse HEAD
if [[ ! -d "${CHANGEBRIDGE_ROOT}/runners" ]]; then
  echo "[bootstrap_changebridge] upstream checkout is incomplete: runners/ is absent." >&2
  echo "Obtain the missing official runner package from the authors before training/sampling." >&2
  exit 2
fi
