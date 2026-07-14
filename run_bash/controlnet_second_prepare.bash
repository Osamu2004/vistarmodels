#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" "${ROOT_DIR}/tools/build_controlnet_second_dataset.py" "${@}"
