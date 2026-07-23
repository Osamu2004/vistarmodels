#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="xbd_pre"
export MASTER_PORT="${MASTER_PORT:-29723}"
exec bash "${SCRIPT_DIR}/vip_eval.bash" "$@"
