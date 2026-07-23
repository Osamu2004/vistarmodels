#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="uavid"
export MASTER_PORT="${MASTER_PORT:-29722}"
exec bash "${SCRIPT_DIR}/vip_eval.bash" "$@"
