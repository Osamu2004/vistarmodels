#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="loveda"
export MASTER_PORT="${MASTER_PORT:-29660}"
exec bash "${SCRIPT_DIR}/segearth_ov_eval.bash" "$@"
