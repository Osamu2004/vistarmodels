#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSUNIVLM_DATASET=levircd
export CD_DIRECTION="${CD_DIRECTION:-forward}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_levircd_${SPLIT:-test}_native_rle_${CD_DIRECTION}_v3}"
exec bash "${SCRIPT_DIR}/rsunivlm_eval.bash" "$@"
