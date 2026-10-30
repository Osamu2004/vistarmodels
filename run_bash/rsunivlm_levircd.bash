#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSUNIVLM_DATASET=levircd
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_levircd_${SPLIT:-test}_native_rle24_${CD_DIRECTION:-symmetric_union}_v1}"
exec bash "${SCRIPT_DIR}/rsunivlm_eval.bash" "$@"
