#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSUNIVLM_DATASET=loveda
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_loveda_val_native_rle24_v1}"
exec bash "${SCRIPT_DIR}/rsunivlm_eval.bash" "$@"
