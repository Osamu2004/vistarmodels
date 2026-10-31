#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSUNIVLM_DATASET=loveda
export TILE_SIZE="${TILE_SIZE:-512}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_loveda_val_native_tile${TILE_SIZE}_rle24_v2}"
exec bash "${SCRIPT_DIR}/rsunivlm_eval.bash" "$@"
