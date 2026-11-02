#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSUNIVLM_DATASET=loveda
export TILE_SIZE="${TILE_SIZE:-512}"
if [[ "${TILE_BATCH_SIZE:-1}" != "1" || -n "${IMPORT_V4_ROOT:-}" ]]; then
  export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_loveda_val_native_tile${TILE_SIZE}_rle_v5_bs${TILE_BATCH_SIZE:-1}}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_loveda_val_native_tile${TILE_SIZE}_rle_v4}"
fi
exec bash "${SCRIPT_DIR}/rsunivlm_eval.bash" "$@"
