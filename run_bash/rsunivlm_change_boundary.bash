#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
SOURCE_RUN="${SOURCE_RUN:-/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1_formatfix_rle_v1}"
if [[ "${RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest tests/test_rsunivlm_change_boundary.py -q
fi
args=(--source_run "$SOURCE_RUN" --workers "${SCORE_WORKERS:-2}")
if [[ -n "${SUMMARY_RUN:-}" ]]; then args+=(--summary_run "$SUMMARY_RUN"); fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then args+=(--output_dir "$OUTPUT_DIR"); fi
exec python -u tools/score_rsunivlm_change_boundary.py "${args[@]}" "$@"
