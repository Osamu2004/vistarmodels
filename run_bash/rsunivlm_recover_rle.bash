#!/usr/bin/env bash
set -euo pipefail
RLE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RLE_REPO}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN="${SOURCE_RUN:-/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1}"
if [[ -z "${REUSE_RUN:-}" ]]; then
  if [[ -f "${SOURCE_RUN}_formatfix_once_v2/plan.json" ]]; then
    REUSE_RUN="${SOURCE_RUN}_formatfix_once_v2"
  else
    REUSE_RUN="${SOURCE_RUN}_formatfix_v1_retry2"
  fi
fi
if [[ ! -d "${REUSE_RUN}" ]]; then
  printf 'Set REUSE_RUN to the actual finished recovery OUTPUT_DIR: %s\n' "${REUSE_RUN}" >&2
  exit 2
fi
RLE_ROWS="${RLE_ROWS:-24}"
RLE_COLS="${RLE_COLS:-24}"
if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" -m pytest tests/test_rsunivlm_constrained_rle.py \
    tests/test_rsunivlm_recovery_once.py tests/test_rsunivlm_levir_labels.py \
    tests/test_rsunivlm_format_repair.py tests/test_rsunivlm_format_repair_pipeline.py \
    tests/test_rsunivlm_failed_recovery.py tests/test_rsunivlm_guarded.py -q
fi
"${PYTHON_BIN}" tools/check_rsunivlm_rle_tokenizer.py --source_run "${SOURCE_RUN}" --rows "${RLE_ROWS}" --cols "${RLE_COLS}"
"${PYTHON_BIN}" tools/rsunivlm_levir_labels.py --source_run "${SOURCE_RUN}" --gt_mode vistar_positive
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export RSUNIVLM_NUM_WORKERS="${RSUNIVLM_NUM_WORKERS:-2}" RSUNIVLM_PREFETCH_FACTOR="${RSUNIVLM_PREFETCH_FACTOR:-2}"
printf '[RLE recovery] Reusing %s; constrained fallback grid=%sx%s\n' "${REUSE_RUN}" "${RLE_ROWS}" "${RLE_COLS}"
"${PYTHON_BIN}" tools/repair_rsunivlm_format.py \
  --source_run "${SOURCE_RUN}" --reuse_run "${REUSE_RUN}" \
  --datasets loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
  --levircd_gt_mode vistar_positive --run_remaining \
  --constrained_rle --rle_rows "${RLE_ROWS}" --rle_cols "${RLE_COLS}"
