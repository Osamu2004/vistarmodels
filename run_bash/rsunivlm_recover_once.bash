#!/usr/bin/env bash
set -euo pipefail
RECOVERY_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RECOVERY_REPO}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN="${SOURCE_RUN:-/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1}"
REUSE_RUN="${REUSE_RUN:-${SOURCE_RUN}_formatfix_v1_retry2}"

if [[ ! -d "${REUSE_RUN}" ]]; then
  printf 'Previous recovery directory missing; set REUSE_RUN to the actual finished OUTPUT_DIR: %s\n' "${REUSE_RUN}" >&2
  exit 2
fi
if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  printf '[recovery once] WSL regression checks\n'
  "${PYTHON_BIN}" -m pytest tests/test_rsunivlm_recovery_once.py \
    tests/test_rsunivlm_levir_labels.py tests/test_rsunivlm_format_repair.py \
    tests/test_rsunivlm_format_repair_pipeline.py tests/test_rsunivlm_failed_recovery.py \
    tests/test_rsunivlm_guarded.py -q
fi
printf '[recovery once] Audit LEVIR GT using the existing Vistar positive rule\n'
"${PYTHON_BIN}" tools/rsunivlm_levir_labels.py --source_run "${SOURCE_RUN}" --gt_mode vistar_positive
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export RSUNIVLM_NUM_WORKERS="${RSUNIVLM_NUM_WORKERS:-2}" RSUNIVLM_PREFETCH_FACTOR="${RSUNIVLM_PREFETCH_FACTOR:-2}"
printf '[recovery once] Reuse previous results, repair, infer remaining queries, then finalize\n'
"${PYTHON_BIN}" tools/repair_rsunivlm_format.py \
  --source_run "${SOURCE_RUN}" --reuse_run "${REUSE_RUN}" \
  --datasets loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second \
  --levircd_gt_mode vistar_positive --run_remaining
