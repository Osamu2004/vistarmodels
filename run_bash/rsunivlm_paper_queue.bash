#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export RSUNIVLM_ROOT="${RSUNIVLM_ROOT:-${ROOT_DIR}/third_party/RSUniVLM_clean_66dce1d}"
export RSUNIVLM_MODEL_PATH="${RSUNIVLM_MODEL_PATH:-/root/data/weight/llava-qwen-gmoe}"
export RSUNIVLM_VISION_TOWER="${RSUNIVLM_VISION_TOWER:-/root/data/weight/siglip-so400m-patch14-384}"
GPU_IDS="${GPU_IDS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CMD=("${PYTHON_BIN:-python}" "${ROOT_DIR}/tools/eval_rsunivlm_paper_queue.py"
  --datasets "${DATASETS:-loveda,flair,uavid,xbd_pre,chn6_cug,levircd,second}"
  --semantic_mode "${SEMANTIC_MODE:-native_classwise}"
  --gpu_ids "${GPU_IDS}" --batch_size "${BATCH_SIZE}"
  --output_dir "${OUTPUT_DIR:-/root/data/experiment/rsunivlm_native_classwise_tile512_bs${BATCH_SIZE}_2gpu_v1}"
  --max_samples "${MAX_SAMPLES:-0}" --dtype "${DTYPE:-float16}" --attention "${ATTENTION:-eager}"
  --max_new_tokens "${MAX_NEW_TOKENS:-32000}" --format_retries "${FORMAT_RETRIES:-0}")
if [[ -n "${DATA_ROOTS_JSON:-}" ]]; then CMD+=(--data_roots_json "${DATA_ROOTS_JSON}"); fi
if [[ "${PLAN_ONLY:-0}" == "1" ]]; then CMD+=(--plan_only); fi
CMD+=("$@")
exec "${CMD[@]}"
