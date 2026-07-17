#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[controlnet_second_gen] no Python interpreter found" >&2
    exit 2
  fi
fi
BASE_MODEL="${BASE_MODEL:-/root/data/weight/stable-diffusion-v1-5}"
CONTROLNET="${CONTROLNET:?set CONTROLNET to the completed training output directory}"
MANIFEST="${MANIFEST:-/root/data/experiment/controlnet_second_data/second/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/controlnet_sd15_second_mask_text_test_256_steps50_seed42}"
GPU_IDS="${GPU_IDS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29642}"
RESOLUTION="${RESOLUTION:-256}"
STEPS="${STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
ARGS=(
  --manifest "${MANIFEST}"
  --base_model "${BASE_MODEL}"
  --controlnet "${CONTROLNET}"
  --output_dir "${OUTPUT_DIR}"
  --resolution "${RESOLUTION}"
  --steps "${STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --controlnet_scale "${CONTROLNET_SCALE}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --max_samples "${MAX_SAMPLES}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi
if [[ "${ENABLE_XFORMERS:-0}" == "1" ]]; then ARGS+=(--enable_xformers); fi
if [[ -n "${NEGATIVE_PROMPT:-}" ]]; then ARGS+=(--negative_prompt "${NEGATIVE_PROMPT}"); fi
if (( $# > 0 )); then
  ARGS+=("$@")
fi

SCRIPT="${ROOT_DIR}/baselines/controlnet/run_controlnet_second.py"
if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  COMMAND=("${PYTHON_BIN}" -u "${SCRIPT}" "${ARGS[@]}")
else
  COMMAND=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" "${SCRIPT}" "${ARGS[@]}")
fi
echo "[controlnet_second_gen] base_model=${BASE_MODEL}"
echo "[controlnet_second_gen] controlnet=${CONTROLNET}"
echo "[controlnet_second_gen] manifest=${MANIFEST}"
echo "[controlnet_second_gen] output=${OUTPUT_DIR}"
echo "[controlnet_second_gen] gpu_ids=${GPU_IDS} nproc=${NPROC_PER_NODE} batch=${BATCH_SIZE}"
echo "[controlnet_second_gen] resolution=${RESOLUTION} steps=${STEPS} cfg=${GUIDANCE_SCALE} control_scale=${CONTROLNET_SCALE} seed=${SEED}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[controlnet_second_gen] dry_run:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/check_controlnet_deps.py" \
  --base_model "${BASE_MODEL}" --controlnet "${CONTROLNET}" \
  --manifest "${MANIFEST}" --require_cuda
"${COMMAND[@]}"

if [[ "${COMPUTE_METRICS:-0}" == "1" ]]; then
  VISTAR_ROOT="${VISTAR_ROOT:-/root/code/vistar}"
  METRIC_SCRIPT="${VISTAR_ROOT}/run_bash/seg/compute_saved_second_gen_metrics.bash"
  [[ -f "${METRIC_SCRIPT}" ]] || { echo "missing VISTAR metric script: ${METRIC_SCRIPT}" >&2; exit 2; }
  METRIC_GPU_IDS="${METRIC_GPU_IDS:-${GPU_IDS%%,*}}"
  echo "[controlnet_second_gen] computing common SECOND generation metrics on GPU ${METRIC_GPU_IDS}"
  GPU_IDS="${METRIC_GPU_IDS}" INPUT_SIZE="${RESOLUTION}" SPLIT=test DIRECTIONS=both \
    NUM_INFERENCE_STEPS="${STEPS}" CONDITION_MASK_MODE=semantic CDGEN_METRICS=1 \
    bash "${METRIC_SCRIPT}" "${OUTPUT_DIR}"
fi
