#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_second_gen}"
RSEDIT_MODEL_DIR="${RSEDIT_MODEL_DIR:-/root/data/weight/rsedit/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsedit_second_fullmask_source_text_steps50_seed42}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
BOOTSTRAP_RSEDIT="${BOOTSTRAP_RSEDIT:-1}"

if [[ "${BOOTSTRAP_RSEDIT}" == "1" ]]; then
  RSEDIT_MODEL_DIR="${RSEDIT_MODEL_DIR}" bash "${ROOT_DIR}/scripts/bootstrap_rsedit.sh"
fi
mkdir -p "${OUTPUT_DIR}"
BUILD_ARGS=(); RUN_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then BUILD_ARGS+=(--max_samples "${MAX_SAMPLES}"); RUN_ARGS+=(--max_samples "${MAX_SAMPLES}"); fi
if [[ "${OVERWRITE}" == "1" ]]; then RUN_ARGS+=(--overwrite); fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_second_manifest_from_vistar_eval.py" \
  --eval_dir "${VISTAR_EVAL_DIR}" --output "${MANIFEST}" "${BUILD_ARGS[@]}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/baselines/rsedit/run_rsedit_manifest.py" \
  --manifest "${MANIFEST}" --output_dir "${OUTPUT_DIR}" --model "${RSEDIT_MODEL_DIR}" "${RUN_ARGS[@]}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/merge_ranked_jsonl.py" --directory "${OUTPUT_DIR}"
echo "[rsedit_second_gen] done: ${OUTPUT_DIR}"
