#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:?set local SD snapshot}"
CONTROLNET="${CONTROLNET:?set trained ControlNet checkpoint}"
VARIANT="${VARIANT:-sd15}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/controlnet_${VARIANT}_second_test_256}"
CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" "${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}" -u \
  "${ROOT_DIR}/baselines/controlnet/run_controlnet_second.py" \
  --manifest /data/vistar/runs/paper_baselines/data/second/test.jsonl \
  --base_model "${BASE_MODEL}" --controlnet "${CONTROLNET}" --output_dir "${OUTPUT_DIR}" "${@}"
