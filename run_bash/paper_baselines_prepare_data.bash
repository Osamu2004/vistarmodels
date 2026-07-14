#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
LOVEDA_ROOT="${LOVEDA_ROOT:-/data/vistar/datasets/atomic/loveda/source_data}"
SECOND_ROOT="${SECOND_ROOT:-/data/vistar/datasets/second_semantic_manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/paper_baselines/data}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_paper_baseline_data.py" \
  --loveda_root "${LOVEDA_ROOT}" \
  --second_root "${SECOND_ROOT}" \
  --output "${OUTPUT_DIR}" "${@}"
