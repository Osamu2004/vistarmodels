#!/usr/bin/env bash
# Six independent semantic change classes, one image order, no reverse query.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${DATASETS:-second}" != "second" ]]; then
  echo "This verification entry evaluates SECOND only." >&2
  exit 2
fi
export DATASETS=second
export DATA_ROOTS_JSON="${DATA_ROOTS_JSON:-${ROOT_DIR}/configs/rsunivlm/paper_data_roots.json}"
export FORMAT_RETRIES="${FORMAT_RETRIES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_second_undirected6_tile512_bs${BATCH_SIZE:-16}_v1}"
exec bash "${ROOT_DIR}/run_bash/rsunivlm_paper_queue.bash" "$@"
