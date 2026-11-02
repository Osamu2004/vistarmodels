#!/usr/bin/env bash
# Official first query, single A/B direction; retain the user's 512 source tiles.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${DATASETS:-levircd}" != "levircd" ]]; then
  echo "This verification entry evaluates LEVIR-CD only." >&2
  exit 2
fi
export DATASETS=levircd
export DATA_ROOTS_JSON="${DATA_ROOTS_JSON:-${ROOT_DIR}/configs/rsunivlm/paper_data_roots.json}"
export FORMAT_RETRIES="${FORMAT_RETRIES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/rsunivlm_levircd_forward_tile512_bs${BATCH_SIZE:-16}_v1}"
exec bash "${ROOT_DIR}/run_bash/rsunivlm_paper_queue.bash" "$@"
