#!/usr/bin/env bash
# Explicit user-run sequence: setup -> data plans -> two-pair CUDA smoke -> full scoring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for value in "${SKIP_SETUP:-0}" "${SETUP_ONLY:-0}"; do
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "Setup flags must be0/1" >&2; exit 2; }
done
if [[ "${SKIP_SETUP:-0}" == 0 ]]; then
  METHOD=both bash scripts/setup_ovcd_envs.bash
fi
if [[ "${SETUP_ONLY:-0}" == 1 ]]; then exit 0; fi
source "${CONDA_SH:-/root/anaconda3/etc/profile.d/conda.sh}"
formal_root="${OUTPUT_ROOT:-/root/data/experiment/ovcd_paper_levir128_second_val593_tile512_v1}"
smoke_root="${SMOKE_OUTPUT_ROOT:-/root/data/experiment/ovcd_paper_levir_second_smoke2_v1}"
[[ "$formal_root" != "$smoke_root" ]] || { echo "Smoke and full outputs must differ" >&2; exit 2; }
PLAN_ONLY=1 MAX_SAMPLES=0 OUTPUT_ROOT="$formal_root" bash run_bash/ovcd_paper_queue.bash
PLAN_ONLY=0 MAX_SAMPLES=2 OUTPUT_ROOT="$smoke_root" bash run_bash/ovcd_paper_queue.bash
PLAN_ONLY=0 MAX_SAMPLES=0 OUTPUT_ROOT="$formal_root" bash run_bash/ovcd_paper_queue.bash
