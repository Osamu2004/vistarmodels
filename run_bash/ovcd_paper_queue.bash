#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IFS=',' read -r -a gpus <<< "${GPU_IDS:-0,1}"
if (( ${#gpus[@]} != 2 )) || [[ "${gpus[0]}" == "${gpus[1]}" ]]; then
  echo "Provide two different GPU IDs: GPU_IDS=0,1" >&2; exit 2
fi
methods=(dynamicearth opendpr)
envs=("${DYNAMIC_ENV:-vistar_dynamicearth}" "${OPENDPR_ENV:-vistar_opendpr}")
root="${OUTPUT_ROOT:-/root/data/experiment/ovcd_paper_levir128_second_val593_tile512_v1}"
# Preflight both methods and all data populations before the first GPU model.
for i in 0 1; do
  METHOD="${methods[$i]}" GPU_IDS="${gpus[$i]}" PLAN_ONLY=1 OUTPUT_DIR="$root/${methods[$i]}" \
    conda run --no-capture-output -n "${envs[$i]}" bash run_bash/ovcd_paper.bash
done
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then exit 0; fi
pids=()
for i in 0 1; do
  METHOD="${methods[$i]}" GPU_IDS="${gpus[$i]}" PLAN_ONLY=0 OUTPUT_DIR="$root/${methods[$i]}" \
    conda run --no-capture-output -n "${envs[$i]}" bash run_bash/ovcd_paper.bash &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
