#!/usr/bin/env bash
# Explicit environment/source/weight setup only. Does not launch inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${CONDA_SH:-/root/anaconda3/etc/profile.d/conda.sh}"
base_env="${BASE_ENV:-vistar_longcat}"
dynamic_env="${DYNAMIC_ENV:-vistar_dynamicearth}"
opendpr_env="${OPENDPR_ENV:-vistar_opendpr}"
case "${METHOD:-both}" in
  both) methods=(dynamicearth opendpr) ;;
  dynamicearth|opendpr) methods=("$METHOD") ;;
  *) echo "METHOD must be both, dynamicearth or opendpr" >&2; exit 2 ;;
esac
if [[ "$dynamic_env" == "$opendpr_env" || "$base_env" == "$dynamic_env" || "$base_env" == "$opendpr_env" ]]; then
  echo "Base environment and both target environments must have different names" >&2
  exit 2
fi
conda activate "$base_env"
base_python="$(command -v python)"
"$base_python" -c 'import importlib.metadata as m; print("Base Torch:",m.version("torch"),"torchvision:",m.version("torchvision"))'
setup_root="${SETUP_ROOT:-/root/data/experiment/ovcd_env_setup_v1}"
for method in "${methods[@]}"; do
  target="$opendpr_env"
  if [[ "$method" == dynamicearth ]]; then target="$dynamic_env"; fi
  if conda env list --json | "$base_python" -c 'import json,pathlib,sys; sys.exit(0 if any(pathlib.Path(p).name==sys.argv[1] for p in json.load(sys.stdin)["envs"]) else 1)' "$target"; then
    echo "[OVCD setup] Reuse environment: $target"
  else
    echo "[OVCD setup] Clone $base_env -> $target"
    conda create --name "$target" --clone "$base_env" --yes
  fi
  conda activate "$target"
  state_dir="$setup_root/$target"
  mkdir -p "$state_dir"
  python tools/ovcd_torch_guard.py capture --state "$state_dir/torch_before.json" \
    --constraints "$state_dir/torch_constraints.txt"
  # Constraints propagate into every nested pip call, including SAM/SimFeatUp.
  # Setup needs network access; inference can reuse the downloaded files later.
  PIP_CONSTRAINT="$state_dir/torch_constraints.txt" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    MAX_JOBS="${MAX_JOBS:-2}" METHOD="$method" PYTHON_BIN="$(command -v python)" \
    bash scripts/bootstrap_ovcd_paper.sh
  python tools/ovcd_torch_guard.py verify --state "$state_dir/torch_before.json"
  python -u tools/check_ovcd_paper_deps.py --method "$method"
done
echo "[OVCD setup] Finished. No model weights loaded for inference and no evaluation started."
echo "[OVCD setup] Next: PLAN_ONLY=1, then a separate MAX_SAMPLES=2 CUDA smoke, then the full queue."
