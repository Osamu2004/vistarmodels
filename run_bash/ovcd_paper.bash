#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
method="${METHOD:-opendpr}"
case "$method" in
  opendpr) source_root="${OPENDPR_ROOT:-$PWD/third_party/OpenDPR}" ;;
  dynamicearth) source_root="${DYNAMIC_EARTH_ROOT:-$PWD/third_party/DynamicEarth}" ;;
  *) echo "Unknown METHOD: $method" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
IFS=',' read -r -a datasets <<< "${DATASETS:-levircd,second}"
args=(--method "$method" --datasets "${datasets[@]}" --source_root "$source_root"
  --levircd_root "${LEVIR_ROOT:-/root/data/LEVIR-CD}" --second_root "${SECOND_ROOT:-/root/data/second_dataset}"
  --output_dir "${OUTPUT_DIR:-/root/data/experiment/${method}_levir128_second_val593_tile512_v1}"
  --sam_checkpoint "${SAM_CHECKPOINT:-/root/data/weight/dynamicearth/sam_vit_h_4b8939.pth}"
  --dinov2_dir "${DINOV2_DIR:-/root/data/weight/dinov2-base}"
  --dino_repo "${DINO_REPO:-$PWD/third_party/DINO}"
  --dino_checkpoint "${DINO_CHECKPOINT:-/root/data/weight/dynamicearth/dino_vitbase16_pretrain.pth}"
  --segearth_weight "${SEGEARTH_WEIGHT:-/root/data/weight/dynamicearth/xclip_jbu_one_million_aid.ckpt}"
  --tile_size "${TILE_SIZE:-512}" --num_workers "${NUM_WORKERS:-2}" --max_samples "${MAX_SAMPLES:-0}")
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
  exec "${PYTHON_BIN:-python}" -u baselines/ovcd/evaluate.py "${args[@]}" --plan_only "$@"
fi
"${PYTHON_BIN:-python}" -u tools/check_ovcd_paper_deps.py --method "$method" --source_root "$source_root" --cuda
exec "${PYTHON_BIN:-python}" -u baselines/ovcd/evaluate.py "${args[@]}" "$@"
