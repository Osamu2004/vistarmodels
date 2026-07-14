#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VISTAR_CODE="${VISTAR_CODE:-${ROOT_DIR}/../vistar}"
LOVEDA_ROOT="${LOVEDA_ROOT:-/root/data/LoveDA}"

# Existing completed Val output. Train predictions are appended here.
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/syntheticgen_loveda_val_512_exactmask_seed42}"
TRAIN_VIEW="${TRAIN_VIEW:-/root/data/experiment/vistar_loveda_train_512_minimal_for_syntheticgen}"

VISTAR_PYTHON_BIN="${VISTAR_PYTHON_BIN:-python}"
SYNTHETICGEN_PYTHON_BIN="${SYNTHETICGEN_PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SEED="${SEED:-42}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-2522}"
EXPECTED_VAL_SAMPLES="${EXPECTED_VAL_SAMPLES:-1669}"
EXPECTED_TOTAL_SAMPLES="${EXPECTED_TOTAL_SAMPLES:-4191}"
RUN_METRICS="${RUN_METRICS:-1}"
DIST_METRICS="${DIST_METRICS:-1}"
SEGGEN_METRICS="${SEGGEN_METRICS:-1}"
METRIC_GPU_IDS="${METRIC_GPU_IDS:-${CUDA_VISIBLE_DEVICES%%,*}}"

_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -d "${VISTAR_CODE}" ]]; then
  echo "[syntheticgen_train_append] Missing VISTAR_CODE: ${VISTAR_CODE}" >&2
  exit 1
fi
if [[ ! -d "${LOVEDA_ROOT}/Train" ]]; then
  echo "[syntheticgen_train_append] Missing LoveDA Train split: ${LOVEDA_ROOT}/Train" >&2
  exit 1
fi
if [[ ! -d "${OUTPUT_DIR}/pred_rgb" ]]; then
  echo "[syntheticgen_train_append] Existing Val pred_rgb directory is missing: ${OUTPUT_DIR}/pred_rgb" >&2
  exit 1
fi

mkdir -p \
  "${TRAIN_VIEW}/cond_mask" \
  "${TRAIN_VIEW}/gt_rgb"

echo "[syntheticgen_train_append] Preparing LoveDA Train with Vistar's 512x512 preprocessing"
echo "[syntheticgen_train_append] VISTAR_CODE=${VISTAR_CODE}"
echo "[syntheticgen_train_append] LOVEDA_ROOT=${LOVEDA_ROOT}"
echo "[syntheticgen_train_append] TRAIN_VIEW=${TRAIN_VIEW}"
echo "[syntheticgen_train_append] OUTPUT_DIR=${OUTPUT_DIR}"

cd "${VISTAR_CODE}"
PYTHONPATH="${VISTAR_CODE}:${PYTHONPATH:-}" \
"${VISTAR_PYTHON_BIN}" - "${LOVEDA_ROOT}" "${TRAIN_VIEW}" "${EXPECTED_TRAIN_SAMPLES}" <<'PY'
from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm

from eval_flux2_loveda import (
    LOVEDA_OFFICIAL_PALETTE_U8,
    _find_mask_path,
    _list_images,
    _resolve_loveda_split_dirs,
    _save_palette_mask,
    _save_rgb_tensor,
)
from eval_flux2_loveda_gen import _load_resized_loveda_pair, _record_name

data_root = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
expected = int(sys.argv[3])

dirs = {
    "cond_mask": output / "cond_mask",
    "gt_rgb": output / "gt_rgb",
}
for path in dirs.values():
    path.mkdir(parents=True, exist_ok=True)

def valid_512(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.size == (512, 512)
    except Exception:
        return False


pairs = []
domain_counts = {}
for split, domain, image_dir, mask_dir in _resolve_loveda_split_dirs(
    data_root, domains="both", splits="train"
):
    for image_path in _list_images(image_dir):
        pairs.append((split, domain, image_path, _find_mask_path(mask_dir, image_path)))
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

if len(pairs) != expected:
    raise RuntimeError(f"expected {expected} LoveDA Train samples, found {len(pairs)}")

written = 0
reused = 0
for split, domain, image_path, mask_path in tqdm(
    pairs,
    total=len(pairs),
    desc="Prepare LoveDA Train 512",
    unit="pair",
    dynamic_ncols=True,
):
        name = _record_name(split, domain, image_path)
        gt_path = dirs["gt_rgb"] / f"{name}_gt_rgb.png"
        cond_path = dirs["cond_mask"] / f"{name}_cond_mask.png"
        if valid_512(gt_path) and valid_512(cond_path):
            reused += 1
            continue
        image, mask, _ = _load_resized_loveda_pair(
            image_path=image_path,
            mask_path=mask_path,
            resize_size=512,
            ignore_index=255,
            reduce_zero_label=True,
        )
        _save_rgb_tensor(image, gt_path)
        _save_palette_mask(
            mask,
            cond_path,
            LOVEDA_OFFICIAL_PALETTE_U8,
        )
        written += 1

prepared = written + reused
print(f"[syntheticgen_train_append] prepared_train={prepared}")
print(f"[syntheticgen_train_append] written={written} reused={reused}")
print(f"[syntheticgen_train_append] domain_counts={domain_counts}")
if prepared != expected:
    raise RuntimeError(f"expected {expected} prepared Train pairs, found {prepared}")
PY

echo "[syntheticgen_train_append] Starting SyntheticGen Train generation"
cd "${ROOT_DIR}"
PYTHON_BIN="${SYNTHETICGEN_PYTHON_BIN}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
VISTAR_EVAL_DIR="${TRAIN_VIEW}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VERIFY_SAMPLE_COUNTS=0 \
SEED="${SEED}" \
OVERWRITE=0 \
bash "${ROOT_DIR}/run_bash/syntheticgen_loveda_gen.bash"

TRAIN_COUNT="$(find "${OUTPUT_DIR}/pred_rgb" -maxdepth 1 -type f -name 'Train_*_pred_rgb.png' | wc -l | tr -d ' ')"
TOTAL_COUNT="$(find "${OUTPUT_DIR}/pred_rgb" -maxdepth 1 -type f -name '*_pred_rgb.png' | wc -l | tr -d ' ')"
VAL_COUNT="$((TOTAL_COUNT - TRAIN_COUNT))"

echo "[syntheticgen_train_append] Train predictions: ${TRAIN_COUNT}/${EXPECTED_TRAIN_SAMPLES}"
echo "[syntheticgen_train_append] Val predictions:   ${VAL_COUNT}/${EXPECTED_VAL_SAMPLES}"
echo "[syntheticgen_train_append] Total predictions: ${TOTAL_COUNT}/${EXPECTED_TOTAL_SAMPLES}"

if [[ "${TRAIN_COUNT}" -ne "${EXPECTED_TRAIN_SAMPLES}" ]]; then
  echo "[syntheticgen_train_append] Train output is incomplete." >&2
  exit 1
fi
if [[ "${VAL_COUNT}" -ne "${EXPECTED_VAL_SAMPLES}" ]]; then
  echo "[syntheticgen_train_append] Existing Val output is incomplete." >&2
  exit 1
fi
if [[ "${TOTAL_COUNT}" -ne "${EXPECTED_TOTAL_SAMPLES}" ]]; then
  echo "[syntheticgen_train_append] Combined Train+Val output is incomplete." >&2
  exit 1
fi

if _truthy "${RUN_METRICS}"; then
  echo "[syntheticgen_train_append] Computing metrics with the existing Vistar evaluator"
  cd "${VISTAR_CODE}"
  GPU_IDS="${METRIC_GPU_IDS}" \
  INPUT_SIZE=512 \
  DIST_METRICS="${DIST_METRICS}" \
  SEGGEN_METRICS="${SEGGEN_METRICS}" \
  bash "${VISTAR_CODE}/run_bash/seg/compute_saved_loveda_gen_metrics.bash" "${OUTPUT_DIR}"
fi

echo "[syntheticgen_train_append] Complete: ${OUTPUT_DIR}"
