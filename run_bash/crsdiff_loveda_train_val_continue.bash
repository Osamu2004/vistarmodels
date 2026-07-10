#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Continue the existing val-only CRS-Diff result in place. Existing val images
# are skipped; train samples use train-prefixed names and are generated next.
VISTAR_EVAL_DIR="${VISTAR_EVAL_DIR:-/root/data/experiment/eval_loveda_gen_gen_only_step300000}" \
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/crsdiff_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_seed0}" \
MANIFEST="${MANIFEST:-/root/data/experiment/crsdiff_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_seed0/manifest_loveda_train_val.jsonl}" \
OVERWRITE="${OVERWRITE:-0}" \
VERIFY_SAMPLE_COUNTS="${VERIFY_SAMPLE_COUNTS:-1}" \
bash "${ROOT_DIR}/run_bash/crsdiff_loveda_gen.bash" "$@"
