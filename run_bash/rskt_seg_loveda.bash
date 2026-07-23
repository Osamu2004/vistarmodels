#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29637}"
export RSKT_DATASET_KEY="loveda"
export RSKT_DATASET_NAME="LoveDA"
export RSKT_DEFAULT_DATA_ROOT="/root/data/LoveDA"
export RSKT_EVALUATOR="eval_rskt_seg_loveda.py"
export RSKT_CLASS_FILENAME="loveda_7_classes.json"
export RSKT_EXPECTED_SAMPLES="1669"
export RSKT_OUTPUT_SLUG="loveda_val_vistar7"
export RSKT_CLASS_DESCRIPTION="7 (background,building,road,water,barren,forest,agriculture)"
export RSKT_METRIC_DESCRIPTION="mIoU(all 7 primary),mACC,mF1,pixel_accuracy,per-class IoU/F1/ACC,WFm"
export RSKT_TAXONOMY_DESCRIPTION="VISTAR seven-class protocol; raw0 ignored; raw1..7 mapped to eval0..6; official RSKT no-data channel omitted"

exec bash "${SCRIPT_DIR}/rskt_seg_multiclass.bash" "$@"
