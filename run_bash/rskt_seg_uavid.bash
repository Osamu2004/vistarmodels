#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_PORT="${MASTER_PORT:-29636}"
export RSKT_DATASET_KEY="uavid"
export RSKT_DATASET_NAME="UAVid"
export RSKT_DEFAULT_DATA_ROOT="/root/data/OVSISBenchDataset/uavid"
export RSKT_EVALUATOR="eval_rskt_seg_uavid.py"
export RSKT_CLASS_FILENAME="uavid_8_classes.json"
export RSKT_EXPECTED_SAMPLES="270"
export RSKT_OUTPUT_SLUG="uavid_vistar8_all"
export RSKT_CLASS_DESCRIPTION="8 (class0=Background clutter)"
export RSKT_METRIC_DESCRIPTION="mIoU(all 8 primary),mIoU(foreground 7 auxiliary),mACC,mF1,pixel_accuracy,WFm"
export RSKT_TAXONOMY_DESCRIPTION="official RSKT and VISTAR eight-class order; moving/static car separate"

exec bash "${SCRIPT_DIR}/rskt_seg_multiclass.bash" "$@"
