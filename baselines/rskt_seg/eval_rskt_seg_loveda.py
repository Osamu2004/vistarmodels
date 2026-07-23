"""Standalone RSKT-Seg evaluation on VISTAR's seven-class LoveDA protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.rskt_seg.eval_rskt_seg_multiclass import (
    DatasetContract,
    run_evaluation,
    strict_protocol_errors as _strict_protocol_errors,
    validate_model_classes,
)


LOVEDA_MODEL_CLASSES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
)
LOVEDA_CONTRACT = DatasetContract(
    dataset_key="loveda",
    model_classes=LOVEDA_MODEL_CLASSES,
    default_data_root="/root/data/LoveDA",
    class_filename="loveda_7_classes.json",
    complete_protocol_name="RSKT_LoveDA_Val_VISTAR_7class",
    partial_protocol_name="RSKT_LoveDA_7class_nonstandard_or_partial",
    primary_metric_description="miou_all_7_evaluation_classes",
    auxiliary_metric_description=None,
    official_rskt_taxonomy_protocol=False,
    taxonomy_note=(
        "VISTAR alignment deliberately removes the official RSKT-Seg "
        "no-data text channel: raw LoveDA ID 0 is ignored, raw IDs 1..7 map "
        "to evaluation IDs 0..6, and only those seven classes compete."
    ),
)
OFFICIAL_EXPECTED_SAMPLES = LOVEDA_CONTRACT.expected_samples


def validate_loveda_model_classes(model_classes: Sequence[str]) -> None:
    validate_model_classes(LOVEDA_CONTRACT, model_classes)


def strict_protocol_errors(
    args: argparse.Namespace,
    full_count: int,
) -> list[str]:
    return _strict_protocol_errors(LOVEDA_CONTRACT, args, full_count)


if __name__ == "__main__":
    run_evaluation(LOVEDA_CONTRACT)
