"""Standalone RSKT-Seg evaluation on VISTAR's eight-class UAVid protocol."""

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
    shortest_edge_shape,
    strict_protocol_errors as _strict_protocol_errors,
    validate_model_classes,
)


OFFICIAL_UAVID_MODEL_CLASSES = (
    "Background clutter",
    "Building",
    "Road",
    "Tree",
    "Low vegetation",
    "Moving car",
    "Static car",
    "Human",
)
UAVID_CONTRACT = DatasetContract(
    dataset_key="uavid",
    model_classes=OFFICIAL_UAVID_MODEL_CLASSES,
    default_data_root="/root/data/OVSISBenchDataset/uavid",
    class_filename="uavid_8_classes.json",
    complete_protocol_name="official_RSKT_UAVid_all_VISTAR_8class",
    partial_protocol_name="RSKT_UAVid_8class_nonstandard_or_partial",
    primary_metric_description="miou_all_8_classes",
    auxiliary_metric_description="miou_foreground7",
    official_rskt_taxonomy_protocol=True,
    taxonomy_note=(
        "The official RSKT-Seg UAVid JSON already has exactly the eight "
        "VISTAR classes; Background clutter is class 0 and no support class "
        "is appended."
    ),
)
OFFICIAL_EXPECTED_SAMPLES = UAVID_CONTRACT.expected_samples


def validate_uavid_model_classes(model_classes: Sequence[str]) -> None:
    validate_model_classes(UAVID_CONTRACT, model_classes)


def strict_protocol_errors(
    args: argparse.Namespace,
    full_count: int,
) -> list[str]:
    return _strict_protocol_errors(UAVID_CONTRACT, args, full_count)


if __name__ == "__main__":
    run_evaluation(UAVID_CONTRACT)
