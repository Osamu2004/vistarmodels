"""Explicit VISTAR fusion of saved binary queries, independent of GT/model code."""
from __future__ import annotations

import numpy as np

DATASETS = ("loveda", "flair", "uavid")
POLICY = "ascending_class_id_last_positive_wins_v1"
TEXT2SEG_PAPER = "https://arxiv.org/abs/2304.10597"
TEXT2SEG_CODE = "https://github.com/zhangjielu321/Text2Seg/blob/dbf6d3df026b5865517defec56decd47ce95226b/text2seg/text2seg.py"


def settings(dataset, classes):
    if dataset not in DATASETS:
        raise ValueError("Exclusive fusion is only defined for LoveDA/FLAIR/UAVid")
    return dict(policy=POLICY, class_order=[dict(id=i, name=name) for i, name in enumerate(classes)],
        overlap="visit ascending class IDs; each positive mask overwrites earlier IDs",
        uncovered_id=len(classes) if dataset == "flair" else 0,
        uncovered_name="prediction-only background" if dataset == "flair" else classes[0],
        uncovered_scoring="retain valid GT pixels; never use GT ignore as a prediction fallback",
        spatial_scope="native-resolution masks after existing source512 restoration/stitching",
        gt_used_for_fusion=False, confidence_available=False, official_text2seg_fusion=False,
        provenance="Text2Seg-inspired per-class interface; user-selected fixed-order VISTAR adaptation",
        references=[TEXT2SEG_PAPER, TEXT2SEG_CODE])


def fuse(mask_pairs, *, num_classes, uncovered_id):
    """Stream all fixed class masks; missing classes are an error, never background.

    No confidence/area/GT tie breaking. ID 0 (when present) is written first,
    so positive foreground queries can overwrite the explicit background query.
    """
    if not 1 < num_classes < 255 or not 0 <= uncovered_id <= num_classes:
        raise ValueError("Invalid class count or uncovered prediction ID")
    prediction = count = None
    areas = {}
    for expected, (class_id, mask) in enumerate(mask_pairs):
        if class_id != expected or class_id >= num_classes:
            raise ValueError("Require exactly one query per class in ascending ID order")
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype not in (np.dtype("uint8"), np.dtype("bool")) or not np.all(array < 2):
            raise ValueError("Expected native-size binary uint8/bool masks, encoded 0/1")
        if not all(array.shape):
            raise ValueError("Empty source query mask")
        if prediction is None:
            prediction = np.full(array.shape, uncovered_id, np.uint8)
            count = np.zeros(array.shape, np.uint8)
        if array.shape != prediction.shape:
            raise ValueError("Source query extents differ; resizing during fusion is forbidden")
        positive = array.astype(bool, copy=False)
        prediction[positive] = class_id
        count += array
        areas[str(class_id)] = int(np.count_nonzero(positive))
    if len(areas) != num_classes:
        raise ValueError("Missing class query masks; refusing to fill predictions")
    return prediction, dict(policy=POLICY, native_pixels=int(prediction.size),
        uncovered_pixels=int(np.count_nonzero(count == 0)),
        overlapping_pixels=int(np.count_nonzero(count > 1)),
        per_class_positive_pixels=areas, uncovered_id=uncovered_id)
