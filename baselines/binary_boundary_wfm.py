"""IDGBR-compatible weighted F-measure for binary semantic boundaries.

The paper metric is computed per image on discrete, full-resolution labels:
Sobel boundary extraction, one 3x3 dilation, and Margolin's weighted
F-measure for the non-edge and edge classes.  Dataset aggregation first
averages each class over images where that class is present in the ground
truth boundary map, then takes the smaller positive class mean.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    sobel,
)

try:
    import cv2
except ImportError:  # pragma: no cover - production environments require OpenCV
    cv2 = None


WFM_EDGE_SIZE = 3
WFM_BETA = 1.0
WFM_SIGMA = 5.0
WFM_PROTOCOL_NAME = "IDGBR_3px_boundary_WFm"
WFM_IGNORE_MARGIN = 2


def _validate_binary_mask(name: str, value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape={mask.shape}")
    if not bool(np.all((mask == 0) | (mask == 1))):
        unique = np.unique(mask)
        preview = unique[:8].tolist()
        raise ValueError(
            f"{name} must contain only discrete binary labels 0/1, got {preview}"
        )
    return mask.astype(np.uint8, copy=False)


def _validate_semantic_mask(
    name: str,
    value: np.ndarray,
    *,
    ignore_index: int,
    num_classes: int | None,
) -> np.ndarray:
    """Validate a discrete semantic mask without changing its class IDs."""

    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape={mask.shape}")
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError(f"{name} must contain numeric labels, got {mask.dtype}")
    if np.issubdtype(mask.dtype, np.floating):
        if not bool(np.all(np.isfinite(mask))):
            raise ValueError(f"{name} contains NaN or Inf labels")
        rounded = np.rint(mask)
        if not bool(np.array_equal(mask, rounded)):
            raise ValueError(f"{name} must contain integral semantic labels")
        mask = rounded
    labels = mask.astype(np.int64, copy=False)
    if num_classes is not None:
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        valid_label = (labels >= 0) & (labels < int(num_classes))
    else:
        # Boundary extraction stores class IDs as uint8, so reserve 255 for
        # the conventional ignore label even when no taxonomy size is given.
        valid_label = (labels >= 0) & (labels < 255)
    allowed = valid_label | (labels == int(ignore_index))
    if not bool(np.all(allowed)):
        unexpected = np.unique(labels[~allowed])[:8].tolist()
        expected = (
            f"0..{int(num_classes) - 1} or {ignore_index}"
            if num_classes is not None
            else f"0..254 or {ignore_index}"
        )
        raise ValueError(
            f"{name} contains invalid semantic labels {unexpected}; expected {expected}"
        )
    return labels


def _extract_boundary(label: np.ndarray) -> np.ndarray:
    """Match IDGBR's cv2 Sobel followed by one 3x3 dilation."""

    values = np.asarray(label, dtype=np.uint8)
    kernel = np.ones((WFM_EDGE_SIZE, WFM_EDGE_SIZE), dtype=np.uint8)
    if cv2 is not None:
        sobel_x = cv2.Sobel(values, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(values, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.asarray(
            (sobel_x * sobel_x + sobel_y * sobel_y) > 0,
            dtype=np.uint8,
        )
        return np.asarray(
            cv2.dilate(edge, kernel, iterations=1),
            dtype=bool,
        )

    # scipy ``mirror`` matches OpenCV's default BORDER_REFLECT_101.  Only the
    # nonzero gradient support matters, so the Sobel scale difference cancels.
    values_float = values.astype(np.float64)
    sobel_x = sobel(values_float, axis=1, mode="mirror")
    sobel_y = sobel(values_float, axis=0, mode="mirror")
    edge = (sobel_x * sobel_x + sobel_y * sobel_y) > 0
    return np.asarray(
        binary_dilation(
            edge,
            structure=kernel.astype(bool),
            iterations=1,
        ),
        dtype=bool,
    )


def _weighted_f_beta_score(
    candidate: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Margolin weighted F-measure with IDGBR's beta=1 and sigma=5."""

    candidate_float = np.asarray(candidate, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    gt_mask = np.asarray(ground_truth, dtype=bool) & valid_mask
    if not bool(np.any(gt_mask)):
        raise ValueError("Weighted F-measure requires a present GT class")

    gt = gt_mask.astype(np.float64)
    not_gt_mask = ~gt_mask
    error = np.abs(candidate_float - gt)
    distance, nearest = distance_transform_edt(
        not_gt_mask,
        return_indices=True,
    )

    propagated = np.array(error)
    propagated[not_gt_mask] = error[
        nearest[0, not_gt_mask],
        nearest[1, not_gt_mask],
    ]
    smoothed = gaussian_filter(
        propagated,
        sigma=WFM_SIGMA,
        truncate=3.0 / WFM_SIGMA,
        mode="constant",
        cval=0.0,
    )
    minimum_error = np.minimum(
        error,
        smoothed,
        where=gt_mask,
        out=np.array(error),
    )

    spatial_weight = np.ones(gt.shape, dtype=np.float64)
    spatial_weight[not_gt_mask] = 2.0 - np.exp(
        np.log(1.0 - 0.5) / 5.0 * distance[not_gt_mask]
    )
    weighted_error = minimum_error * spatial_weight

    negative_mask = valid_mask & not_gt_mask
    true_positive_w = float(np.sum(gt_mask)) - float(
        np.sum(weighted_error[gt_mask])
    )
    false_positive_w = float(np.sum(weighted_error[negative_mask]))
    recall = 1.0 - float(np.mean(weighted_error[gt_mask]))
    eps = float(np.spacing(1))
    precision = true_positive_w / (
        eps + true_positive_w + false_positive_w
    )
    # Preserve the public IDGBR expression.  WFM_BETA is fixed to one.
    score = (1.0 + WFM_BETA**2) * (recall * precision) / (
        eps + recall + WFM_BETA * precision
    )
    return float(score)


def score_binary_boundary_wfm(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one binary prediction/target pair at its native image extent."""

    prediction_mask = _validate_binary_mask("prediction", prediction)
    target_mask = _validate_binary_mask("target", target)
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction/target shape mismatch for boundary WFm: "
            f"prediction={prediction_mask.shape}, target={target_mask.shape}"
        )

    if valid is None:
        valid_mask = np.ones(target_mask.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != target_mask.shape:
            raise ValueError(
                "Validity-mask shape mismatch for boundary WFm: "
                f"valid={valid_mask.shape}, target={target_mask.shape}"
            )
    if not bool(np.any(valid_mask)):
        raise ValueError("Boundary WFm requires at least one valid pixel")

    target_for_edge = np.array(target_mask, copy=True)
    prediction_for_edge = np.array(prediction_mask, copy=True)
    target_for_edge[~valid_mask] = 0
    prediction_for_edge[~valid_mask] = 0
    target_edge = _extract_boundary(target_for_edge) & valid_mask
    prediction_edge = _extract_boundary(prediction_for_edge) & valid_mask

    scores: dict[int, float] = {}
    for class_id in np.unique(target_edge[valid_mask]).tolist():
        class_id = int(class_id)
        scores[class_id] = _weighted_f_beta_score(
            prediction_edge == bool(class_id),
            target_edge == bool(class_id),
            valid_mask,
        )

    present_scores = list(scores.values())
    return {
        "wfm_3px_nonedge": scores.get(0),
        "wfm_3px_edge": scores.get(1),
        "wfm_3px_mean_present": float(np.mean(present_scores)),
        "wfm_3px_min_present": float(np.min(present_scores)),
        "wfm_3px_valid_pixels": int(np.sum(valid_mask)),
        "wfm_3px_gt_edge_pixels": int(np.sum(target_edge & valid_mask)),
        "wfm_3px_pred_edge_pixels": int(
            np.sum(prediction_edge & valid_mask)
        ),
    }


def score_semantic_boundary_wfm(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    ignore_index: int = 255,
    ignore_margin: int = WFM_IGNORE_MARGIN,
    num_classes: int | None = None,
    allow_prediction_ignore: bool = False,
) -> dict[str, Any]:
    """Score semantic boundaries while excluding target ignore support.

    The semantic class IDs are collapsed only *after* Sobel boundary
    extraction: every transition between two retained semantic labels is an
    edge.  Target ignore pixels and a configurable surrounding margin are
    excluded so that boundaries against classes outside the evaluated
    taxonomy cannot affect the score.  FLAIR uses ``ignore_margin=2``: one
    pixel for the Sobel support and one for the 3x3 boundary dilation.
    """

    if int(ignore_margin) < 0:
        raise ValueError(f"ignore_margin must be non-negative, got {ignore_margin}")
    prediction_mask = _validate_semantic_mask(
        "prediction",
        prediction,
        ignore_index=int(ignore_index),
        num_classes=num_classes,
    )
    target_mask = _validate_semantic_mask(
        "target",
        target,
        ignore_index=int(ignore_index),
        num_classes=num_classes,
    )
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction/target shape mismatch for semantic boundary WFm: "
            f"prediction={prediction_mask.shape}, target={target_mask.shape}"
        )

    valid = target_mask != int(ignore_index)
    if not bool(np.any(valid)):
        raise ValueError("Semantic boundary WFm requires at least one valid GT pixel")
    invalid_prediction = valid & (prediction_mask == int(ignore_index))
    if bool(np.any(invalid_prediction)) and not bool(allow_prediction_ignore):
        raise ValueError(
            "Prediction uses ignore_index on "
            f"{int(np.sum(invalid_prediction))} valid GT pixels"
        )

    if int(ignore_margin) > 0 and bool(np.any(~valid)):
        ignore_support = binary_dilation(
            ~valid,
            structure=np.ones(
                (2 * int(ignore_margin) + 1, 2 * int(ignore_margin) + 1),
                dtype=bool,
            ),
            iterations=1,
        )
        valid_eval = valid & ~ignore_support
    else:
        valid_eval = valid
    if not bool(np.any(valid_eval)):
        raise ValueError(
            "No valid evaluation pixels remain after semantic ignore masking"
        )

    target_for_edge = np.array(target_mask, copy=True)
    prediction_for_edge = np.array(prediction_mask, copy=True)
    # Match the saved FLAIR reference exactly: only the original GT-ignore
    # region is filled before Sobel; the wider support is applied afterwards.
    target_for_edge[~valid] = 0
    prediction_for_edge[~valid] = 0
    target_edge = _extract_boundary(target_for_edge) & valid_eval
    prediction_edge = _extract_boundary(prediction_for_edge) & valid_eval

    scores: dict[int, float] = {}
    for class_id in np.unique(target_edge[valid_eval]).tolist():
        class_id = int(class_id)
        scores[class_id] = _weighted_f_beta_score(
            prediction_edge == bool(class_id),
            target_edge == bool(class_id),
            valid_eval,
        )
    present_scores = list(scores.values())
    return {
        "wfm_3px_nonedge": scores.get(0),
        "wfm_3px_edge": scores.get(1),
        "wfm_3px_mean_present": float(np.mean(present_scores)),
        "wfm_3px_min_present": float(np.min(present_scores)),
        "wfm_3px_valid_pixels": int(np.sum(valid_eval)),
        "wfm_3px_gt_ignore_pixels": int(np.sum(~valid)),
        "wfm_3px_pred_rejected_pixels": int(np.sum(invalid_prediction)),
        "wfm_3px_ignore_excluded_pixels": int(np.sum(~valid_eval)),
        "wfm_3px_ignore_margin": int(ignore_margin),
        "wfm_3px_gt_edge_pixels": int(np.sum(target_edge & valid_eval)),
        "wfm_3px_pred_edge_pixels": int(
            np.sum(prediction_edge & valid_eval)
        ),
    }


def aggregate_binary_boundary_wfm(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-image class WFm values using the public IDGBR rule."""

    if not rows:
        raise ValueError("Cannot aggregate boundary WFm over zero images")

    class_specs = (
        (0, "non_edge", "wfm_3px_nonedge"),
        (1, "edge", "wfm_3px_edge"),
    )
    sums = {class_id: 0.0 for class_id, _, _ in class_specs}
    counts = {class_id: 0 for class_id, _, _ in class_specs}
    for row in rows:
        for class_id, _, key in class_specs:
            value = row.get(key)
            if value is not None:
                value_float = float(value)
                if not math.isfinite(value_float):
                    raise ValueError(f"Non-finite per-image boundary WFm: {value}")
                sums[class_id] += value_float
                counts[class_id] += 1

    means = {
        class_id: (
            sums[class_id] / counts[class_id]
            if counts[class_id]
            else 0.0
        )
        for class_id, _, _ in class_specs
    }
    # Match IDGBR's filtering of absent/non-positive aggregate class scores.
    positive_means = [
        means[class_id]
        for class_id, _, _ in class_specs
        if means[class_id] > 0
    ]
    mean_wf = float(np.mean(positive_means)) if positive_means else 0.0
    min_wf = float(np.min(positive_means)) if positive_means else 0.0

    return {
        "wfm_3px": min_wf,
        "wfm_3px_percent": 100.0 * min_wf,
        "wfm_3px_nonedge": means[0],
        "wfm_3px_edge": means[1],
        "wfm_3px_mean": mean_wf,
        "wfm_3px_nonedge_num_images": counts[0],
        "wfm_3px_edge_num_images": counts[1],
        "wfm_edge_size": WFM_EDGE_SIZE,
        "wfm_beta": WFM_BETA,
        "wfm_aggregate": "min",
        "wfm_num_samples": len(rows),
        "wfm_protocol": {
            "name": WFM_PROTOCOL_NAME,
            "edge_extraction": "Sobel ksize=3, gradient magnitude > 0",
            "dilation_kernel": [WFM_EDGE_SIZE, WFM_EDGE_SIZE],
            "dilation_iterations": 1,
            "weighted_f_measure": "Margolin",
            "beta": WFM_BETA,
            "sigma": WFM_SIGMA,
            "class_names": ["non_edge", "edge"],
            "per_class_aggregation": "mean_over_gt_present_images",
            "aggregate": (
                "minimum_of_positive_image_averaged_binary_class_scores"
            ),
        },
    }


__all__ = [
    "WFM_BETA",
    "WFM_EDGE_SIZE",
    "WFM_PROTOCOL_NAME",
    "WFM_SIGMA",
    "WFM_IGNORE_MARGIN",
    "aggregate_binary_boundary_wfm",
    "score_binary_boundary_wfm",
    "score_semantic_boundary_wfm",
]
