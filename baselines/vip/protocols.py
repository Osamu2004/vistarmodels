"""CPU-testable class-vocabulary and sliding-window rules for VIP."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


DATASET_BACKGROUND_INDICES: dict[str, int | None] = {
    "loveda": 0,
    "flair": None,
    "uavid": 0,
    "xbd_pre": 0,
    "chn6_cug": 0,
}
PREDICTION_IGNORE_INDEX = 255


def normalize_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()


def read_vip_class_groups(path: str | Path) -> tuple[tuple[str, ...], ...]:
    """Read VIP's one-output-class-per-line alias format."""

    source = normalize_path(path)
    groups: list[tuple[str, ...]] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            raise ValueError(
                "VIP treats every physical line as an output class; blank and "
                f"comment lines are not allowed: {source}:{line_number}"
            )
        if "," in line and ", " not in line:
            raise ValueError(
                "VIP aliases must use a comma followed by one space: "
                f"{source}:{line_number}"
            )
        aliases = tuple(part.strip() for part in line.split(", "))
        if any(not alias for alias in aliases):
            raise ValueError(f"Empty VIP alias in {source}:{line_number}")
        groups.append(aliases)
    if not groups:
        raise ValueError(f"VIP class vocabulary is empty: {source}")
    return tuple(groups)


def flatten_class_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    aliases: list[str] = []
    class_indices: list[int] = []
    for class_id, group in enumerate(groups):
        if not group:
            raise ValueError(f"VIP class {class_id} has no aliases")
        aliases.extend(str(alias) for alias in group)
        class_indices.extend([class_id] * len(group))
    return tuple(aliases), tuple(class_indices)


def resolve_low_confidence_policy(
    dataset: str,
    requested_action: str,
) -> tuple[str, int, int | None]:
    """Resolve the released VIP background/ignore threshold behavior."""

    if dataset not in DATASET_BACKGROUND_INDICES:
        raise ValueError(f"Unsupported VIP dataset: {dataset}")
    if requested_action not in {"auto", "background", "ignore"}:
        raise ValueError(
            "VIP low-confidence action must be auto, background, or ignore"
        )
    background_index = DATASET_BACKGROUND_INDICES[dataset]
    resolved_action = requested_action
    if resolved_action == "auto":
        resolved_action = (
            "background" if background_index is not None else "ignore"
        )
    if resolved_action == "background":
        if background_index is None:
            raise ValueError(
                f"{dataset} has no evaluated background class; use "
                "low-confidence action 'ignore'"
            )
        return resolved_action, int(background_index), None
    return (
        resolved_action,
        PREDICTION_IGNORE_INDEX,
        PREDICTION_IGNORE_INDEX,
    )


def compute_square_padding(
    height: int,
    width: int,
    crop_size: int,
) -> tuple[int, int, int, int]:
    """Symmetrically pad an edge crop to VIP's fixed square token grid."""

    if height <= 0 or width <= 0 or crop_size <= 0:
        raise ValueError("height, width, and crop_size must be positive")
    if height > crop_size or width > crop_size:
        raise ValueError(
            f"Cannot pad {(height, width)} down to crop_size={crop_size}"
        )
    pad_h = crop_size - height
    pad_w = crop_size - width
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return left, right, top, bottom


def sliding_window_boxes(
    height: int,
    width: int,
    crop_size: int,
    stride: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Match VIP's overlap-and-shift sliding-window coordinates."""

    if min(height, width, crop_size, stride) <= 0:
        raise ValueError("spatial sizes and stride must be positive")
    h_grids = max(height - crop_size + stride - 1, 0) // stride + 1
    w_grids = max(width - crop_size + stride - 1, 0) // stride + 1
    boxes: list[tuple[int, int, int, int]] = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_size, height)
            x2 = min(x1 + crop_size, width)
            y1 = max(y2 - crop_size, 0)
            x1 = max(x2 - crop_size, 0)
            boxes.append((y1, y2, x1, x2))
    return tuple(boxes)
