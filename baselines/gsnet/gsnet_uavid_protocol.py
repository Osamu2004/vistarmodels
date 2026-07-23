"""CPU-testable GSNet-to-UAVid evaluation contract."""

from __future__ import annotations

import argparse
from typing import Sequence


UAVID_MODEL_CLASSES = (
    "background clutter",
    "building",
    "road",
    "tree",
    "low vegetation",
    "moving car",
    "static car",
    "human",
)
UAVID_EXPECTED_SAMPLES = 270
UAVID_TILE_SIZE = 512
UAVID_MODEL_INPUT_SIZE = 512
UAVID_NUM_LAYERS = 2
UAVID_PROMPT_ENSEMBLE = "single"
UAVID_AMP = "fp32"


def validate_uavid_model_classes(model_classes: Sequence[str]) -> None:
    """Require the exact VISTAR eight-class order and no support channel."""

    actual = tuple(str(value).strip().casefold() for value in model_classes)
    if actual != UAVID_MODEL_CLASSES:
        raise ValueError(
            "GSNet UAVid must use exactly the eight VISTAR classes in order "
            f"{list(UAVID_MODEL_CLASSES)}; got {list(actual)}. No extra "
            "negative, background, unknown, other, or support class is "
            "allowed."
        )


def tile_grid_shape(
    height: int,
    width: int,
    tile_size: int,
) -> tuple[int, int]:
    if min(height, width, tile_size) <= 0:
        raise ValueError("height, width, and tile_size must be positive")
    return (
        (int(height) + int(tile_size) - 1) // int(tile_size),
        (int(width) + int(tile_size) - 1) // int(tile_size),
    )


def tile_grid_count(height: int, width: int, tile_size: int) -> int:
    rows, columns = tile_grid_shape(height, width, tile_size)
    return rows * columns


def strict_protocol_errors(
    args: argparse.Namespace,
    *,
    full_count: int,
    selected_count: int,
) -> list[str]:
    """Return all deviations from the committed full UAVid reproduction."""

    errors: list[str] = []
    if int(args.expected_samples) != UAVID_EXPECTED_SAMPLES:
        errors.append(
            f"expected_samples must be {UAVID_EXPECTED_SAMPLES}"
        )
    if int(full_count) != UAVID_EXPECTED_SAMPLES:
        errors.append(
            "full UAVid population must contain "
            f"{UAVID_EXPECTED_SAMPLES} pairs"
        )
    if int(selected_count) != UAVID_EXPECTED_SAMPLES:
        errors.append(
            "strict UAVid evaluation must select all "
            f"{UAVID_EXPECTED_SAMPLES} pairs"
        )
    if int(args.tile_size) != UAVID_TILE_SIZE:
        errors.append(f"tile_size must be {UAVID_TILE_SIZE}")
    if int(args.input_size) != UAVID_MODEL_INPUT_SIZE:
        errors.append(f"input_size must be {UAVID_MODEL_INPUT_SIZE}")
    if int(args.num_layers) != UAVID_NUM_LAYERS:
        errors.append(f"num_layers must be {UAVID_NUM_LAYERS}")
    if str(args.prompt_ensemble) != UAVID_PROMPT_ENSEMBLE:
        errors.append(
            f"prompt_ensemble must be {UAVID_PROMPT_ENSEMBLE}"
        )
    if str(args.amp) != UAVID_AMP:
        errors.append(f"amp must be {UAVID_AMP}")
    return errors


__all__ = [
    "UAVID_AMP",
    "UAVID_EXPECTED_SAMPLES",
    "UAVID_MODEL_CLASSES",
    "UAVID_MODEL_INPUT_SIZE",
    "UAVID_NUM_LAYERS",
    "UAVID_PROMPT_ENSEMBLE",
    "UAVID_TILE_SIZE",
    "strict_protocol_errors",
    "tile_grid_count",
    "tile_grid_shape",
    "validate_uavid_model_classes",
]
