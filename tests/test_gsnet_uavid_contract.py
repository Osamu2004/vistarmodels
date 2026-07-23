from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from baselines.gsnet.gsnet_uavid_protocol import (
    UAVID_EXPECTED_SAMPLES,
    UAVID_MODEL_CLASSES,
    strict_protocol_errors,
    tile_grid_count,
    tile_grid_shape,
    validate_uavid_model_classes,
)
from baselines.segearth_ov.protocols import DATASET_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_JSON = (
    REPO_ROOT
    / "baselines"
    / "gsnet"
    / "configs"
    / "uavid_8_classes.json"
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "expected_samples": 270,
        "tile_size": 512,
        "input_size": 512,
        "num_layers": 2,
        "prompt_ensemble": "single",
        "amp": "fp32",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_class_json_matches_vistar_uavid_eight_class_order() -> None:
    classes = json.loads(CLASS_JSON.read_text(encoding="utf-8"))
    assert tuple(classes) == UAVID_MODEL_CLASSES
    assert tuple(classes) == tuple(DATASET_SPECS["uavid"]["classes"])
    assert len(classes) == 8
    assert classes[0] == "background clutter"
    validate_uavid_model_classes(classes)


@pytest.mark.parametrize(
    "extra_class",
    ["negative", "background", "unknown", "other", "support"],
)
def test_extra_support_or_negative_channel_is_rejected(
    extra_class: str,
) -> None:
    with pytest.raises(ValueError, match="No extra"):
        validate_uavid_model_classes([*UAVID_MODEL_CLASSES, extra_class])


def test_strict_full_population_and_inference_contract() -> None:
    assert UAVID_EXPECTED_SAMPLES == 270
    assert (
        strict_protocol_errors(
            _args(),
            full_count=270,
            selected_count=270,
        )
        == []
    )
    errors = strict_protocol_errors(
        _args(tile_size=640, amp="fp16"),
        full_count=269,
        selected_count=2,
    )
    assert "full UAVid population must contain 270 pairs" in errors
    assert "strict UAVid evaluation must select all 270 pairs" in errors
    assert "tile_size must be 512" in errors
    assert "amp must be fp32" in errors


def test_native_nonoverlap_tile_grid_and_padding_extent() -> None:
    assert tile_grid_shape(2160, 3840, 512) == (5, 8)
    assert tile_grid_count(2160, 3840, 512) == 40
    assert tile_grid_shape(3000, 4000, 512) == (6, 8)
    assert tile_grid_count(3000, 4000, 512) == 48
    assert tile_grid_count(512, 512, 512) == 1
