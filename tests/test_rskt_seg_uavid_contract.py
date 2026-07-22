from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from baselines.rskt_seg.eval_rskt_seg_uavid import (
    OFFICIAL_EXPECTED_SAMPLES,
    OFFICIAL_UAVID_MODEL_CLASSES,
    shortest_edge_shape,
    strict_protocol_errors,
    validate_uavid_model_classes,
)
from baselines.segearth_ov.protocols import DATASET_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_JSON = REPO_ROOT / "baselines" / "rskt_seg" / "configs" / "uavid_8_classes.json"


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "expected_samples": 270,
        "min_size_test": 640,
        "max_size_test": 2560,
        "num_layers": 5,
        "prompt_ensemble": "single",
        "amp": "fp32",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_class_json_is_exact_official_eight_class_vocabulary() -> None:
    classes = json.loads(CLASS_JSON.read_text(encoding="utf-8"))
    assert tuple(classes) == OFFICIAL_UAVID_MODEL_CLASSES
    assert len(classes) == 8
    assert classes[0] == "Background clutter"
    assert tuple(value.casefold() for value in classes) == tuple(
        DATASET_SPECS["uavid"]["classes"]
    )
    validate_uavid_model_classes(classes)


@pytest.mark.parametrize(
    "extra_class",
    ["negative", "background", "unknown", "other"],
)
def test_extra_negative_or_background_channel_is_rejected(extra_class: str) -> None:
    with pytest.raises(ValueError, match="no extra negative class"):
        validate_uavid_model_classes(
            [*OFFICIAL_UAVID_MODEL_CLASSES, extra_class]
        )


def test_strict_official_spatial_and_population_contract() -> None:
    assert OFFICIAL_EXPECTED_SAMPLES == 270
    assert strict_protocol_errors(_args(), full_count=270) == []
    errors = strict_protocol_errors(
        _args(min_size_test=512, amp="fp16"),
        full_count=269,
    )
    assert "full UAVid population must contain 270 pairs" in errors
    assert "min_size_test must be 640" in errors
    assert "amp must be fp32" in errors


def test_shortest_edge_shape_matches_official_uavid_resize() -> None:
    assert shortest_edge_shape(2160, 3840, 640, 2560) == (640, 1138)
    assert shortest_edge_shape(3000, 4000, 640, 2560) == (640, 853)
    assert shortest_edge_shape(512, 512, 640, 2560) == (640, 640)
