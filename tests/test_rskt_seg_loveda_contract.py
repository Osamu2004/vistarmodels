from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from baselines.rskt_seg.eval_rskt_seg_loveda import (
    LOVEDA_CONTRACT,
    LOVEDA_MODEL_CLASSES,
    OFFICIAL_EXPECTED_SAMPLES,
    strict_protocol_errors,
    validate_loveda_model_classes,
)
from baselines.rskt_seg.eval_rskt_seg_multiclass import shortest_edge_shape
from baselines.segearth_ov.protocols import DATASET_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_JSON = (
    REPO_ROOT
    / "baselines"
    / "rskt_seg"
    / "configs"
    / "loveda_7_classes.json"
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "expected_samples": 1669,
        "min_size_test": 640,
        "max_size_test": 2560,
        "num_layers": 5,
        "prompt_ensemble": "single",
        "amp": "fp32",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_class_json_matches_vistar_seven_class_order() -> None:
    classes = json.loads(CLASS_JSON.read_text(encoding="utf-8"))
    assert tuple(classes) == LOVEDA_MODEL_CLASSES
    assert tuple(value.casefold() for value in classes) == tuple(
        DATASET_SPECS["loveda"]["classes"]
    )
    assert len(classes) == 7
    validate_loveda_model_classes(classes)


def test_official_rskt_no_data_channel_is_deliberately_rejected() -> None:
    assert LOVEDA_CONTRACT.official_rskt_taxonomy_protocol is False
    assert "raw LoveDA ID 0 is ignored" in LOVEDA_CONTRACT.taxonomy_note
    with pytest.raises(ValueError, match="no extra negative class"):
        validate_loveda_model_classes(["no-data", *LOVEDA_MODEL_CLASSES])


def test_strict_loveda_population_and_spatial_contract() -> None:
    assert OFFICIAL_EXPECTED_SAMPLES == 1669
    assert strict_protocol_errors(_args(), full_count=1669) == []
    errors = strict_protocol_errors(
        _args(expected_samples=2, max_size_test=1024),
        full_count=2,
    )
    assert "expected_samples must be 1669" in errors
    assert "full LoveDA population must contain 1669 pairs" in errors
    assert "max_size_test must be 2560" in errors


def test_loveda_shortest_edge_resize_and_native_restore_contract() -> None:
    assert shortest_edge_shape(1024, 1024, 640, 2560) == (640, 640)
    assert shortest_edge_shape(512, 1024, 640, 2560) == (640, 1280)
