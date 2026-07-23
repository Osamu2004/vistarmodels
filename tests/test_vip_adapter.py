from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from baselines.segearth_ov.protocols import DATASET_SPECS
from baselines.vip.protocols import (
    compute_square_padding,
    flatten_class_groups,
    read_vip_class_groups,
    sliding_window_boxes,
)


def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def test_bundled_vip_alias_files_preserve_dataset_class_order() -> None:
    root = Path(__file__).resolve().parents[1] / "baselines" / "vip" / "configs"
    filenames = {
        "loveda": "loveda.txt",
        "flair": "flair_12.txt",
        "uavid": "uavid_8.txt",
        "xbd_pre": "xbd_pre.txt",
        "chn6_cug": "chn6_cug.txt",
    }
    for dataset, filename in filenames.items():
        groups = read_vip_class_groups(root / filename)
        expected = tuple(_canonical(name) for name in DATASET_SPECS[dataset]["classes"])
        actual = tuple(_canonical(group[0]) for group in groups)
        assert actual == expected


def test_vip_alias_parser_matches_official_comma_space_contract(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text("background, other\nbuilding, roof, house\n", encoding="utf-8")
    groups = read_vip_class_groups(valid)
    assert groups == (("background", "other"), ("building", "roof", "house"))
    aliases, indices = flatten_class_groups(groups)
    assert aliases == ("background", "other", "building", "roof", "house")
    assert indices == (0, 0, 1, 1, 1)

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("background,other\nbuilding\n", encoding="utf-8")
    with pytest.raises(ValueError, match="comma followed by one space"):
        read_vip_class_groups(invalid)


def test_vip_release_geometry_has_complete_overlap_coverage() -> None:
    boxes = sliding_window_boxes(448, 448, 336, 112)
    assert boxes == (
        (0, 336, 0, 336),
        (0, 336, 112, 448),
        (112, 448, 0, 336),
        (112, 448, 112, 448),
    )
    coverage = np.zeros((448, 448), dtype=np.int32)
    for y1, y2, x1, x2 in boxes:
        coverage[y1:y2, x1:x2] += 1
    assert int(np.min(coverage)) == 1
    assert int(np.max(coverage)) == 4


def test_vip_rectangular_edge_crop_is_padded_to_fixed_token_grid() -> None:
    assert compute_square_padding(252, 336, 336) == (0, 0, 42, 42)
    assert compute_square_padding(335, 333, 336) == (1, 2, 0, 1)
    with pytest.raises(ValueError):
        compute_square_padding(337, 336, 336)
