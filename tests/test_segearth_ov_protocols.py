from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from baselines.segearth_ov.protocols import (
    DATASET_SPECS,
    EvalSample,
    confusion_matrix,
    discover_chn6_cug,
    discover_loveda,
    discover_uavid,
    discover_xbd_pre,
    load_rgb,
    load_target,
    metrics_for_dataset,
    metrics_from_confusion,
    read_class_groups,
    resize_keep_ratio,
    validate_class_groups,
)


def _save_rgb(path: Path, shape: tuple[int, int] = (8, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((*shape, 3), dtype=np.uint8)
    values[..., 0] = 100
    Image.fromarray(values, mode="RGB").save(path)


def _save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def test_all_bundled_class_files_match_fixed_protocols() -> None:
    config_root = Path(__file__).resolve().parents[1] / "baselines" / "segearth_ov" / "configs"
    filenames = {
        "loveda": "loveda.txt",
        "flair": "flair_12.txt",
        "uavid": "uavid_8.txt",
        "xbd_pre": "xbd_pre.txt",
        "chn6_cug": "chn6_cug.txt",
    }
    for dataset, filename in filenames.items():
        groups = read_class_groups(config_root / filename)
        validate_class_groups(dataset, groups)
        assert len(groups) == len(DATASET_SPECS[dataset]["classes"])


def test_uavid_spec_exactly_matches_vistar_eight_class_protocol() -> None:
    spec = DATASET_SPECS["uavid"]
    assert spec["classes"] == (
        "background clutter",
        "building",
        "road",
        "tree",
        "low vegetation",
        "moving car",
        "static car",
        "human",
    )
    assert np.asarray(spec["palette"]).tolist() == [
        [0, 0, 0],
        [128, 0, 0],
        [128, 64, 128],
        [0, 128, 0],
        [128, 128, 0],
        [64, 0, 128],
        [192, 0, 192],
        [64, 64, 0],
    ]
    assert spec["expected_samples"] == 270
    assert spec["primary_metric"] == "miou"


def test_loveda_discovery_and_reduce_zero_label(tmp_path: Path) -> None:
    for domain, raw in (
        ("Urban", np.asarray([[0, 1], [2, 7]], dtype=np.uint8)),
        ("Rural", np.asarray([[7, 6], [1, 0]], dtype=np.uint8)),
    ):
        _save_rgb(tmp_path / "Val" / domain / "images_png" / f"{domain}.png", (2, 2))
        _save_mask(tmp_path / "Val" / domain / "masks_png" / f"{domain}.png", raw)

    records, audit = discover_loveda(tmp_path)
    assert len(records) == 2
    assert audit["domain_counts"] == {"Urban": 1, "Rural": 1}
    urban = next(record for record in records if record.domain == "Urban")
    image = load_rgb(urban, "loveda")
    target = load_target(urban, "loveda", height=2, width=2)
    assert image.shape == (2, 2, 3)
    assert target.tolist() == [[255, 0], [1, 6]]


def test_loveda_rgb_palette_decode_uses_non_overflowing_distance(tmp_path: Path) -> None:
    image_path = tmp_path / "images_png" / "palette.png"
    mask_path = tmp_path / "masks_png" / "palette.png"
    _save_rgb(image_path, (1, 7))
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(DATASET_SPECS["loveda"]["palette"], dtype=np.uint8)
    Image.fromarray(palette.reshape(1, 7, 3), mode="RGB").save(mask_path)
    sample = EvalSample("palette", image_path, mask_path)
    target = load_target(sample, "loveda", height=1, width=7)
    assert target.tolist() == [list(range(7))]


def test_uavid_role_token_pairing_and_eight_class_rgb_decode(tmp_path: Path) -> None:
    image_path = tmp_path / "Images" / "seq10_Images_000000.png"
    mask_path = tmp_path / "Labels" / "seq10_Labels_000000.png"
    _save_rgb(image_path, (2, 4))
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(DATASET_SPECS["uavid"]["palette"], dtype=np.uint8)
    Image.fromarray(palette.reshape(2, 4, 3), mode="RGB").save(mask_path)

    records, audit = discover_uavid(tmp_path)
    assert len(records) == 1
    assert records[0].mask_path == mask_path
    assert records[0].mask_id_base == "zero"
    assert audit["complete_pairing"] is True
    assert audit["mask_encoding"]["rgb_files"] == 1
    target = load_target(records[0], "uavid", height=2, width=4)
    assert target.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_uavid_explicit_one_based_indexed_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "Images" / "frame.png"
    mask_path = tmp_path / "Labels" / "frame.png"
    _save_rgb(image_path, (2, 4))
    _save_mask(mask_path, np.arange(1, 9, dtype=np.uint8).reshape(2, 4))

    records, audit = discover_uavid(tmp_path, mask_id_base="one")
    assert audit["mask_encoding"]["resolved_mask_id_base"] == "one"
    target = load_target(records[0], "uavid", height=2, width=4)
    assert target.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_uavid_primary_metric_includes_all_eight_classes() -> None:
    confusion = np.diag(np.arange(1, 9, dtype=np.int64))
    metrics = metrics_for_dataset(confusion, "uavid")
    assert metrics["miou"] == 1.0
    assert metrics["miou_foreground7"] == 1.0
    assert len(DATASET_SPECS["uavid"]["classes"]) == 8

    background_confusion = np.zeros((8, 8), dtype=np.int64)
    background_confusion[0, 1] = 9
    background_confusion[1, 0] = 1
    background_confusion[1, 1] = 1
    metrics = metrics_for_dataset(background_confusion, "uavid")
    assert metrics["miou_foreground7"] == pytest.approx(1.0 / 11.0)


def test_xbd_pre_prepared_layout_filters_to_pre_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "test" / "images"
    label_dir = tmp_path / "test" / "targets_cvt"
    _save_rgb(image_dir / "event_pre_disaster.png", (3, 4))
    _save_rgb(image_dir / "event_post_disaster.png", (3, 4))
    _save_mask(label_dir / "event_pre_disaster.png", np.asarray([[0, 1, 0, 0]] * 3))

    records, audit = discover_xbd_pre(tmp_path)
    assert [record.name for record in records] == ["event_pre_disaster"]
    assert audit["num_pairs"] == 1
    target = load_target(records[0], "xbd_pre", height=3, width=4)
    assert set(np.unique(target).tolist()) == {0, 1}


def test_chn6_raw_sat_mask_name_matching(tmp_path: Path) -> None:
    _save_rgb(tmp_path / "images" / "area_sat.jpg", (3, 5))
    _save_mask(tmp_path / "gt" / "area_mask.png", np.asarray([[0, 255, 0, 1, 0]] * 3))
    records, audit = discover_chn6_cug(tmp_path)
    assert len(records) == 1
    assert records[0].mask_path.name == "area_mask.png"
    assert audit["num_pairs"] == 1
    target = load_target(records[0], "chn6_cug", height=3, width=5)
    assert target[0].tolist() == [0, 1, 0, 1, 0]


def test_keep_ratio_resize_uses_longest_side() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_keep_ratio(image, 448)
    assert resized.shape == (224, 448, 3)


def test_confusion_metrics_ignore_and_binary_primary_names() -> None:
    target = np.asarray([[0, 1, 255], [1, 0, 255]], dtype=np.int64)
    prediction = np.asarray([[0, 0, 1], [1, 0, 0]], dtype=np.int64)
    matrix = confusion_matrix(
        prediction,
        target,
        num_classes=2,
        ignore_index=255,
    )
    assert matrix.tolist() == [[2, 0], [1, 1]]
    metrics = metrics_from_confusion(matrix, ("background", "building"))
    assert metrics["iou_background"] == 2 / 3
    assert metrics["iou_building"] == 1 / 2
    assert metrics["miou"] == (2 / 3 + 1 / 2) / 2
    assert metrics["valid_pixels"] == 4
    xbd_metrics = metrics_for_dataset(matrix, "xbd_pre")
    assert xbd_metrics["building_iou"] == xbd_metrics["iou_building"] == 1 / 2
    assert xbd_metrics["background_iou"] == xbd_metrics["iou_background"] == 2 / 3
