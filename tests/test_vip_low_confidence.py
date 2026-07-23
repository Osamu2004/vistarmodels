from __future__ import annotations

import pytest

from baselines.vip.protocols import resolve_low_confidence_policy


def test_flair_auto_rejects_to_ignore_instead_of_building() -> None:
    assert resolve_low_confidence_policy("flair", "auto") == (
        "ignore",
        255,
        255,
    )


@pytest.mark.parametrize(
    ("dataset", "expected_index"),
    (
        ("loveda", 0),
        ("uavid", 0),
        ("xbd_pre", 0),
        ("chn6_cug", 0),
    ),
)
def test_explicit_background_datasets_auto_reject_to_background(
    dataset: str,
    expected_index: int,
) -> None:
    assert resolve_low_confidence_policy(dataset, "auto") == (
        "background",
        expected_index,
        None,
    )


def test_flair_cannot_force_class_zero_as_background() -> None:
    with pytest.raises(ValueError, match="no evaluated background class"):
        resolve_low_confidence_policy("flair", "background")
