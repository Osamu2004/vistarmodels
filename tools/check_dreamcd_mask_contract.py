#!/usr/bin/env python3
"""Regression check for DreamCD's raw binary-change-mask convention.

The official repository ships three SECOND examples.  Their BCD files must
equal ``255`` exactly where the two dense pseudo-semantic maps differ, and
``0`` elsewhere.  This checker also exercises every supported external binary
encoding conversion without requiring the DreamCD/PyTorch environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.dreamcd.mask_contract import (
    DREAMCD_RAW_CHANGED,
    DREAMCD_RAW_UNCHANGED,
    derive_dreamcd_raw_from_second_target_change,
    derive_dreamcd_raw_from_semantic_pair,
    normalise_binary_change_to_dreamcd_raw,
)


def load_l(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamcd_root", default="third_party/DreamCD")
    args = parser.parse_args()
    root = Path(args.dreamcd_root).expanduser().resolve()
    example = root / "example" / "second"
    bcd_dir, mask_a_dir, mask_b_dir = example / "bcd_mask", example / "mask_A", example / "mask_B"
    masks = sorted(bcd_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No official SECOND BCD examples found under {bcd_dir}")

    for bcd_path in masks:
        mask_a = load_l(mask_a_dir / bcd_path.name)
        mask_b = load_l(mask_b_dir / bcd_path.name)
        actual = load_l(bcd_path)
        expected = derive_dreamcd_raw_from_semantic_pair(mask_a, mask_b)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"Official example violates 255=changed contract: {bcd_path}")
        print(f"OK official {bcd_path.name}: changed={(actual == DREAMCD_RAW_CHANGED).sum()} unchanged={(actual == DREAMCD_RAW_UNCHANGED).sum()}")

    binary_01 = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    binary_0255 = binary_01 * 255
    expected = binary_0255
    for mode, source in (("auto", binary_0255), ("auto", binary_01), ("white_changed", binary_0255), ("zero_changed", 1 - binary_01), ("white_unchanged", 255 - binary_0255)):
        actual = normalise_binary_change_to_dreamcd_raw(source, mode)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"Unexpected conversion for binary_change_mode={mode}")

    second_target = np.asarray([[0, 4], [2, 0]], dtype=np.uint8)
    if not np.array_equal(derive_dreamcd_raw_from_second_target_change(second_target), expected):
        raise AssertionError("SECOND sparse target-change conversion is incorrect")
    print("OK binary encoding conversions and SECOND sparse target-change conversion")


if __name__ == "__main__":
    main()
