"""Mask conventions shared by the DreamCD SECOND adapter.

DreamCD's official ``ChangeAnywhere2`` loader reads a raw BCD image and maps
``raw == 255`` to its internal sampling mask value ``0``.  The sampler keeps
the source latent where that internal value is ``1`` and synthesizes where it
is ``0``.  Therefore the only valid raw on-disk contract is:

``255 = changed`` and ``0 = unchanged``.

Keep all conversion in this dependency-free module so the manifest builder,
inference wrapper, and its regression check cannot drift apart.
"""

from __future__ import annotations

import numpy as np


DREAMCD_RAW_CHANGED = np.uint8(255)
DREAMCD_RAW_UNCHANGED = np.uint8(0)


def changed_from_dreamcd_raw(raw_mask: np.ndarray) -> np.ndarray:
    """Return a boolean changed map from a raw DreamCD BCD mask."""
    return np.asarray(raw_mask, dtype=np.uint8) == DREAMCD_RAW_CHANGED


def normalise_binary_change_to_dreamcd_raw(mask: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Convert a binary input encoding to DreamCD's raw 255=change contract.

    ``auto`` treats a {0,255} file as the official DreamCD convention.  For
    {0,1} and arbitrary-valued masks it treats non-zero values as changes,
    which is the usual convention for external binary CD datasets.  Explicit
    modes are provided when a source uses a different encoding.
    """
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Binary change mask must be HxW, got shape {values.shape}.")
    values = values.astype(np.uint8, copy=False)
    unique = set(int(value) for value in np.unique(values).tolist())

    if mode == "auto":
        if unique.issubset({0, 255}):
            changed = values == 255
        else:
            changed = values != 0
    elif mode == "white_changed":
        changed = values >= 128
    elif mode == "white_unchanged":
        changed = values < 128
    elif mode == "zero_changed":
        changed = values == 0
    elif mode == "nonzero_changed":
        changed = values != 0
    else:
        raise ValueError(f"Unsupported binary_change_mode={mode!r}")
    return np.where(changed, DREAMCD_RAW_CHANGED, DREAMCD_RAW_UNCHANGED).astype(np.uint8)


def derive_dreamcd_raw_from_semantic_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Derive a raw DreamCD BCD mask from paired dense pseudo-semantic maps."""
    first, second = np.asarray(mask_a), np.asarray(mask_b)
    if first.shape != second.shape:
        raise ValueError(f"Cannot derive a BCD mask from mismatched shapes: {first.shape} vs {second.shape}.")
    return np.where(first != second, DREAMCD_RAW_CHANGED, DREAMCD_RAW_UNCHANGED).astype(np.uint8)


def derive_dreamcd_raw_from_second_target_change(target_change_ids: np.ndarray) -> np.ndarray:
    """Derive raw DreamCD BCD from a sparse official SECOND target change map."""
    target = np.asarray(target_change_ids)
    if target.ndim != 2:
        raise ValueError(f"SECOND target change mask must be HxW, got shape {target.shape}.")
    return np.where(target != 0, DREAMCD_RAW_CHANGED, DREAMCD_RAW_UNCHANGED).astype(np.uint8)
