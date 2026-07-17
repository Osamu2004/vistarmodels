from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SECOND_PALETTE = np.asarray(
    [
        [255, 255, 255],  # unchanged
        [0, 0, 255],      # inland water
        [128, 128, 128],  # bare land
        [0, 128, 0],      # grass
        [0, 255, 0],      # forest
        [128, 0, 0],      # building
        [255, 255, 0],    # playground
    ],
    dtype=np.uint8,
)

SECOND_RGB_TO_ID = {
    (255, 255, 255): 0,
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (128, 128, 128): 2,
    (159, 129, 183): 2,
    (192, 192, 192): 2,
    (0, 128, 0): 3,
    (0, 255, 0): 4,
    (128, 0, 0): 5,
    (255, 0, 0): 6,
    (165, 0, 165): 6,
    (255, 255, 0): 6,
}

SECOND_CLASS_NAMES = {
    1: "inland water",
    2: "bare land",
    3: "grass",
    4: "forest",
    5: "building",
    6: "playground",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"manifest is empty: {source}")
    return rows


def _format_list(values: Iterable[str]) -> str:
    items = list(values)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def controlnet_prompt(row: dict[str, Any], resolution: int) -> str:
    saved = row.get("controlnet_prompt")
    if isinstance(saved, str) and saved.strip():
        return saved.strip()
    direction = str(row.get("direction", "t1_to_t2"))
    target_time = "post-change" if direction == "t1_to_t2" else "pre-change"
    class_ids = sorted({int(value) for value in row.get("changed_class_ids", []) if 1 <= int(value) <= 6})
    changed_names = _format_list(f"changed {SECOND_CLASS_NAMES[class_id]}" for class_id in class_ids)
    if changed_names:
        semantics = (
            f"The target-side semantic change mask contains {changed_names}; "
            "unchanged pixels are labeled unchanged."
        )
    else:
        semantics = "The target-side semantic change mask contains only unchanged pixels."
    return f"Generate a realistic {resolution} by {resolution} {target_time} remote sensing image. {semantics}"


def _resize_rgb(image: Image.Image, resolution: int) -> Image.Image:
    image = image.convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    return image


def load_rgb(row: dict[str, Any], key: str, resolution: int) -> Image.Image:
    if key not in row:
        raise KeyError(f"manifest row {row.get('name', '<unnamed>')} is missing {key}")
    path = Path(str(row[key])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return _resize_rgb(Image.open(path), resolution)


def load_mask_ids(row: dict[str, Any], resolution: int) -> np.ndarray:
    materialized = row.get("target_mask_ids")
    source_value = materialized or row.get("target_mask_source")
    if not source_value:
        raise KeyError(f"manifest row {row.get('name', '<unnamed>')} has no target mask")
    source = Path(str(source_value)).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image.open(source)
    array = np.asarray(image)
    if array.ndim == 2:
        ids = array.astype(np.uint8)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        equal_channels = np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        if equal_channels and int(rgb[..., 0].max()) <= 6:
            ids = rgb[..., 0].copy()
        else:
            ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            for color, class_id in SECOND_RGB_TO_ID.items():
                ids[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
            if np.any(ids == 255):
                unknown = np.unique(rgb[ids == 255].reshape(-1, 3), axis=0)
                preview = [tuple(int(channel) for channel in color) for color in unknown[:10]]
                raise ValueError(f"{source} contains unknown SECOND RGB colors: {preview}")
    invalid = sorted(int(value) for value in np.unique(ids) if int(value) > 6)
    if invalid:
        raise ValueError(f"{source} contains invalid SECOND IDs: {invalid}")
    if ids.shape != (resolution, resolution):
        ids = np.asarray(
            Image.fromarray(ids, mode="L").resize((resolution, resolution), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return ids


def colorize_mask(ids: np.ndarray) -> Image.Image:
    if ids.ndim != 2:
        raise ValueError(f"expected HxW IDs, got {ids.shape}")
    if np.any(ids > 6):
        raise ValueError(f"mask contains IDs outside 0..6: {np.unique(ids).tolist()}")
    return Image.fromarray(SECOND_PALETTE[ids], mode="RGB")


def binary_mask(ids: np.ndarray) -> Image.Image:
    binary = ((ids > 0) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(binary[..., None], 3, axis=2), mode="RGB")
