"""Strict semantic-descriptor decoding and benchmark contracts.

RSUniVLM emits row-wise ``class*count`` text, not a dense logit tensor.
Malformed text must never silently become an all-background prediction.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from baselines.levircd_common import discover_samples
from baselines.segearth_ov.protocols import (
    LOVEDA_CLASSES,
    LOVEDA_PALETTE,
    discover_loveda,
)

UPSTREAM_URL = "https://github.com/xuliu-cyber/RSUniVLM"
UPSTREAM_COMMIT = "66dce1dc68b81a6c62f02fb8e5b306571092a864"
OFFICIAL_WEIGHT_URL = "https://drive.google.com/drive/folders/1TtaoOPmh167gpgHHWRNBMCaA7t_XZ4Vg"
GRID_SIZE = 24
MAX_DESCRIPTOR_SIDE = 64
RLE_GRID_POLICY = "native_rectangular"
PROTOCOL_VERSION = 2
CD_CLASSES = ("others", "building")
LOVEDA_PROMPT = (
    "[SEG] Please segment this remote-sensing image into the following classes: "
    "background, building, road, water, barren, forest, agriculture. "
    "Output only the 24-by-24 segmentation mask in row-wise run-length format: "
    "class*count, class*count, with one row per line."
)
CD_PROMPT = "[SEG] Please segment the building area that have changed in the second image."


@dataclass(frozen=True)
class Sample:
    name: str
    images: tuple[Path, ...]
    label: Path


def discover(dataset: str, data_root: Path, split: str) -> tuple[list[Sample], dict[str, Any]]:
    if dataset == "loveda":
        if split.casefold() != "val":
            raise ValueError("LoveDA segmentation evaluates Val only")
        records, audit = discover_loveda(data_root)
        return [Sample(x.name, (x.image_path,), x.mask_path) for x in records], audit
    split_root, records = discover_samples(data_root, split)
    # The older shared helper intersects triplets. Reject missing or extra
    # members here so an incomplete population cannot receive a full score.
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    indices = []
    for folder in ("A", "B", "label"):
        files = [p for p in (split_root / folder).iterdir()
                 if p.is_file() and not p.name.startswith(".") and p.suffix.casefold() in extensions]
        stems = [p.stem.casefold() for p in files]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate LEVIR-CD stems under {split_root / folder}")
        indices.append(set(stems))
    if not (indices[0] == indices[1] == indices[2]):
        raise ValueError("LEVIR-CD A/B/label stem sets differ; refusing an intersected subset")
    return [Sample(x.stem, (x.image_a, x.image_b), x.label) for x in records], {
        "dataset": "LEVIR-CD", "split": split, "resolved_data_root": str(split_root),
        "num_pairs": len(records),
    }


def normalize_label(label: str) -> str:
    return re.sub(r"[\s_-]+", " ", label.strip().strip("\"'").casefold())


def label_map(dataset: str) -> dict[str, int]:
    if dataset == "levircd":
        return {"others": 0, "other": 0, "background": 0, "unchanged": 0,
                "no change": 0, "building": 1, "buildings": 1}
    mapping = {name: i for i, name in enumerate(LOVEDA_CLASSES)}
    aliases = {
        "others": "background", "other": "background", "buildings": "building",
        "agricultural": "agriculture", "agricultural land": "agriculture",
        "agriculture land": "agriculture", "barren land": "barren",
    }
    mapping.update({key: mapping[value] for key, value in aliases.items()})
    return mapping


def decode_rle(text: str, dataset: str, *, grid_size: int | None = GRID_SIZE) -> np.ndarray:
    """Decode a complete rectangle; optional grid_size retains strict square mode.

    The official demo restores the dimensions actually emitted by the model.
    Native mode never adds rows or pads ragged rows to make a valid rectangle.
    """
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("Expected a bounded text segmentation response")
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed segmentation code fence")
        value = "\n".join(lines[1:-1]).strip()
    rows = [line.strip() for line in value.splitlines() if line.strip()]
    if rows and rows[0].casefold() in ("mask:", "segmentation mask:", "[seg]"):
        rows = rows[1:]
    if grid_size is not None and not 1 <= grid_size <= MAX_DESCRIPTOR_SIDE:
        raise ValueError("Invalid requested descriptor grid size")
    if not 1 <= len(rows) <= MAX_DESCRIPTOR_SIDE:
        raise ValueError(f"Expected 1..{MAX_DESCRIPTOR_SIDE} RLE rows, received {len(rows)}")
    if grid_size is not None and len(rows) != grid_size:
        raise ValueError(f"Expected {grid_size} RLE rows, received {len(rows)}")
    mapping = label_map(dataset)
    decoded_rows: list[list[int]] = []
    width = grid_size
    for row_index, row in enumerate(rows):
        values: list[int] = []
        for item in row.split(","):
            match = re.fullmatch(r"\s*([A-Za-z][A-Za-z _'-]*)\s*\*\s*([0-9]+)\s*", item)
            if match is None:
                raise ValueError(f"Invalid RLE item at row {row_index + 1}: {item!r}")
            label, count_text = match.groups()
            key = normalize_label(label)
            if key not in mapping:
                raise ValueError(f"Unknown {dataset} label {label!r} at row {row_index + 1}")
            count = int(count_text)
            if count <= 0 or len(values) + count > (width or MAX_DESCRIPTOR_SIDE):
                raise ValueError(f"RLE count exceeds row width at row {row_index + 1}")
            values.extend([mapping[key]] * count)
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError(f"RLE row {row_index + 1} has width {len(values)}, expected {width}")
        decoded_rows.append(values)
    return np.asarray(decoded_rows, dtype=np.uint8)


def restore_grid(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid native image extent: {shape}")
    # Match official app.py's INTER_NEAREST pixel alignment. Its integer
    # patch expansion is unnecessary when resizing class IDs directly.
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)


def loveda_spatial_settings(tile_size: int) -> dict[str, Any]:
    """Distinguish the source-image crop from the fixed vision/mask grids."""
    if tile_size not in (0, 512):
        raise ValueError("LoveDA uses 512-pixel non-overlapping source tiles, or 0 for legacy whole-image inference")
    return {
        "mode": "nonoverlapping_source_tiles" if tile_size else "whole_image",
        "source_tile_size": tile_size,
        "source_stride": tile_size,
        "rgb_padding": "black_right_bottom" if tile_size else "none",
        "vision_input_size": [384, 384],
        "vision_resize": "official_SigLipImageProcessor_per_tile" if tile_size else "official_SigLipImageProcessor_whole",
        "nominal_mask_grid_size": GRID_SIZE,
        "mask_grid_policy": RLE_GRID_POLICY,
        "max_descriptor_side": MAX_DESCRIPTOR_SIDE,
        "nearest_implementation": "opencv_INTER_NEAREST",
        "restoration": "nearest_to_tile_then_stitch_crop_to_native" if tile_size else "nearest_to_native_extent",
        "tta": "none",
    }


def fuse_change(grids: Sequence[np.ndarray]) -> np.ndarray:
    if not grids or any(x.shape != grids[0].shape for x in grids):
        raise ValueError("Change masks must be nonempty with matching shapes")
    if any(not np.all((x == 0) | (x == 1)) for x in grids):
        raise ValueError("Change masks must contain binary IDs")
    return np.maximum.reduce(grids).astype(np.uint8)


def expected_samples(dataset: str, split: str) -> int:
    if dataset == "loveda":
        return 1669
    return {"train": 445, "val": 64, "test": 128}[split]


def source_revision(source_root: Path) -> str:
    """Record the pinned commit; local working-tree changes are allowed."""
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, timeout=15,
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f"RSUniVLM source revision {revision} != pinned {UPSTREAM_COMMIT}")
    return revision


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    """Validate a local full checkpoint and identify cache-incompatible changes.

    JSON/tokenizer metadata is hashed; large weight payloads use size/mtime.
    This is a resume identity, not a cryptographic model-weight attestation.
    """
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "llava_qwen_gmoe":
        raise ValueError("Expected the official llava_qwen_gmoe checkpoint, not a generic Qwen/LLaVA")
    weight_files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No model weights under {model_path}; download {OFFICIAL_WEIGHT_URL}")
    for index_path in model_path.glob("*.index.json"):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if "weight_map" in index:
            for name in set(index["weight_map"].values()):
                shard = (model_path / name).resolve()
                if not shard.is_relative_to(model_path.resolve()) or not shard.is_file() or shard.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing/unsafe weight shard in {index_path}: {name}")
    metadata = []
    for path in sorted(model_path.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix in (".json", ".model", ".txt") or path in weight_files:
            stat = path.stat()
            item: dict[str, Any] = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path not in weight_files:
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif stat.st_size == 0:
                raise ValueError(f"Empty checkpoint shard: {path}")
            metadata.append(item)
    if not (model_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer.json under {model_path}")
    return {"path": str(model_path), "files": metadata, "vision_tower": config.get("mm_vision_tower")}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def input_identity(sample: Sample) -> list[dict[str, Any]]:
    return [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (*sample.images, sample.label)]


def palette(dataset: str) -> np.ndarray:
    if dataset == "loveda":
        return LOVEDA_PALETTE
    return np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
