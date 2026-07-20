from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BINARY_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)


@dataclass(frozen=True)
class LevirSample:
    stem: str
    image_a: Path
    image_b: Path
    label: Path


def normalize_wsl_unc(value: str | Path) -> str:
    text = str(value)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [part for part in text.strip("\\").split("\\") if part]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def resolve_path(value: str | Path) -> Path:
    return Path(normalize_wsl_unc(value)).expanduser().resolve()


def resolve_split_root(data_root: str | Path, split: str) -> Path:
    root = resolve_path(data_root)
    aliases = {
        "train": ("train", "Train", "training", "Training"),
        "val": ("val", "Val", "valid", "Valid", "validation", "Validation"),
        "test": ("test", "Test", "testing", "Testing"),
    }
    candidates = [root / name for name in aliases[split]] + [root]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("A", "B", "label")):
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise NotADirectoryError(
        "Cannot find LEVIR-CD A/B/label directories. Checked:\n" + checked
    )


def _index_images(directory: Path) -> dict[str, Path]:
    images = {
        path.stem.casefold(): path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.casefold() in IMAGE_EXTENSIONS
    }
    if not images:
        raise FileNotFoundError(f"No supported images found under {directory}")
    return images


def discover_samples(
    data_root: str | Path,
    split: str,
    max_samples: int = 0,
) -> tuple[Path, list[LevirSample]]:
    split_root = resolve_split_root(data_root, split)
    a_index = _index_images(split_root / "A")
    b_index = _index_images(split_root / "B")
    label_index = _index_images(split_root / "label")
    common = sorted(set(a_index) & set(b_index) & set(label_index))
    if max_samples > 0:
        common = common[: int(max_samples)]
    if not common:
        raise RuntimeError(f"No complete LEVIR-CD triplets found under {split_root}")
    samples = [
        LevirSample(
            stem=a_index[key].stem,
            image_a=a_index[key],
            image_b=b_index[key],
            label=label_index[key],
        )
        for key in common
    ]
    return split_root, samples


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(
            np.uint8
        )


def binary_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    pred = prediction.reshape(-1).astype(bool)
    gt = target.reshape(-1).astype(bool)
    return {
        "tp": int(np.count_nonzero(pred & gt)),
        "tn": int(np.count_nonzero(~pred & ~gt)),
        "fp": int(np.count_nonzero(pred & ~gt)),
        "fn": int(np.count_nonzero(~pred & gt)),
    }


def binary_metrics(counts: dict[str, int], epsilon: float = 1.0e-7) -> dict[str, float]:
    tp, tn, fp, fn = (float(counts[key]) for key in ("tp", "tn", "fp", "fn"))
    iou_change = tp / (tp + fp + fn + epsilon)
    iou_nochange = tn / (tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return {
        "miou": float(0.5 * (iou_change + iou_nochange)),
        "oa": float((tp + tn) / (tp + tn + fp + fn + epsilon)),
        "iou_change": float(iou_change),
        "iou_nochange": float(iou_nochange),
        "f1_score_change": float(
            (2.0 * precision * recall) / (precision + recall + epsilon)
        ),
        "precision_change": float(precision),
        "recall_change": float(recall),
    }


def sum_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    return {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("tp", "tn", "fp", "fn")
    }


def make_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = resolve_path(output_dir)
    names = (
        "input_A",
        "input_B",
        "gt_mask",
        "pred_mask",
        "pred_rgb",
        "error_map",
        "overlay_A",
        "overlay_B",
        "runtime",
    )
    directories = {name: root / name for name in names}
    root.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    directories["root"] = root
    return directories


def _save_gray(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) > 0).astype(np.uint8) * 255, mode="L").save(
        path
    )


def _save_binary_rgb(mask: np.ndarray, path: Path) -> None:
    ids = (mask.astype(np.uint8) > 0).astype(np.uint8)
    Image.fromarray(BINARY_PALETTE[ids], mode="RGB").save(path)


def make_error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RCD-paper colors: TP white, TN black, FP green, FN red."""

    pred = prediction.astype(bool)
    gt = target.astype(bool)
    result = np.zeros((*target.shape, 3), dtype=np.uint8)
    result[pred & gt] = (255, 255, 255)
    result[pred & ~gt] = (0, 255, 0)
    result[~pred & gt] = (255, 0, 0)
    return result


def make_overlay(image: np.ndarray, prediction: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    output = image.astype(np.float32).copy()
    changed = prediction.astype(bool)
    color = np.asarray([255.0, 64.0, 64.0], dtype=np.float32)
    output[changed] = (1.0 - alpha) * output[changed] + alpha * color
    return output.clip(0, 255).astype(np.uint8)


def save_artifacts(
    directories: dict[str, Path],
    stem: str,
    image_a: np.ndarray,
    image_b: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    save_images: bool,
    overlay_alpha: float = 0.45,
) -> None:
    if not save_images:
        return
    Image.fromarray(image_a, mode="RGB").save(directories["input_A"] / f"{stem}.png")
    Image.fromarray(image_b, mode="RGB").save(directories["input_B"] / f"{stem}.png")
    _save_gray(target, directories["gt_mask"] / f"{stem}.png")
    _save_gray(prediction, directories["pred_mask"] / f"{stem}_pred_mask.png")
    _save_binary_rgb(prediction, directories["pred_rgb"] / f"{stem}_pred_rgb.png")
    Image.fromarray(make_error_map(prediction, target), mode="RGB").save(
        directories["error_map"] / f"{stem}.png"
    )
    Image.fromarray(
        make_overlay(image_a, prediction, alpha=overlay_alpha), mode="RGB"
    ).save(directories["overlay_A"] / f"{stem}.png")
    Image.fromarray(
        make_overlay(image_b, prediction, alpha=overlay_alpha), mode="RGB"
    ).save(directories["overlay_B"] / f"{stem}.png")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tile_coordinates(height: int, width: int, tile_size: int) -> list[tuple[int, int]]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    return [
        (top, left)
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    ]


def pad_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_h = max(0, int(height) - tile.shape[0])
    pad_w = max(0, int(width) - tile.shape[1])
    if pad_h == 0 and pad_w == 0:
        return tile
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
