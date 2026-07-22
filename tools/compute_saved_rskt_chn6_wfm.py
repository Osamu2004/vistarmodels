#!/usr/bin/env python3
"""Recompute CHN6-CUG metrics and WFm from saved RSKT-Seg masks.

This entry point deliberately does not import Detectron2 or construct the
RSKT-Seg model.  It is intended for complete legacy result directories whose
``run_config.json`` predates the tiled-protocol metadata now required by the
inference launcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_binary_boundary_wfm,
)


EXPECTED_NUM_SAMPLES = 903
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
Image.MAX_IMAGE_PIXELS = None


def _path(value: str) -> Path:
    text = str(value).strip().strip("\"'").replace("\\", "/")
    if text.casefold().startswith("//wsl.localhost/"):
        parts = [part for part in text.split("/") if part]
        if len(parts) >= 3:
            text = "/" + "/".join(parts[2:])
    return Path(text).expanduser().resolve()


def _resolve_chn6_dirs(data_root: Path) -> tuple[Path, Path, str]:
    layouts = (
        ("images", "gt"),
        ("images", "labels"),
        ("images", "masks"),
        ("image", "label"),
        ("imgs", "gt"),
        ("imgs", "masks"),
    )
    checked: list[tuple[Path, Path]] = []
    for split_root, split_name in ((data_root, data_root.name), (data_root / "val", "val")):
        for image_name, mask_name in layouts:
            image_dir = split_root / image_name
            mask_dir = split_root / mask_name
            checked.append((image_dir, mask_dir))
            if image_dir.is_dir() and mask_dir.is_dir():
                return image_dir, mask_dir, split_name
    details = "\n".join(f"  images={images} masks={masks}" for images, masks in checked)
    raise NotADirectoryError(
        f"Cannot find CHN6-CUG image/mask folders under {data_root}. Checked:\n{details}"
    )


def _list_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found under {image_dir}")
    return images


def _find_mask(mask_dir: Path, image_path: Path) -> Path:
    stems = [image_path.stem.replace("_sat", "_mask")] if "_sat" in image_path.stem else []
    stems.append(image_path.stem)
    for stem in stems:
        for extension in MASK_EXTENSIONS:
            candidate = mask_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Cannot find a CHN6-CUG mask matching {image_path.name} under {mask_dir}"
    )


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return (array != 0).astype(np.uint8)


def _confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_road = prediction.reshape(-1) == 1
    gt_road = target.reshape(-1) == 1
    return {
        "tp": int(np.count_nonzero(pred_road & gt_road)),
        "fp": int(np.count_nonzero(pred_road & ~gt_road)),
        "fn": int(np.count_nonzero(~pred_road & gt_road)),
        "tn": int(np.count_nonzero(~pred_road & ~gt_road)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    eps = 1.0e-12
    road_iou = tp / max(tp + fp + fn, eps)
    background_iou = tn / max(tn + fp + fn, eps)
    return {
        "road_iou": float(road_iou),
        "background_iou": float(background_iou),
        "miou": float((road_iou + background_iou) * 0.5),
        "road_f1": float((2.0 * tp) / max(2.0 * tp + fp + fn, eps)),
        "road_precision": float(tp / max(tp + fp, eps)),
        "road_recall": float(tp / max(tp + fn, eps)),
        "pixel_accuracy": float((tp + tn) / max(tp + fp + fn + tn, eps)),
    }


def _score_one(
    item: tuple[int, str, str, str],
) -> dict[str, Any]:
    index, image_text, mask_text, prediction_text = item
    image_path = Path(image_text)
    mask_path = Path(mask_text)
    prediction_path = Path(prediction_text)
    prediction = _load_binary_mask(prediction_path)
    target = _load_binary_mask(mask_path)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch for {image_path.name}: "
            f"prediction={prediction.shape}, target={target.shape}"
        )
    counts = _confusion(prediction, target)
    return {
        "index": int(index),
        "image": str(image_path),
        "ground_truth": str(mask_path),
        "prediction": str(prediction_path),
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        **counts,
        **_metrics(counts),
        **score_binary_boundary_wfm(prediction, target),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_existing_metrics(
    existing: dict[str, Any],
    counts: dict[str, int],
    metrics: dict[str, float],
) -> dict[str, str]:
    validation: dict[str, str] = {}
    for key, value in counts.items():
        if key in existing and int(existing[key]) != int(value):
            raise RuntimeError(
                f"Saved prediction {key}={value} does not reproduce "
                f"metrics.json {key}={existing[key]}"
            )
        if key in existing:
            validation[key] = "exact_match"
    for key in ("road_iou", "background_iou", "miou"):
        if key not in existing:
            continue
        if not math.isclose(
            float(existing[key]),
            float(metrics[key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"Saved predictions reproduce {key}={metrics[key]}, but "
                f"metrics.json records {key}={existing[key]}"
            )
        validation[key] = "match_within_1e-12"
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute IDGBR-compatible 3-pixel boundary WFm from a complete "
            "saved RSKT-Seg CHN6-CUG prediction directory."
        )
    )
    parser.add_argument("evaluation_dir")
    parser.add_argument("--data_root", default="/root/data/CHN6-CUG/val")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected_num_samples", type=int, default=EXPECTED_NUM_SAMPLES)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.expected_num_samples <= 0:
        parser.error("--expected_num_samples must be positive")

    evaluation_dir = _path(args.evaluation_dir)
    prediction_dir = evaluation_dir / "pred_mask"
    if not prediction_dir.is_dir():
        raise NotADirectoryError(f"Missing saved prediction directory: {prediction_dir}")

    data_root = _path(args.data_root)
    image_dir, mask_dir, split = _resolve_chn6_dirs(data_root)
    images = _list_images(image_dir)
    if len(images) != args.expected_num_samples:
        raise RuntimeError(
            f"Expected {args.expected_num_samples} CHN6-CUG images, found "
            f"{len(images)} under {image_dir}"
        )

    work: list[tuple[int, str, str, str]] = []
    expected_predictions: set[Path] = set()
    for index, image_path in enumerate(images):
        mask_path = _find_mask(mask_dir, image_path)
        prediction_path = (
            prediction_dir / f"{index:06d}_{image_path.stem}_pred_mask.png"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing cached prediction: {prediction_path}")
        expected_predictions.add(prediction_path.resolve())
        work.append(
            (index, str(image_path), str(mask_path), str(prediction_path))
        )
    discovered_predictions = {
        path.resolve() for path in prediction_dir.glob("*_pred_mask.png")
    }
    unexpected = sorted(discovered_predictions - expected_predictions)
    if unexpected:
        preview = "\n  ".join(str(path) for path in unexpected[:10])
        raise RuntimeError(
            "Prediction directory contains files outside the complete CHN6-CUG "
            f"mapping:\n  {preview}"
        )

    if args.workers == 1:
        rows = [
            _score_one(item)
            for item in tqdm(work, desc="RSKT-Seg CHN6-CUG saved WFm")
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(
                tqdm(
                    executor.map(_score_one, work, chunksize=4),
                    total=len(work),
                    desc="RSKT-Seg CHN6-CUG saved WFm",
                )
            )

    counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for row in rows:
        for key in counts:
            counts[key] += int(row[key])
    metrics = _metrics(counts)
    boundary_metrics = aggregate_binary_boundary_wfm(rows)

    metrics_path = evaluation_dir / "metrics.json"
    existing: dict[str, Any] = {}
    validation: dict[str, str] = {}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        validation = _validate_existing_metrics(existing, counts, metrics)
        backup_path = evaluation_dir / "metrics_before_wfm.json"
        if not backup_path.exists():
            shutil.copy2(metrics_path, backup_path)

    result = {
        "dataset": "CHN6-CUG",
        "split": split,
        "method": "RSKT-Seg",
        "training_dataset": "DLRSD",
        "evaluation_setting": "cross-dataset/out-of-domain",
        "metric": "WFm",
        "protocol": "IDGBR_3px_boundary_WFm",
        "source": "saved_discrete_pred_mask_with_original_dataset_ground_truth",
        "evaluation_directory": str(evaluation_dir),
        "data_root": str(data_root),
        "prediction_root": str(prediction_dir),
        "num_samples": len(rows),
        "expected_num_samples": int(args.expected_num_samples),
        "complete_coverage": len(rows) == int(args.expected_num_samples),
        **counts,
        **metrics,
        **boundary_metrics,
        "existing_metrics_validation": validation,
    }
    updated_metrics = {
        **existing,
        **counts,
        **metrics,
        **boundary_metrics,
        "wfm_source": result["source"],
        "wfm_complete_coverage": result["complete_coverage"],
    }
    _write_json(evaluation_dir / "wfm_metrics.json", result)
    _write_csv(evaluation_dir / "wfm_per_image.csv", rows)
    _write_json(metrics_path, updated_metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[compute_saved_rskt_chn6_wfm] saved: {evaluation_dir / 'wfm_metrics.json'}")
    print(f"[compute_saved_rskt_chn6_wfm] updated: {metrics_path}")


if __name__ == "__main__":
    main()
