"""Shared FLAIR #1 test-set protocol for open-vocabulary segmentation.

The official release stores native 512 x 512 five-band aerial GeoTIFFs and
single-band label GeoTIFFs in ten domain archives.  This module owns dataset
discovery, the GSNet-compatible 12-class label mapping, RGB loading, and the
region-overlap metrics shared by the GSNet and RSKT-Seg adapters.

Model-specific resizing and forward passes deliberately do not live here.
The two evaluators must preserve their own published preprocessing while
sharing exactly the same samples, class order, ignore policy, and metrics.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np


# Raw FLAIR #1 labels are one-based; tuple index i names raw label i + 1.
FLAIR_RAW_CLASSES: Tuple[str, ...] = (
    "building",
    "pervious surface",
    "impervious surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
    "swimming pool",
    "snow",
    "clear cut",
    "mixed",
    "ligneous",
    "greenhouse",
    "other",
)

# Human-readable names used in tables and result files.
FLAIR_GSNET_CLASSES: Tuple[str, ...] = FLAIR_RAW_CLASSES[:12]

# Exact text vocabulary in the official GSNet datasets/flair.json file.
FLAIR_GSNET_MODEL_CLASSES: Tuple[str, ...] = (
    "building",
    "pervious-surface",
    "impervious-surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
)

# The fixed seed-0 farthest-RGB palette used by the Alters FLAIR evaluator.
# Sharing it makes saved qualitative masks directly comparable across models.
FLAIR_VISUAL_PALETTE_U8: Tuple[Tuple[int, int, int], ...] = (
    (255, 128, 128),
    (0, 0, 0),
    (0, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 255),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 255),
    (128, 255, 128),
)
FLAIR_IGNORE_VISUALIZATION_RGB: Tuple[int, int, int] = (127, 127, 127)

FLAIR1_TEST_DOMAIN_COUNTS: Mapping[str, int] = {
    "D012_2019": 1750,
    "D022_2021": 500,
    "D026_2020": 1825,
    "D064_2021": 1775,
    "D068_2021": 1775,
    "D071_2020": 1800,
    "D075_2021": 750,
    "D076_2019": 1750,
    "D083_2020": 2025,
    "D085_2019": 1750,
}
FLAIR1_EXPECTED_SAMPLES = 15_700
FLAIR1_EXPECTED_ZONES = 193
FLAIR1_NATIVE_SIZE = 512
IGNORE_INDEX = 255

_IMAGE_RE = re.compile(r"^IMG_(?P<sample>\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_MASK_RE = re.compile(r"^MSK_(?P<sample>\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?<![A-Z0-9])D(?P<number>\d{3})[_-](?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_ZONE_RE = re.compile(
    r"(?<![A-Z0-9])Z(?P<number>\d+)_(?P<code>[A-Z]+)(?![A-Z0-9])",
    re.IGNORECASE,
)
_EXPECTED_DOMAIN_ORDER = {
    name: index for index, name in enumerate(FLAIR1_TEST_DOMAIN_COUNTS)
}

PathLike = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class FlairRecord:
    """One paired official FLAIR #1 test patch."""

    sample_id: str
    domain: str
    zone: str
    image_path: Path
    mask_path: Path

    @property
    def output_name(self) -> str:
        """Return a collision-free stem shared by prediction and GT files."""

        return f"{self.domain}_{self.zone}_{self.sample_id}"


def normalize_wsl_path(path: PathLike) -> Path:
    """Resolve either a Linux path or a Windows WSL UNC path."""

    text = os.fspath(path).strip().strip("\"'").replace("\\", "/")
    match = re.match(
        r"^//wsl(?:\.localhost|\$)?/[^/]+(?P<path>/.*)$",
        text,
        re.IGNORECASE,
    )
    if match:
        text = match.group("path")
    return Path(text).expanduser()


def _domain_from_path(path: Path) -> str | None:
    match = _DOMAIN_RE.search(path.as_posix())
    if match is None:
        return None
    return f"D{match.group('number')}_{match.group('year')}".upper()


def _zone_from_path(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        match = _ZONE_RE.fullmatch(part)
        if match is not None:
            return f"Z{int(match.group('number'))}_{match.group('code').upper()}"
    match = _ZONE_RE.search(path.as_posix())
    if match is None:
        return None
    return f"Z{int(match.group('number'))}_{match.group('code').upper()}"


def _candidate_key(
    path: Path,
    filename_match: re.Match[str],
) -> Tuple[str, str, str] | None:
    domain = _domain_from_path(path)
    zone = _zone_from_path(path)
    if domain is None or zone is None:
        return None
    return domain, zone, filename_match.group("sample")


def _record_sort_key(record: FlairRecord) -> Tuple[int, int, str, int, str]:
    zone_match = _ZONE_RE.fullmatch(record.zone)
    zone_number = int(zone_match.group("number")) if zone_match else 10**9
    zone_code = zone_match.group("code") if zone_match else record.zone
    try:
        sample_number = int(record.sample_id)
    except ValueError:
        sample_number = 10**18
    return (
        _EXPECTED_DOMAIN_ORDER.get(record.domain, 10**9),
        zone_number,
        zone_code,
        sample_number,
        record.sample_id,
    )


def _preview_paths(paths: Iterable[Path], limit: int = 5) -> list[str]:
    return [str(path) for path in list(paths)[:limit]]


def discover_flair1_test(
    data_root: PathLike,
    strict: bool = True,
) -> tuple[list[FlairRecord], dict[str, Any]]:
    """Recursively pair official FLAIR patches by domain, zone, and patch ID.

    Strict mode requires the complete ten-domain, 193-zone, 15,700-patch test
    split.  ``strict=False`` is intended only for smoke tests.
    """

    root = normalize_wsl_path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"FLAIR data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"FLAIR data root is not a directory: {root}")

    all_files = [path for path in root.rglob("*") if path.is_file()]
    zip_paths = sorted(path for path in all_files if path.suffix.lower() == ".zip")
    image_paths = sorted(path for path in all_files if _IMAGE_RE.fullmatch(path.name))
    mask_paths = sorted(path for path in all_files if _MASK_RE.fullmatch(path.name))

    if not image_paths and not mask_paths:
        expected_archives = {
            path.stem.upper()
            for path in zip_paths
            if path.stem.upper() in FLAIR1_TEST_DOMAIN_COUNTS
        }
        if expected_archives:
            archive_list = ", ".join(sorted(expected_archives))
            raise RuntimeError(
                "FLAIR #1 archives are present but no extracted IMG_/MSK_ "
                f"GeoTIFFs were found under {root}. Found: {archive_list}. "
                "Extract every archive into a same-stem domain directory."
            )
        raise FileNotFoundError(
            f"No official FLAIR IMG_*.tif or MSK_*.tif files were found under {root}."
        )

    images: dict[Tuple[str, str, str], Path] = {}
    masks: dict[Tuple[str, str, str], Path] = {}
    unassigned_images: list[Path] = []
    unassigned_masks: list[Path] = []
    unexpected_domain_files: Counter[str] = Counter()

    def add_candidates(
        paths: Sequence[Path],
        pattern: re.Pattern[str],
        destination: dict[Tuple[str, str, str], Path],
        unassigned: list[Path],
        kind: str,
    ) -> None:
        for path in paths:
            filename_match = pattern.fullmatch(path.name)
            assert filename_match is not None
            key = _candidate_key(path, filename_match)
            if key is None:
                unassigned.append(path)
                continue
            if key[0] not in FLAIR1_TEST_DOMAIN_COUNTS:
                unexpected_domain_files[key[0]] += 1
                continue
            previous = destination.get(key)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate FLAIR {kind} for key {key}: {previous} and {path}"
                )
            destination[key] = path

    add_candidates(image_paths, _IMAGE_RE, images, unassigned_images, "image")
    add_candidates(mask_paths, _MASK_RE, masks, unassigned_masks, "mask")
    if unassigned_images or unassigned_masks:
        examples = _preview_paths(unassigned_images + unassigned_masks)
        raise RuntimeError(
            "Could not infer both official domain and zone for some FLAIR files. "
            "Do not flatten the domain archives. Examples: "
            f"{examples}"
        )

    image_keys = set(images)
    mask_keys = set(masks)
    missing_masks = sorted(image_keys - mask_keys)
    missing_images = sorted(mask_keys - image_keys)
    if missing_masks or missing_images:
        raise RuntimeError(
            "Incomplete FLAIR image/mask pairing: "
            f"{len(missing_masks)} images lack masks and "
            f"{len(missing_images)} masks lack images. "
            f"Missing-mask examples: {missing_masks[:5]}; "
            f"missing-image examples: {missing_images[:5]}."
        )

    records = [
        FlairRecord(
            sample_id=sample_id,
            domain=domain,
            zone=zone,
            image_path=images[(domain, zone, sample_id)],
            mask_path=masks[(domain, zone, sample_id)],
        )
        for domain, zone, sample_id in image_keys & mask_keys
    ]
    records.sort(key=_record_sort_key)

    domain_counts = Counter(record.domain for record in records)
    zones_by_domain: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        zones_by_domain[record.domain].add(record.zone)
    found_domains = set(domain_counts)
    expected_domains = set(FLAIR1_TEST_DOMAIN_COUNTS)
    count_mismatches = {
        domain: {
            "expected": FLAIR1_TEST_DOMAIN_COUNTS[domain],
            "found": domain_counts.get(domain, 0),
        }
        for domain in FLAIR1_TEST_DOMAIN_COUNTS
        if domain_counts.get(domain, 0) != FLAIR1_TEST_DOMAIN_COUNTS[domain]
    }
    num_zones = sum(len(zones) for zones in zones_by_domain.values())
    audit: dict[str, Any] = {
        "dataset": "FLAIR#1",
        "split": "flair#1-test",
        "evaluation_protocol": "GSNet FLAIR 12-class protocol",
        "resolved_data_root": str(root.resolve()),
        "strict_protocol": bool(strict),
        "expected_num_samples": FLAIR1_EXPECTED_SAMPLES,
        "num_pairs": len(records),
        "expected_num_zones": FLAIR1_EXPECTED_ZONES,
        "num_zones": num_zones,
        "expected_domain_counts": dict(FLAIR1_TEST_DOMAIN_COUNTS),
        "domain_counts": {
            domain: domain_counts.get(domain, 0)
            for domain in FLAIR1_TEST_DOMAIN_COUNTS
        },
        "zones_by_domain": {
            domain: sorted(zones_by_domain.get(domain, set()))
            for domain in FLAIR1_TEST_DOMAIN_COUNTS
        },
        "missing_domains": sorted(expected_domains - found_domains),
        "unexpected_domains_ignored": dict(sorted(unexpected_domain_files.items())),
        "domain_count_mismatches": count_mismatches,
        "num_discovered_img_files": len(image_paths),
        "num_discovered_msk_files": len(mask_paths),
        "zip_archives": [str(path) for path in zip_paths],
        "classes": list(FLAIR_GSNET_CLASSES),
        "model_classes": list(FLAIR_GSNET_MODEL_CLASSES),
        "raw_to_eval_mapping": {str(raw_id): raw_id - 1 for raw_id in range(1, 13)},
        "ignored_raw_labels": [0, *range(13, 20), 255],
        "native_patch_size": [FLAIR1_NATIVE_SIZE, FLAIR1_NATIVE_SIZE],
        "rgb_bands": [1, 2, 3],
    }

    if strict:
        errors: list[str] = []
        if found_domains != expected_domains:
            errors.append(
                f"expected domains {sorted(expected_domains)}, found {sorted(found_domains)}"
            )
        if count_mismatches:
            errors.append(f"per-domain count mismatches: {count_mismatches}")
        if len(records) != FLAIR1_EXPECTED_SAMPLES:
            errors.append(
                f"expected {FLAIR1_EXPECTED_SAMPLES} pairs, found {len(records)}"
            )
        if num_zones != FLAIR1_EXPECTED_ZONES:
            errors.append(
                f"expected {FLAIR1_EXPECTED_ZONES} zones, found {num_zones}"
            )
        if errors:
            raise RuntimeError(
                "Invalid official FLAIR #1 test split: " + "; ".join(errors)
            )
    return records, audit


def _read_rgb_chw(path: PathLike) -> np.ndarray:
    path = Path(path)
    errors: list[str] = []
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as dataset:
            if dataset.count != 5:
                raise ValueError(
                    f"expected the official 5 bands, found {dataset.count}"
                )
            return np.asarray(dataset.read([1, 2, 3]))
    except Exception as exc:  # fallback is intentional for lightweight envs
        errors.append(f"rasterio: {exc}")

    try:
        import tifffile  # type: ignore

        array = np.asarray(tifffile.imread(path))
        if array.ndim != 3:
            raise ValueError(
                f"expected a 3-D multi-band TIFF, got shape {array.shape}"
            )
        if array.shape[0] == 5 and array.shape[-1] != 5:
            return array[:3]
        if array.shape[-1] == 5:
            return np.moveaxis(array[..., :3], -1, 0)
        raise ValueError(
            f"expected one channel axis of length 5, got TIFF shape {array.shape}"
        )
    except Exception as exc:
        errors.append(f"tifffile: {exc}")

    raise RuntimeError(
        "Unable to read the official five-band FLAIR image with rasterio or "
        f"tifffile ({path}): {'; '.join(errors)}"
    )


def load_flair_rgb_u8(image_path: PathLike) -> np.ndarray:
    """Read official bands 1--3 as an HWC uint8 RGB array."""

    path = Path(image_path)
    array = _read_rgb_chw(path)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB data for {path}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected numeric RGB data for {path}, got dtype {array.dtype}")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"FLAIR RGB image contains NaN/Inf values: {path}")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            f"Expected official 8-bit RGB values in [0,255] for {path}, "
            f"found range [{minimum},{maximum}]"
        )
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        values *= 255.0
    rgb = np.rint(values).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.moveaxis(rgb, 0, -1))


def load_flair_rgb(image_path: PathLike):
    """Read bands 1--3 as a CHW torch.float32 tensor in [0,1]."""

    import torch

    rgb_hwc = load_flair_rgb_u8(image_path)
    rgb_chw = np.moveaxis(rgb_hwc.astype(np.float32) / 255.0, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(rgb_chw))


def _read_mask_array(path: PathLike) -> np.ndarray:
    path = Path(path)
    errors: list[str] = []
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as dataset:
            if dataset.count < 1:
                raise ValueError("mask has no raster bands")
            return np.asarray(dataset.read(1))
    except Exception as exc:
        errors.append(f"rasterio: {exc}")

    try:
        import tifffile  # type: ignore

        array = np.asarray(tifffile.imread(path))
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"expected a single-band mask, got shape {array.shape}")
        return array
    except Exception as exc:
        errors.append(f"tifffile: {exc}")

    try:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"expected a single-band mask, got shape {array.shape}")
        return array
    except Exception as exc:
        errors.append(f"PIL: {exc}")
    raise RuntimeError(f"Unable to read FLAIR mask {path}: {'; '.join(errors)}")


def map_flair_raw_mask(
    raw_mask: np.ndarray,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Map raw IDs 1..12 to 0..11 and all other official IDs to ignore."""

    array = np.asarray(raw_mask)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected numeric FLAIR mask, got dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("FLAIR mask contains NaN/Inf values")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError("FLAIR mask contains non-integral labels")
        array = rounded
    raw = np.asarray(array, dtype=np.int64)
    invalid = (raw < 0) | ((raw > 19) & (raw != 255))
    if bool(np.any(invalid)):
        unexpected = [int(value) for value in np.unique(raw[invalid])]
        raise ValueError(
            f"Unexpected raw FLAIR label IDs: {unexpected}; expected 0..19 or 255"
        )
    mapped = np.full(raw.shape, int(ignore_index), dtype=np.int64)
    retained = (raw >= 1) & (raw <= 12)
    mapped[retained] = raw[retained] - 1
    return mapped


def load_flair_mask_array(
    mask_path: PathLike,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Read and map one FLAIR mask to an HW int64 array."""

    return np.ascontiguousarray(
        map_flair_raw_mask(_read_mask_array(mask_path), ignore_index=ignore_index)
    )


def load_flair_mask(mask_path: PathLike, ignore_index: int = IGNORE_INDEX):
    """Read and map one FLAIR mask to an HW torch.long tensor."""

    import torch

    return torch.from_numpy(
        load_flair_mask_array(mask_path, ignore_index=ignore_index)
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def flair_confusion_matrix(
    prediction: Any,
    target: Any,
    *,
    num_classes: int = len(FLAIR_GSNET_CLASSES),
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Build a GT-row/prediction-column confusion matrix on valid pixels."""

    pred = _to_numpy(prediction)
    gt = _to_numpy(target)
    if pred.shape != gt.shape:
        raise ValueError(
            f"FLAIR prediction/target shape mismatch: {pred.shape} vs {gt.shape}"
        )
    pred = pred.astype(np.int64, copy=False).reshape(-1)
    gt = gt.astype(np.int64, copy=False).reshape(-1)
    valid = (
        (gt != int(ignore_index))
        & (gt >= 0)
        & (gt < int(num_classes))
        & (pred >= 0)
        & (pred < int(num_classes))
    )
    if not bool(np.any(valid)):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    indices = gt[valid] * int(num_classes) + pred[valid]
    return np.bincount(
        indices,
        minlength=int(num_classes) ** 2,
    ).reshape(num_classes, num_classes).astype(np.int64)


def flair_metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    """Compute mIoU, mF1, accuracy, and class metrics from a confusion matrix."""

    conf = np.asarray(confusion, dtype=np.float64)
    expected = (len(FLAIR_GSNET_CLASSES), len(FLAIR_GSNET_CLASSES))
    if conf.shape != expected:
        raise ValueError(f"FLAIR confusion matrix must be {expected}, got {conf.shape}")
    true_positive = np.diag(conf)
    gt_count = conf.sum(axis=1)
    pred_count = conf.sum(axis=0)
    iou_denom = gt_count + pred_count - true_positive
    f1_denom = gt_count + pred_count
    iou = np.divide(
        true_positive,
        iou_denom,
        out=np.full_like(true_positive, np.nan),
        where=iou_denom > 0,
    )
    f1 = np.divide(
        2.0 * true_positive,
        f1_denom,
        out=np.full_like(true_positive, np.nan),
        where=f1_denom > 0,
    )
    accuracy = np.divide(
        true_positive,
        gt_count,
        out=np.full_like(true_positive, np.nan),
        where=gt_count > 0,
    )
    metrics: dict[str, Any] = {
        "miou": float(np.nanmean(iou)) if bool(np.any(iou_denom > 0)) else 0.0,
        "mf1": float(np.nanmean(f1)) if bool(np.any(f1_denom > 0)) else 0.0,
        "macc": (
            float(np.nanmean(accuracy)) if bool(np.any(gt_count > 0)) else 0.0
        ),
        "pixel_accuracy": float(true_positive.sum() / max(conf.sum(), 1.0)),
        "valid_pixels": int(conf.sum()),
    }
    for class_id, class_name in enumerate(FLAIR_GSNET_CLASSES):
        key = class_name.replace(" ", "_").replace("-", "_")
        metrics[f"iou_{key}"] = (
            None if np.isnan(iou[class_id]) else float(iou[class_id])
        )
        metrics[f"f1_{key}"] = (
            None if np.isnan(f1[class_id]) else float(f1[class_id])
        )
        metrics[f"acc_{key}"] = (
            None if np.isnan(accuracy[class_id]) else float(accuracy[class_id])
        )
        metrics[f"accuracy_{key}"] = metrics[f"acc_{key}"]
    return metrics


__all__ = [
    "FLAIR1_EXPECTED_SAMPLES",
    "FLAIR1_EXPECTED_ZONES",
    "FLAIR1_NATIVE_SIZE",
    "FLAIR1_TEST_DOMAIN_COUNTS",
    "FLAIR_GSNET_CLASSES",
    "FLAIR_GSNET_MODEL_CLASSES",
    "FLAIR_IGNORE_VISUALIZATION_RGB",
    "FLAIR_RAW_CLASSES",
    "FLAIR_VISUAL_PALETTE_U8",
    "IGNORE_INDEX",
    "FlairRecord",
    "discover_flair1_test",
    "flair_confusion_matrix",
    "flair_metrics_from_confusion",
    "load_flair_mask",
    "load_flair_mask_array",
    "load_flair_rgb",
    "load_flair_rgb_u8",
    "map_flair_raw_mask",
    "normalize_wsl_path",
]
