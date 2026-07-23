"""Dataset and metric protocols for standalone SegEarth-OV evaluation.

The model adapter deliberately lives in ``eval_segearth_ov.py``.  This module
contains only CPU-testable dataset discovery, label decoding, visualization,
and confusion-matrix logic shared by LoveDA, FLAIR #1, UAVid, xBD-pre, and
CHN6-CUG.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


from baselines.flair_protocol import (
    FLAIR1_EXPECTED_SAMPLES,
    FLAIR_GSNET_CLASSES,
    FLAIR_GSNET_MODEL_CLASSES,
    FLAIR_IGNORE_VISUALIZATION_RGB,
    FLAIR_VISUAL_PALETTE_U8,
    IGNORE_INDEX,
    discover_flair1_test,
    load_flair_mask_array,
    load_flair_rgb_u8,
)


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

LOVEDA_CLASSES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
)
LOVEDA_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)
UAVID_CLASSES = (
    "background clutter",
    "building",
    "road",
    "tree",
    "low vegetation",
    "moving car",
    "static car",
    "human",
)
UAVID_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [128, 0, 0],
        [128, 64, 128],
        [0, 128, 0],
        [128, 128, 0],
        [64, 0, 128],
        [192, 0, 192],
        [64, 64, 0],
    ],
    dtype=np.uint8,
)
BINARY_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "loveda": {
        "display_name": "LoveDA",
        "split": "Val",
        "classes": LOVEDA_CLASSES,
        "text_groups": (
            ("background",),
            ("building", "roof", "house"),
            ("road",),
            ("water",),
            ("barren",),
            ("forest",),
            ("agricultural",),
        ),
        "palette": LOVEDA_PALETTE,
        "ignore_index": IGNORE_INDEX,
        "expected_samples": 1669,
        "probability_threshold": 0.3,
        "cls_token_lambda": -0.3,
        "primary_metric": "miou",
        "label_protocol": "raw 0 ignored; raw 1..7 mapped to evaluation IDs 0..6",
    },
    "flair": {
        "display_name": "FLAIR#1",
        "split": "flair#1-test",
        "classes": tuple(FLAIR_GSNET_CLASSES),
        "text_groups": tuple((name,) for name in FLAIR_GSNET_MODEL_CLASSES),
        "palette": np.asarray(FLAIR_VISUAL_PALETTE_U8, dtype=np.uint8),
        "ignore_index": IGNORE_INDEX,
        "expected_samples": FLAIR1_EXPECTED_SAMPLES,
        "probability_threshold": 0.0,
        "cls_token_lambda": -0.3,
        "primary_metric": "miou",
        "label_protocol": "raw 1..12 mapped to 0..11; raw 0,13..19,255 ignored",
    },
    "uavid": {
        "display_name": "UAVid",
        "split": "OVSISBench/all 270 paired assets",
        "classes": UAVID_CLASSES,
        "text_groups": tuple((name,) for name in UAVID_CLASSES),
        "palette": UAVID_PALETTE,
        "ignore_index": IGNORE_INDEX,
        "expected_samples": 270,
        "probability_threshold": 0.3,
        "cls_token_lambda": -0.3,
        "primary_metric": "miou",
        "label_protocol": (
            "VISTAR/OVSISBench eight-class order; official RGB, zero-based "
            "0..7, or one-based 1..8 masks; 255 ignored; moving and static "
            "cars remain separate"
        ),
    },
    "xbd_pre": {
        "display_name": "xBD-pre",
        "split": "test/pre-disaster",
        "classes": ("background", "building"),
        "text_groups": (("background",), ("building",)),
        "palette": BINARY_PALETTE,
        "ignore_index": None,
        "expected_samples": 933,
        "probability_threshold": 0.0,
        "cls_token_lambda": 0.0,
        "primary_metric": "building_iou",
        "label_protocol": "features.xy WKT rounded then cv2.fillPoly; nonzero is building",
    },
    "chn6_cug": {
        "display_name": "CHN6-CUG",
        "split": "val",
        "classes": ("background", "road"),
        "text_groups": (("background",), ("road",)),
        "palette": BINARY_PALETTE,
        "ignore_index": None,
        "expected_samples": 903,
        "probability_threshold": 0.8,
        "cls_token_lambda": -0.3,
        "primary_metric": "road_iou",
        "label_protocol": "zero is background; every nonzero mask value is road",
    },
}


@dataclass(frozen=True)
class EvalSample:
    """One image/ground-truth pair with a collision-free output name."""

    name: str
    image_path: Path
    mask_path: Path
    domain: str = ""
    zone: str = ""
    sample_id: str = ""
    mask_id_base: str = ""


def normalize_path(value: str | Path) -> Path:
    """Resolve Linux paths and WSL UNC paths without touching the filesystem."""

    text = str(value).strip().strip("\"'").replace("\\", "/")
    match = re.match(r"^//wsl(?:\.localhost|\$)?/[^/]+(?P<path>/.*)$", text, re.I)
    if match:
        text = match.group("path")
    return Path(text).expanduser().resolve()


def read_class_groups(path: Path) -> tuple[tuple[str, ...], ...]:
    """Read SegEarth-OV's one-class-per-line, comma-alias vocabulary format."""

    groups: list[tuple[str, ...]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            raise ValueError(
                "Official SegEarth-OV treats every physical line as a class; "
                f"blank/comment lines are not allowed: {path}:{line_number}"
            )
        raw_aliases = line.split(",")
        aliases = tuple(alias.strip() for alias in raw_aliases)
        if any(not alias for alias in aliases):
            raise ValueError(f"Empty class alias in {path}:{line_number}")
        if list(raw_aliases) != list(aliases):
            raise ValueError(
                "Whitespace around comma-separated aliases changes the official "
                f"text prompt: {path}:{line_number}"
            )
        groups.append(aliases)
    if not groups:
        raise ValueError(f"SegEarth-OV class file is empty: {path}")
    return tuple(groups)


def validate_class_groups(
    dataset: str,
    groups: Sequence[Sequence[str]],
) -> None:
    """Require the output-class order needed by each fixed GT protocol."""

    expected = tuple(
        tuple(str(alias).casefold() for alias in group)
        for group in DATASET_SPECS[dataset]["text_groups"]
    )
    actual = tuple(
        tuple(str(alias).casefold() for alias in group)
        for group in groups
    )
    if actual != expected:
        raise ValueError(
            f"Class-file groups for {dataset} must be {list(expected)}, got {list(actual)}"
        )


def _list_images(directory: Path, *, pre_disaster_only: bool = False) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    images = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (not pre_disaster_only or path.stem.endswith("_pre_disaster"))
    ]
    if not images:
        qualifier = " pre-disaster" if pre_disaster_only else ""
        raise FileNotFoundError(f"No{qualifier} images found under {directory}")
    return images


def _find_same_stem(directory: Path, stem: str) -> Path:
    for extension in MASK_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask with stem {stem!r} under {directory}")


def _normalize_uavid_pair_stem(stem: str) -> str:
    """Remove image/label role tokens while preserving sequence and frame IDs."""

    normalized = str(stem).strip().casefold()
    normalized = re.sub(
        r"(^|[_.-])(images?|imgs?|labels?|masks?|annotations?)(?=$|[_.-])",
        r"\1",
        normalized,
    )
    normalized = re.sub(r"[_.-]+", "_", normalized)
    return normalized.strip("_")


def _resolve_uavid_dirs(data_root: Path) -> tuple[Path, Path, str]:
    candidates = (
        (data_root / "Images", data_root / "Labels", "direct"),
        (data_root / "images", data_root / "labels", "direct-lowercase"),
        (data_root / "uavid" / "Images", data_root / "uavid" / "Labels", "nested-uavid"),
        (data_root / "UAVid" / "Images", data_root / "UAVid" / "Labels", "nested-UAVid"),
        (data_root / "test" / "Images", data_root / "test" / "Labels", "test-nested"),
        (data_root / "Test" / "Images", data_root / "Test" / "Labels", "Test-nested"),
    )
    for image_dir, mask_dir, layout in candidates:
        if image_dir.is_dir() and mask_dir.is_dir():
            return image_dir, mask_dir, layout
    detail = "\n".join(f"  images={images} labels={labels}" for images, labels, _ in candidates)
    raise NotADirectoryError(
        "Cannot find the VISTAR/OVSISBench UAVid Images/Labels layout:\n" + detail
    )


def _list_masks(directory: Path) -> list[Path]:
    masks = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.casefold() in MASK_EXTENSIONS
    ]
    if not masks:
        raise FileNotFoundError(f"No UAVid labels found under {directory}")
    return masks


def _find_uavid_mask(
    image_path: Path,
    *,
    by_stem: dict[str, list[Path]],
    by_normalized_stem: dict[str, list[Path]],
) -> Path:
    candidate_stems = [image_path.stem]
    for source_role, target_role in (
        ("images", "Labels"),
        ("image", "Label"),
        ("imgs", "Labels"),
        ("img", "Label"),
    ):
        replaced = re.sub(
            rf"(^|[_.-]){source_role}(?=$|[_.-])",
            rf"\1{target_role}",
            image_path.stem,
            flags=re.IGNORECASE,
        )
        if replaced != image_path.stem:
            candidate_stems.append(replaced)

    for candidate_stem in candidate_stems:
        matches = by_stem.get(candidate_stem.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple UAVid labels match {image_path.name} via stem "
                f"{candidate_stem!r}: " + ", ".join(path.name for path in matches)
            )

    normalized = _normalize_uavid_pair_stem(image_path.stem)
    matches = by_normalized_stem.get(normalized, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple UAVid labels match normalized frame ID {normalized!r} "
            f"for {image_path.name}: " + ", ".join(path.name for path in matches)
        )
    raise FileNotFoundError(
        f"Cannot pair UAVid image {image_path.name} with a label. Tried exact "
        f"stem, Images-to-Labels substitution, and normalized frame ID {normalized!r}."
    )


def _audit_uavid_mask_encoding(
    mask_paths: Sequence[Path],
    requested_base: str,
) -> tuple[str, dict[str, Any]]:
    if requested_base not in {"auto", "zero", "one"}:
        raise ValueError(
            f"UAVid mask_id_base must be auto, zero, or one; got {requested_base!r}"
        )

    official_colors = {
        tuple(int(channel) for channel in color)
        for color in UAVID_PALETTE.tolist()
    }
    indexed_values: set[int] = set()
    invalid_rgb_colors: set[tuple[int, int, int]] = set()
    pil_modes: dict[str, int] = {}
    indexed_files = 0
    rgb_files = 0
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            pil_modes[image.mode] = pil_modes.get(image.mode, 0) + 1
            array = np.asarray(image)
        if array.ndim == 2:
            indexed_files += 1
            indexed_values.update(
                int(value)
                for value in np.unique(array.astype(np.int64, copy=False)).tolist()
                if int(value) != IGNORE_INDEX
            )
        elif array.ndim == 3 and array.shape[-1] >= 3:
            rgb_files += 1
            for color in np.unique(array[..., :3].reshape(-1, 3), axis=0).tolist():
                key = tuple(int(channel) for channel in color)
                if key not in official_colors:
                    invalid_rgb_colors.add(key)
                    if len(invalid_rgb_colors) >= 32:
                        break
        else:
            raise ValueError(f"Unsupported UAVid mask shape {array.shape}: {mask_path}")

    if invalid_rgb_colors:
        raise ValueError(
            "UAVid RGB masks contain colors outside the fixed VISTAR eight-class "
            f"palette; examples={sorted(invalid_rgb_colors)[:10]}"
        )

    zero_values = set(range(len(UAVID_CLASSES)))
    one_values = set(range(1, len(UAVID_CLASSES) + 1))
    zero_ok = bool(indexed_values) and indexed_values.issubset(zero_values)
    one_ok = bool(indexed_values) and indexed_values.issubset(one_values)
    if requested_base == "auto":
        if indexed_files == 0:
            resolved_base = "zero"
            decision = "all masks use the official eight-color RGB palette"
        elif zero_ok and not one_ok:
            resolved_base = "zero"
            decision = "indexed labels match contiguous class IDs 0..7"
        elif one_ok and not zero_ok:
            resolved_base = "one"
            decision = "indexed labels match dataset IDs 1..8 and are shifted to 0..7"
        elif zero_ok and one_ok:
            raise ValueError(
                "Cannot auto-resolve UAVid indexed-mask base because observed values "
                f"{sorted(indexed_values)} fit both 0-based and 1-based encodings. "
                "Set MASK_ID_BASE=zero or MASK_ID_BASE=one explicitly."
            )
        else:
            raise ValueError(
                "UAVid indexed labels do not match 0..7 or 1..8: "
                f"observed={sorted(indexed_values)}"
            )
    else:
        resolved_base = requested_base
        allowed = zero_values if resolved_base == "zero" else one_values
        if indexed_values and not indexed_values.issubset(allowed):
            raise ValueError(
                f"MASK_ID_BASE={resolved_base} conflicts with UAVid values "
                f"{sorted(indexed_values)}; expected a subset of {sorted(allowed)}"
            )
        decision = f"mask base explicitly set to {resolved_base}"

    return resolved_base, {
        "num_masks": len(mask_paths),
        "pil_modes": pil_modes,
        "indexed_files": indexed_files,
        "rgb_files": rgb_files,
        "indexed_values_excluding_ignore": sorted(indexed_values),
        "ignore_index": IGNORE_INDEX,
        "requested_mask_id_base": requested_base,
        "resolved_mask_id_base": resolved_base,
        "decision": decision,
    }


def discover_uavid(
    data_root: Path,
    *,
    mask_id_base: str = "auto",
) -> tuple[list[EvalSample], dict[str, Any]]:
    """Discover every paired asset in the VISTAR/OVSISBench UAVid layout."""

    image_dir, mask_dir, layout = _resolve_uavid_dirs(data_root)
    images = _list_images(image_dir)
    masks = _list_masks(mask_dir)
    by_stem: dict[str, list[Path]] = {}
    by_normalized_stem: dict[str, list[Path]] = {}
    for mask_path in masks:
        by_stem.setdefault(mask_path.stem.casefold(), []).append(mask_path)
        normalized = _normalize_uavid_pair_stem(mask_path.stem)
        by_normalized_stem.setdefault(normalized, []).append(mask_path)

    pairs = [
        (
            image_path,
            _find_uavid_mask(
                image_path,
                by_stem=by_stem,
                by_normalized_stem=by_normalized_stem,
            ),
        )
        for image_path in images
    ]
    used_masks = [mask_path.resolve() for _, mask_path in pairs]
    if len(set(used_masks)) != len(used_masks):
        raise RuntimeError("Multiple UAVid images resolved to the same label")
    unused_masks = sorted(set(path.resolve() for path in masks) - set(used_masks))
    if unused_masks:
        preview = ", ".join(path.name for path in unused_masks[:20])
        raise RuntimeError(
            f"UAVid Labels contains {len(unused_masks)} unpaired masks: {preview}"
        )

    resolved_base, mask_audit = _audit_uavid_mask_encoding(masks, mask_id_base)
    records: list[EvalSample] = []
    domain_counts: dict[str, int] = {}
    for image_path, mask_path in pairs:
        normalized = _normalize_uavid_pair_stem(image_path.stem)
        domain_match = re.match(r"^(seq[^_]+)", normalized, flags=re.IGNORECASE)
        domain = domain_match.group(1) if domain_match else ""
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        records.append(
            EvalSample(
                name=f"UAVid_{normalized}",
                image_path=image_path,
                mask_path=mask_path,
                domain=domain,
                sample_id=normalized,
                mask_id_base=resolved_base,
            )
        )
    records.sort(key=lambda record: record.name.casefold())
    return records, {
        "dataset": "UAVid",
        "split": "OVSISBench/all 270 paired assets",
        "resolved_data_root": str(data_root),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "layout": layout,
        "num_pairs": len(records),
        "num_images": len(images),
        "num_masks": len(masks),
        "complete_pairing": len(records) == len(images) == len(masks),
        "domain_counts": domain_counts,
        "mask_encoding": mask_audit,
    }


def discover_loveda(data_root: Path) -> tuple[list[EvalSample], dict[str, Any]]:
    """Discover the official 1,669-image LoveDA validation split."""

    candidates: list[tuple[str, Path, Path]] = []
    validation_root = (
        data_root if data_root.name.casefold() == "val" else data_root / "Val"
    )
    for domain in ("Urban", "Rural"):
        image_dir = validation_root / domain / "images_png"
        mask_dir = validation_root / domain / "masks_png"
        if image_dir.is_dir() and mask_dir.is_dir():
            candidates.append((domain, image_dir, mask_dir))
    if not candidates:
        layouts = (
            ("Val", data_root / "img_dir" / "val", data_root / "ann_dir" / "val"),
            (data_root.name, data_root / "images_png", data_root / "masks_png"),
        )
        for domain, image_dir, mask_dir in layouts:
            if image_dir.is_dir() and mask_dir.is_dir():
                candidates.append((domain, image_dir, mask_dir))
                break
    if not candidates:
        raise NotADirectoryError(
            "Cannot find LoveDA validation data. Expected "
            f"{validation_root}/{{Urban,Rural}}/{{images_png,masks_png}} or the "
            "MMSeg img_dir/val + ann_dir/val layout."
        )

    records: list[EvalSample] = []
    domain_counts: dict[str, int] = {}
    for domain, image_dir, mask_dir in candidates:
        images = _list_images(image_dir)
        domain_counts[domain] = len(images)
        for image_path in images:
            mask_path = _find_same_stem(mask_dir, image_path.stem)
            safe_domain = re.sub(r"[^A-Za-z0-9_-]+", "_", domain)
            records.append(
                EvalSample(
                    name=f"Val_{safe_domain}_{image_path.stem}",
                    image_path=image_path,
                    mask_path=mask_path,
                    domain=domain,
                    sample_id=image_path.stem,
                )
            )
    records.sort(key=lambda record: (record.domain.casefold(), record.image_path.name))
    return records, {
        "dataset": "LoveDA",
        "split": "Val",
        "resolved_data_root": str(data_root),
        "num_pairs": len(records),
        "domain_counts": domain_counts,
        "domains": list(domain_counts),
    }


def discover_flair(data_root: Path, *, strict: bool) -> tuple[list[EvalSample], dict[str, Any]]:
    records, audit = discover_flair1_test(data_root, strict=strict)
    return [
        EvalSample(
            name=record.output_name,
            image_path=record.image_path,
            mask_path=record.mask_path,
            domain=record.domain,
            zone=record.zone,
            sample_id=record.sample_id,
        )
        for record in records
    ], audit


def _resolve_xbd_dirs(data_root: Path) -> tuple[Path, Path, str]:
    split_roots = [(data_root, data_root.name)]
    if data_root.name.casefold() != "test":
        split_roots.append((data_root / "test", "test"))
    checked: list[tuple[Path, Path]] = []
    layouts = (
        ("images", "labels"),
        ("images", "targets"),
        ("images", "targets_cvt"),
        ("images", "masks_building"),
        ("images_pre", "targets_cvt_pre"),
        ("images_pre", "labels_pre"),
    )
    for split_root, split_name in split_roots:
        for image_name, label_name in layouts:
            image_dir = split_root / image_name
            label_dir = split_root / label_name
            checked.append((image_dir, label_dir))
            if image_dir.is_dir() and label_dir.is_dir():
                return image_dir, label_dir, split_name
    detail = "\n".join(f"  images={a} labels={b}" for a, b in checked)
    raise NotADirectoryError(f"Cannot find xBD test image/label folders:\n{detail}")


def _find_xbd_mask(label_dir: Path, image_path: Path) -> Path:
    candidates = [label_dir / f"{image_path.stem}.json"]
    for extension in MASK_EXTENSIONS:
        candidates.extend(
            (
                label_dir / f"{image_path.stem}{extension}",
                label_dir / f"{image_path.stem}_target{extension}",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No xBD label matching {image_path.name} under {label_dir}")


def discover_xbd_pre(data_root: Path) -> tuple[list[EvalSample], dict[str, Any]]:
    image_dir, label_dir, split = _resolve_xbd_dirs(data_root)
    images = _list_images(
        image_dir,
        pre_disaster_only=image_dir.name.casefold() != "images_pre",
    )
    records = [
        EvalSample(
            name=image_path.stem,
            image_path=image_path,
            mask_path=_find_xbd_mask(label_dir, image_path),
            sample_id=image_path.stem,
        )
        for image_path in images
    ]
    return records, {
        "dataset": "xBD-pre",
        "split": split,
        "resolved_data_root": str(data_root),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "num_pairs": len(records),
    }


def _resolve_chn6_dirs(data_root: Path) -> tuple[Path, Path, str]:
    split_roots = [(data_root, data_root.name), (data_root / "val", "val")]
    layouts = (
        ("images", "gt"),
        ("images", "labels"),
        ("images", "masks"),
        ("image", "label"),
        ("imgs", "gt"),
        ("imgs", "masks"),
        ("image_cvt", "label_cvt"),
    )
    checked: list[tuple[Path, Path]] = []
    for split_root, split_name in split_roots:
        for image_name, mask_name in layouts:
            image_dir = split_root / image_name
            mask_dir = split_root / mask_name
            checked.append((image_dir, mask_dir))
            if image_dir.is_dir() and mask_dir.is_dir():
                return image_dir, mask_dir, split_name
    detail = "\n".join(f"  images={a} masks={b}" for a, b in checked)
    raise NotADirectoryError(f"Cannot find CHN6-CUG image/mask folders:\n{detail}")


def _find_chn6_mask(mask_dir: Path, image_path: Path) -> Path:
    stems = [image_path.stem]
    if "_sat" in image_path.stem:
        stems.insert(0, image_path.stem.replace("_sat", "_mask"))
    for stem in stems:
        try:
            return _find_same_stem(mask_dir, stem)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"No CHN6-CUG mask matching {image_path.name} under {mask_dir}")


def discover_chn6_cug(data_root: Path) -> tuple[list[EvalSample], dict[str, Any]]:
    image_dir, mask_dir, split = _resolve_chn6_dirs(data_root)
    images = _list_images(image_dir)
    records = [
        EvalSample(
            name=image_path.stem,
            image_path=image_path,
            mask_path=_find_chn6_mask(mask_dir, image_path),
            sample_id=image_path.stem,
        )
        for image_path in images
    ]
    return records, {
        "dataset": "CHN6-CUG",
        "split": split,
        "resolved_data_root": str(data_root),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "num_pairs": len(records),
    }


def discover_dataset(
    dataset: str,
    data_root: Path,
    *,
    strict: bool,
    mask_id_base: str = "auto",
) -> tuple[list[EvalSample], dict[str, Any]]:
    if dataset == "loveda":
        records, audit = discover_loveda(data_root)
    elif dataset == "flair":
        records, audit = discover_flair(data_root, strict=strict)
    elif dataset == "uavid":
        records, audit = discover_uavid(data_root, mask_id_base=mask_id_base)
    elif dataset == "xbd_pre":
        records, audit = discover_xbd_pre(data_root)
    elif dataset == "chn6_cug":
        records, audit = discover_chn6_cug(data_root)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"Unsupported dataset: {dataset}")
    expected = int(DATASET_SPECS[dataset]["expected_samples"])
    if strict and len(records) != expected:
        raise RuntimeError(
            f"Invalid full {DATASET_SPECS[dataset]['display_name']} population: "
            f"expected {expected} samples, found {len(records)}"
        )
    audit = dict(audit)
    audit.update({"strict_protocol": bool(strict), "expected_num_samples": expected})
    return records, audit


def load_rgb(sample: EvalSample, dataset: str) -> np.ndarray:
    if dataset == "flair":
        rgb = load_flair_rgb_u8(sample.image_path)
    else:
        with Image.open(sample.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {sample.image_path}, got {rgb.shape}")
    return np.ascontiguousarray(rgb)


def _decode_loveda_rgb(mask_rgb: np.ndarray) -> np.ndarray:
    # int16 would overflow when squaring a channel difference of 255.
    values = np.asarray(mask_rgb, dtype=np.int32)[..., :3]
    difference = values[..., None, :] - LOVEDA_PALETTE.astype(np.int32)[None, None]
    squared_distance = np.sum(difference * difference, axis=-1)
    matched = np.min(squared_distance, axis=-1) == 0
    if not bool(np.all(matched)):
        unknown = np.unique(values[~matched].reshape(-1, 3), axis=0)[:20].tolist()
        raise ValueError(f"LoveDA RGB mask contains unsupported colors: {unknown}")
    return squared_distance.argmin(axis=-1).astype(np.int64)


def _load_loveda_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        return _decode_loveda_rgb(array)
    if array.ndim != 2:
        raise ValueError(f"Unsupported LoveDA mask shape {array.shape}: {path}")
    raw = np.asarray(array, dtype=np.int64)
    unexpected = (raw < 0) | ((raw > 7) & (raw != IGNORE_INDEX))
    if bool(np.any(unexpected)):
        raise ValueError(f"Unexpected LoveDA raw IDs {np.unique(raw[unexpected]).tolist()}: {path}")
    mapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (raw >= 1) & (raw <= 7)
    mapped[valid] = raw[valid] - 1
    return mapped


def _decode_uavid_rgb(mask_rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(mask_rgb, dtype=np.int32)[..., :3]
    difference = values[..., None, :] - UAVID_PALETTE.astype(np.int32)[None, None]
    squared_distance = np.sum(difference * difference, axis=-1)
    matched = np.min(squared_distance, axis=-1) == 0
    if not bool(np.all(matched)):
        unknown = np.unique(values[~matched].reshape(-1, 3), axis=0)[:20].tolist()
        raise ValueError(f"UAVid RGB mask contains unsupported colors: {unknown}")
    return squared_distance.argmin(axis=-1).astype(np.int64)


def _load_uavid_mask(path: Path, mask_id_base: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        return _decode_uavid_rgb(array)
    if array.ndim != 2:
        raise ValueError(f"Unsupported UAVid mask shape {array.shape}: {path}")

    raw = np.asarray(array, dtype=np.int64)
    mapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
    if mask_id_base == "zero":
        valid = (raw >= 0) & (raw < len(UAVID_CLASSES))
        mapped[valid] = raw[valid]
        unexpected = ~valid & (raw != IGNORE_INDEX)
    elif mask_id_base == "one":
        valid = (raw >= 1) & (raw <= len(UAVID_CLASSES))
        mapped[valid] = raw[valid] - 1
        # Dataset ID zero is the ignored label under one-based encoding.
        unexpected = ~valid & (raw != 0) & (raw != IGNORE_INDEX)
    else:
        raise ValueError(f"Unresolved UAVid mask_id_base={mask_id_base!r}")
    if bool(np.any(unexpected)):
        raise ValueError(
            f"Unexpected UAVid mask IDs {np.unique(raw[unexpected]).tolist()}: {path}"
        )
    return mapped


def load_target(sample: EvalSample, dataset: str, *, height: int, width: int) -> np.ndarray:
    if dataset == "flair":
        target = load_flair_mask_array(sample.mask_path, ignore_index=IGNORE_INDEX)
    elif dataset == "loveda":
        target = _load_loveda_mask(sample.mask_path)
    elif dataset == "uavid":
        target = _load_uavid_mask(sample.mask_path, sample.mask_id_base)
    elif dataset == "xbd_pre" and sample.mask_path.suffix.casefold() == ".json":
        from baselines.rskt_seg.xbd_label_utils import load_xbd_building_mask

        target = load_xbd_building_mask(sample.mask_path, height=height, width=width)
    else:
        with Image.open(sample.mask_path) as image:
            target = (np.asarray(image.convert("L"), dtype=np.uint8) != 0).astype(np.int64)
    target = np.asarray(target, dtype=np.int64)
    if target.shape != (height, width):
        raise ValueError(
            f"Image/GT shape mismatch for {sample.name}: image={(height, width)}, "
            f"mask={target.shape} ({sample.mask_path})"
        )
    return np.ascontiguousarray(target)


def resize_keep_ratio(rgb: np.ndarray, max_size: int) -> np.ndarray:
    """Match MMSeg ``Resize(scale=(S,S), keep_ratio=True)`` for RGB input."""

    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    height, width = rgb.shape[:2]
    scale = min(float(max_size) / float(width), float(max_size) / float(height))
    # MMCV's rescale helper rounds positive dimensions with ``int(x + .5)``.
    resized_width = max(1, int(width * scale + 0.5))
    resized_height = max(1, int(height * scale + 0.5))
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    return np.asarray(
        image.resize((resized_width, resized_height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None,
    prediction_ignore_index: int | None = None,
) -> np.ndarray:
    """Build a confusion matrix, optionally retaining prediction abstentions.

    When ``prediction_ignore_index`` is set, the returned matrix has one
    additional prediction column.  Pixels rejected by the model are recorded
    in that column, so they remain false negatives for their ground-truth
    classes instead of disappearing from the metric denominator.
    """

    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    valid_target = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid_target &= target != int(ignore_index)

    if prediction_ignore_index is None:
        valid = valid_target & (prediction >= 0) & (prediction < num_classes)
        if not bool(np.any(valid)):
            return np.zeros((num_classes, num_classes), dtype=np.int64)
        indices = target[valid] * num_classes + prediction[valid]
        return np.bincount(
            indices, minlength=num_classes**2
        ).reshape(num_classes, num_classes)

    valid_prediction = (prediction >= 0) & (prediction < num_classes)
    rejected_prediction = prediction == int(prediction_ignore_index)
    unexpected = valid_target & ~valid_prediction & ~rejected_prediction
    if bool(np.any(unexpected)):
        values = np.unique(prediction[unexpected])[:8].tolist()
        raise ValueError(
            "Prediction contains invalid class IDs on valid GT pixels: "
            f"{values}; expected 0..{num_classes - 1} or "
            f"{int(prediction_ignore_index)}"
        )
    if not bool(np.any(valid_target)):
        return np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    prediction_column = np.where(
        rejected_prediction[valid_target],
        num_classes,
        prediction[valid_target],
    )
    indices = target[valid_target] * (num_classes + 1) + prediction_column
    return np.bincount(
        indices, minlength=num_classes * (num_classes + 1)
    ).reshape(num_classes, num_classes + 1)


def metrics_from_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.float64)
    num_classes = len(class_names)
    supported_shapes = (
        (num_classes, num_classes),
        (num_classes, num_classes + 1),
    )
    if matrix.shape not in supported_shapes:
        raise ValueError(
            f"Confusion matrix must be one of {supported_shapes}, got {matrix.shape}"
        )
    class_matrix = matrix[:, :num_classes]
    rejected_pixels = (
        float(matrix[:, num_classes].sum())
        if matrix.shape[1] == num_classes + 1
        else 0.0
    )
    true_positive = np.diag(class_matrix)
    gt_count = matrix.sum(axis=1)
    pred_count = class_matrix.sum(axis=0)
    iou_denom = gt_count + pred_count - true_positive
    f1_denom = gt_count + pred_count
    iou = np.divide(true_positive, iou_denom, out=np.full(num_classes, np.nan), where=iou_denom > 0)
    f1 = np.divide(2 * true_positive, f1_denom, out=np.full(num_classes, np.nan), where=f1_denom > 0)
    accuracy = np.divide(true_positive, gt_count, out=np.full(num_classes, np.nan), where=gt_count > 0)
    valid_pixels = float(matrix.sum())
    result: dict[str, Any] = {
        "miou": float(np.nanmean(iou)) if bool(np.any(iou_denom > 0)) else 0.0,
        "mf1": float(np.nanmean(f1)) if bool(np.any(f1_denom > 0)) else 0.0,
        "macc": float(np.nanmean(accuracy)) if bool(np.any(gt_count > 0)) else 0.0,
        "pixel_accuracy": float(true_positive.sum() / max(valid_pixels, 1.0)),
        "valid_pixels": int(valid_pixels),
        "rejected_pixels": int(rejected_pixels),
        "rejection_rate": float(rejected_pixels / max(valid_pixels, 1.0)),
    }
    for class_id, class_name in enumerate(class_names):
        key = re.sub(r"[^a-z0-9]+", "_", class_name.casefold()).strip("_")
        result[f"iou_{key}"] = None if np.isnan(iou[class_id]) else float(iou[class_id])
        result[f"f1_{key}"] = None if np.isnan(f1[class_id]) else float(f1[class_id])
        result[f"acc_{key}"] = None if np.isnan(accuracy[class_id]) else float(accuracy[class_id])
    return result


def metrics_for_dataset(confusion: np.ndarray, dataset: str) -> dict[str, Any]:
    """Add the established binary-benchmark aliases to generic metrics."""

    class_names = tuple(str(name) for name in DATASET_SPECS[dataset]["classes"])
    result = metrics_from_confusion(confusion, class_names)
    if dataset == "uavid":
        foreground_iou = [
            result[f"iou_{re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')}"]
            for name in class_names[1:]
        ]
        foreground_f1 = [
            result[f"f1_{re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')}"]
            for name in class_names[1:]
        ]
        valid_iou = [value for value in foreground_iou if value is not None]
        valid_f1 = [value for value in foreground_f1 if value is not None]
        result["miou_foreground7"] = float(np.mean(valid_iou)) if valid_iou else 0.0
        result["mf1_foreground7"] = float(np.mean(valid_f1)) if valid_f1 else 0.0
    if len(class_names) == 2:
        for class_name in class_names:
            key = re.sub(r"[^a-z0-9]+", "_", class_name.casefold()).strip("_")
            result[f"{key}_iou"] = result[f"iou_{key}"]
            result[f"{key}_f1"] = result[f"f1_{key}"]
    primary_metric = str(DATASET_SPECS[dataset]["primary_metric"])
    if primary_metric not in result:
        raise RuntimeError(
            f"Primary metric {primary_metric!r} is absent for dataset {dataset!r}"
        )
    return result


def colorize_mask(
    mask: np.ndarray,
    palette: np.ndarray,
    *,
    ignore_index: int | None,
) -> np.ndarray:
    ids = np.asarray(mask, dtype=np.int64)
    valid = (ids >= 0) & (ids < len(palette))
    output = np.zeros((*ids.shape, 3), dtype=np.uint8)
    output[valid] = np.asarray(palette, dtype=np.uint8)[ids[valid]]
    if ignore_index is not None:
        output[ids == int(ignore_index)] = np.asarray(
            FLAIR_IGNORE_VISUALIZATION_RGB,
            dtype=np.uint8,
        )
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "BINARY_PALETTE",
    "DATASET_SPECS",
    "EvalSample",
    "IGNORE_INDEX",
    "LOVEDA_CLASSES",
    "LOVEDA_PALETTE",
    "UAVID_CLASSES",
    "UAVID_PALETTE",
    "colorize_mask",
    "confusion_matrix",
    "discover_dataset",
    "discover_uavid",
    "load_rgb",
    "load_target",
    "metrics_from_confusion",
    "metrics_for_dataset",
    "normalize_path",
    "read_class_groups",
    "resize_keep_ratio",
    "validate_class_groups",
    "write_json",
]
