from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


LOVEDA_COLORS: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "background",
    (255, 255, 255): "building",
    (255, 0, 0): "road",
    (0, 0, 255): "water",
    (255, 255, 0): "barren land",
    (0, 255, 0): "forest",
    (0, 255, 255): "agriculture",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LOVEDA_PALETTE_U8 = np.asarray(list(LOVEDA_COLORS.keys()), dtype=np.uint8)
LOVEDA_CLASS_NAMES = list(LOVEDA_COLORS.values())


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [part for part in text.strip("\\").split("\\") if part]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _resolve(path: str) -> Path:
    return Path(_normalize_wsl_unc(path)).expanduser().resolve()


def _sample_name(path: Path) -> str:
    suffix = "_cond_mask.png"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _discover_images(root: Path) -> list[Path]:
    excluded_tokens = ("mask", "label", "annotation", "ground_truth")

    def is_reference_rgb(path: Path) -> bool:
        relative_parts = [part.lower() for part in path.relative_to(root).parts]
        return not any(
            token in part
            for part in relative_parts
            for token in excluded_tokens
        )

    images = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and is_reference_rgb(path)
    )
    if not images:
        raise FileNotFoundError(f"No reference images found under: {root}")
    return images


def _visible_classes(mask_path: Path, strict_palette: bool) -> list[str]:
    with Image.open(mask_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    colors = sorted({tuple(int(value) for value in color) for color in np.unique(rgb.reshape(-1, 3), axis=0)})
    unknown = [color for color in colors if color not in LOVEDA_COLORS]
    if unknown and strict_palette:
        raise ValueError(
            f"Mask {mask_path} contains colors outside the official LoveDA palette: {unknown}. "
            "Use the official-palette Vistar eval directory or pass --no-strict_palette."
        )
    classes = [LOVEDA_COLORS[color] for color in colors if color in LOVEDA_COLORS and color != (0, 0, 0)]
    return classes


def _validate_size(path: Path, expected_size: int) -> None:
    with Image.open(path) as image:
        if image.size != (expected_size, expected_size):
            raise ValueError(
                f"Unified LoveDA protocol requires saved {expected_size}x{expected_size} inputs, "
                f"but {path} has size {image.size}. Rebuild the Vistar source directory at 512; "
                "do not silently upsample a 256 evaluation set."
            )


def _prompt_for_mask(mask_path: Path, prompt_prefix: str, strict_palette: bool) -> str:
    classes = _visible_classes(mask_path, strict_palette)
    if not classes:
        return prompt_prefix.rstrip(" .") + "."
    return f"{prompt_prefix.rstrip(' .')} containing {', '.join(classes)}."


def _parse_loveda_splits(value: str) -> list[str]:
    text = str(value).strip().lower()
    if text in {"all", "both", "train+val", "val+train", "train,val", "val,train"}:
        return ["Train", "Val"]
    result: list[str] = []
    for item in text.replace(",", "+").split("+"):
        item = item.strip()
        if item in {"train", "training"}:
            result.append("Train")
        elif item in {"val", "eval", "validation"}:
            result.append("Val")
        elif item:
            raise ValueError(f"Unsupported LoveDA split: {item}")
    if not result:
        raise ValueError(f"No LoveDA split selected from: {value!r}")
    return list(dict.fromkeys(result))


def _parse_loveda_domains(value: str) -> list[str]:
    text = str(value).strip().lower()
    if text in {"all", "both", "urban+rural", "rural+urban", "urban,rural", "rural,urban"}:
        return ["Urban", "Rural"]
    result: list[str] = []
    for item in text.replace(",", "+").split("+"):
        item = item.strip()
        if item == "urban":
            result.append("Urban")
        elif item == "rural":
            result.append("Rural")
        elif item:
            raise ValueError(f"Unsupported LoveDA domain: {item}")
    if not result:
        raise ValueError(f"No LoveDA domain selected from: {value!r}")
    return list(dict.fromkeys(result))


def _resolve_loveda_dirs(root: Path, splits: str, domains: str) -> list[tuple[str, str, Path, Path]]:
    direct_images = root / "images_png"
    direct_masks = root / "masks_png"
    if direct_images.is_dir() and direct_masks.is_dir():
        return [(root.name, root.name, direct_images, direct_masks)]

    result: list[tuple[str, str, Path, Path]] = []
    for split in _parse_loveda_splits(splits):
        for domain in _parse_loveda_domains(domains):
            image_dir = root / split / domain / "images_png"
            mask_dir = root / split / domain / "masks_png"
            if image_dir.is_dir() and mask_dir.is_dir():
                result.append((split, domain, image_dir, mask_dir))
    if not result:
        raise NotADirectoryError(
            "Cannot find the official LoveDA folders. Expected either "
            f"{direct_images} + {direct_masks}, or "
            f"{root}/{{Train,Val}}/{{Urban,Rural}}/{{images_png,masks_png}}."
        )
    return result


def _list_flat_images(root: Path) -> list[Path]:
    images = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
    )
    if not images:
        raise FileNotFoundError(f"No LoveDA RGB images found under: {root}")
    return images


def _find_mask(mask_dir: Path, image_path: Path) -> Path:
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find LoveDA mask for {image_path.name} under: {mask_dir}")


def _raw_loveda_mask_ids(mask_path: Path, strict_palette: bool, reduce_zero_label: bool) -> np.ndarray:
    with Image.open(mask_path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        raw = array.astype(np.int64, copy=False)
        if reduce_zero_label:
            allowed = (raw == 0) | (raw == 255) | ((raw >= 1) & (raw <= len(LOVEDA_CLASS_NAMES)))
            ids = np.zeros(raw.shape, dtype=np.int64)
            valid = (raw >= 1) & (raw <= len(LOVEDA_CLASS_NAMES))
            ids[valid] = raw[valid] - 1
        else:
            allowed = (raw == 255) | ((raw >= 0) & (raw < len(LOVEDA_CLASS_NAMES)))
            ids = np.where((raw >= 0) & (raw < len(LOVEDA_CLASS_NAMES)), raw, 0)
        if strict_palette and not bool(np.all(allowed)):
            unknown = np.unique(raw[~allowed]).tolist()[:20]
            raise ValueError(f"LoveDA mask {mask_path} contains unsupported IDs: {unknown}")
        return ids
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.int32, copy=False)
        palette = LOVEDA_PALETTE_U8.astype(np.int32)
        squared = np.sum((rgb[..., None, :] - palette[None, None, :, :]) ** 2, axis=-1)
        ids = squared.argmin(axis=-1).astype(np.int64)
        if strict_palette:
            matched = np.min(squared, axis=-1) == 0
            if not bool(np.all(matched)):
                unknown = np.unique(rgb[~matched].reshape(-1, 3), axis=0)[:20].tolist()
                raise ValueError(f"LoveDA RGB mask {mask_path} contains unsupported colors: {unknown}")
        return ids
    raise ValueError(f"Unsupported LoveDA mask shape {array.shape}: {mask_path}")


def _prompt_for_raw_loveda_mask(
    mask_path: Path,
    prompt_prefix: str,
    strict_palette: bool,
    reduce_zero_label: bool,
) -> str:
    ids = _raw_loveda_mask_ids(mask_path, strict_palette, reduce_zero_label)
    classes = [
        LOVEDA_CLASS_NAMES[class_id]
        for class_id in sorted(int(value) for value in np.unique(ids))
        if class_id != 0
    ]
    if not classes:
        return prompt_prefix.rstrip(" .") + "."
    return f"{prompt_prefix.rstrip(' .')} containing {', '.join(classes)}."


def _record_name(split: str, domain: str, image_path: Path) -> str:
    if str(split).lower() in {"val", "eval", "validation"}:
        return f"{domain}_{image_path.stem}"
    return f"{split}_{domain}_{image_path.stem}"


def _stable_reference(
    *,
    name: str,
    target_path: Path,
    references: list[Path],
    seed: int,
    target_digest: str,
    digest_cache: dict[Path, str],
) -> Path:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(references)
    target_resolved = target_path.resolve()
    for offset in range(len(references)):
        candidate = references[(start + offset) % len(references)]
        if candidate == target_resolved:
            continue
        candidate_digest = digest_cache.get(candidate)
        if candidate_digest is None:
            candidate_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            digest_cache[candidate] = candidate_digest
        if candidate_digest != target_digest:
            return candidate
    raise ValueError(
        "Reference pool contains no image other than the paired target. "
        "Provide an independent LoveDA/external reference directory."
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-auditable TISynth LoveDA inference manifest."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--loveda_root", help="official LoveDA dataset root")
    source.add_argument("--eval_dir", help="legacy Vistar eval dir with cond_mask and gt_rgb")
    parser.add_argument("--splits", default="train,val", help="LoveDA splits for direct-dataset mode")
    parser.add_argument("--domains", default="both", help="LoveDA domains for direct-dataset mode")
    parser.add_argument(
        "--reference_dir",
        default="",
        help="reference-image pool; defaults to loveda_root/Train in direct-dataset mode",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0, help="deterministic reference-selection seed")
    parser.add_argument(
        "--prompt_prefix",
        default="a high-resolution remote sensing satellite image",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--expected_samples",
        type=int,
        default=0,
        help="validate the discovered sample count before applying --max_samples (0 disables)",
    )
    parser.add_argument("--expected_size", type=int, default=512)
    parser.add_argument(
        "--reduce_zero_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="map official raw LoveDA IDs 1..7 to model IDs 0..6",
    )
    parser.add_argument(
        "--strict_palette",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require exact official LoveDA RGB colors",
    )
    parser.add_argument(
        "--allow_eval_gt_reference_pool",
        action="store_true",
        help="allow references to be drawn from eval_dir/gt_rgb (self-target is still excluded)",
    )
    args = parser.parse_args()

    output = _resolve(args.output)
    eval_dir = _resolve(args.eval_dir) if args.eval_dir else None
    loveda_root = _resolve(args.loveda_root) if args.loveda_root else None
    if loveda_root is not None:
        default_reference = loveda_root / "Train" if (loveda_root / "Train").is_dir() else loveda_root
        reference_dir = _resolve(args.reference_dir) if args.reference_dir else default_reference.resolve()
    else:
        if not args.reference_dir:
            raise ValueError("--reference_dir is required with legacy --eval_dir mode")
        reference_dir = _resolve(args.reference_dir)
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Missing TISynth reference folder: {reference_dir}")

    references = _discover_images(reference_dir)
    reference_digest_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    if loveda_root is not None:
        for split, domain, image_dir, mask_dir in _resolve_loveda_dirs(
            loveda_root, str(args.splits), str(args.domains)
        ):
            for target_path in _list_flat_images(image_dir):
                source_rows.append(
                    {
                        "name": _record_name(split, domain, target_path),
                        "condition_image": _find_mask(mask_dir, target_path),
                        "target_image": target_path,
                        "condition_format": "loveda_raw",
                        "split": split,
                        "domain": domain,
                    }
                )
    else:
        assert eval_dir is not None
        cond_dir = eval_dir / "cond_mask"
        gt_dir = eval_dir / "gt_rgb"
        if not cond_dir.is_dir() or not gt_dir.is_dir():
            raise NotADirectoryError(f"Legacy eval source must contain cond_mask and gt_rgb: {eval_dir}")
        if _is_relative_to(reference_dir, gt_dir) or _is_relative_to(gt_dir, reference_dir):
            if not args.allow_eval_gt_reference_pool:
                raise ValueError(
                    f"Reference pool overlaps evaluation GT: {reference_dir}. Pass an independent pool or "
                    "explicitly use --allow_eval_gt_reference_pool."
                )
        for cond_path in sorted(cond_dir.glob("*_cond_mask.png")):
            name = _sample_name(cond_path)
            target_path = gt_dir / f"{name}_gt_rgb.png"
            if not target_path.is_file():
                raise FileNotFoundError(f"Missing paired GT for {name}: {target_path}")
            _validate_size(cond_path, int(args.expected_size))
            _validate_size(target_path, int(args.expected_size))
            source_rows.append(
                {
                    "name": name,
                    "condition_image": cond_path,
                    "target_image": target_path,
                    "condition_format": "rgb_palette",
                }
            )

    discovered_samples = len(source_rows)
    if int(args.expected_samples) > 0 and discovered_samples != int(args.expected_samples):
        raise ValueError(
            f"LoveDA source count mismatch: discovered={discovered_samples}, "
            f"expected={int(args.expected_samples)}. Check --splits/--domains or disable the count check."
        )
    if int(args.max_samples) > 0:
        source_rows = source_rows[: int(args.max_samples)]
    for source_row in tqdm(source_rows, desc="Build TISynth LoveDA manifest"):
        name = str(source_row["name"])
        cond_path = Path(source_row["condition_image"])
        target_path = Path(source_row["target_image"])
        reference_path = _stable_reference(
            name=name,
            target_path=target_path,
            references=references,
            seed=int(args.seed),
            target_digest=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            digest_cache=reference_digest_cache,
        )
        if source_row["condition_format"] == "loveda_raw":
            prompt = _prompt_for_raw_loveda_mask(
                cond_path,
                str(args.prompt_prefix),
                bool(args.strict_palette),
                bool(args.reduce_zero_label),
            )
        else:
            prompt = _prompt_for_mask(cond_path, str(args.prompt_prefix), bool(args.strict_palette))
        rows.append(
            {
                **source_row,
                "name": name,
                "condition_image": str(cond_path.resolve()),
                "target_image": str(target_path.resolve()),
                "reference_image": str(reference_path),
                "reference_file_sha256": reference_digest_cache[reference_path],
                "prompt": prompt,
                "reference_seed": int(args.seed),
                "reference_protocol": "sha256 deterministic sample from configured pool; identical target excluded",
                "reduce_zero_label": bool(args.reduce_zero_label),
            }
        )

    if not rows:
        raise FileNotFoundError("No paired LoveDA image/mask samples were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_mode": "official_loveda" if loveda_root is not None else "vistar_eval",
        "loveda_root": str(loveda_root) if loveda_root is not None else None,
        "eval_dir": str(eval_dir) if eval_dir is not None else None,
        "splits": str(args.splits),
        "domains": str(args.domains),
        "reference_dir": str(reference_dir),
        "reference_images": len(references),
        "discovered_samples": discovered_samples,
        "samples": len(rows),
        "reference_seed": int(args.seed),
        "strict_palette": bool(args.strict_palette),
        "reduce_zero_label": bool(args.reduce_zero_label),
        "expected_size": int(args.expected_size),
        "allow_eval_gt_reference_pool": bool(args.allow_eval_gt_reference_pool),
    }
    output.with_suffix(".protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[build_tisynth_loveda_manifest] wrote: {output}")


if __name__ == "__main__":
    main()
