from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
    parser.add_argument("--eval_dir", required=True, help="Vistar eval dir with cond_mask and gt_rgb")
    parser.add_argument("--reference_dir", required=True, help="independent, recursively scanned reference-image pool")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0, help="deterministic reference-selection seed")
    parser.add_argument(
        "--prompt_prefix",
        default="a high-resolution remote sensing satellite image",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--expected_size", type=int, default=512)
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

    eval_dir = _resolve(args.eval_dir)
    reference_dir = _resolve(args.reference_dir)
    output = _resolve(args.output)
    cond_dir = eval_dir / "cond_mask"
    gt_dir = eval_dir / "gt_rgb"
    if not cond_dir.is_dir():
        raise NotADirectoryError(f"Missing LoveDA condition folder: {cond_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"Missing LoveDA GT folder: {gt_dir}")
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Missing TISynth reference folder: {reference_dir}")
    if _is_relative_to(reference_dir, gt_dir) or _is_relative_to(gt_dir, reference_dir):
        if not args.allow_eval_gt_reference_pool:
            raise ValueError(
                f"Reference pool overlaps evaluation GT: {reference_dir}. This can bias paired/distribution "
                "metrics. Use an independent training/external reference pool, or explicitly pass "
                "--allow_eval_gt_reference_pool for an ablation."
            )

    references = _discover_images(reference_dir)
    reference_digest_cache: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    for cond_path in sorted(cond_dir.glob("*_cond_mask.png")):
        name = _sample_name(cond_path)
        target_path = gt_dir / f"{name}_gt_rgb.png"
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing paired GT for {name}: {target_path}")
        _validate_size(cond_path, int(args.expected_size))
        _validate_size(target_path, int(args.expected_size))
        reference_path = _stable_reference(
            name=name,
            target_path=target_path,
            references=references,
            seed=int(args.seed),
            target_digest=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            digest_cache=reference_digest_cache,
        )
        rows.append(
            {
                "name": name,
                "condition_image": str(cond_path.resolve()),
                "target_image": str(target_path.resolve()),
                "reference_image": str(reference_path),
                "reference_file_sha256": reference_digest_cache[reference_path],
                "prompt": _prompt_for_mask(cond_path, str(args.prompt_prefix), bool(args.strict_palette)),
                "reference_seed": int(args.seed),
                "reference_protocol": "sha256 deterministic sample from independent pool; identical target file excluded",
            }
        )
        if int(args.max_samples) > 0 and len(rows) >= int(args.max_samples):
            break

    if not rows:
        raise FileNotFoundError(f"No *_cond_mask.png files found under: {cond_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "eval_dir": str(eval_dir),
        "reference_dir": str(reference_dir),
        "reference_images": len(references),
        "samples": len(rows),
        "reference_seed": int(args.seed),
        "strict_palette": bool(args.strict_palette),
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
