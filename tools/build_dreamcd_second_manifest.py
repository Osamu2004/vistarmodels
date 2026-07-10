from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DIRECTIONS = ("t1_to_t2", "t2_to_t1")


@dataclass(frozen=True)
class SecondDirs:
    t1_dir: Path
    t2_dir: Path
    label1_dir: Path
    label2_dir: Path
    change_dir: Path | None


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _candidate_bases(root: Path, split: str) -> list[Path]:
    split_aliases = {
        "train": ("train", "Train", "training", "Training"),
        "test": ("test", "Test", "testing", "Testing"),
        "val": ("val", "Val", "valid", "Valid", "validation", "Validation"),
    }
    bases = [root]
    if split and split != "auto":
        bases.extend(root / name for name in split_aliases.get(split, (split,)))
    else:
        for key in ("train", "test", "val"):
            bases.extend(root / name for name in split_aliases[key])
    return bases


def _candidate_dirs(root: Path, names: tuple[str, ...], split: str) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for base in _candidate_bases(root, split):
        for name in names:
            path = base / name
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _first_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def _resolve_second_dirs(root: Path, split: str) -> SecondDirs:
    t1_dir = _first_dir(
        _candidate_dirs(
            root,
            (
                "img_A",
                "im1",
                "img1",
                "image1",
                "Image1",
                "images1",
                "T1",
                "t1",
                "A",
                "images/T1",
                "images/t1",
                "images/A",
            ),
            split,
        )
    )
    t2_dir = _first_dir(
        _candidate_dirs(
            root,
            (
                "img_B",
                "im2",
                "img2",
                "image2",
                "Image2",
                "images2",
                "T2",
                "t2",
                "B",
                "images/T2",
                "images/t2",
                "images/B",
            ),
            split,
        )
    )
    label1_dir = _first_dir(
        _candidate_dirs(
            root,
            (
                "mask_A",
                "label1",
                "Label1",
                "labels1",
                "Labels1",
                "semantic1",
                "Semantic1",
                "labels/label1",
                "labels/Label1",
            ),
            split,
        )
    )
    label2_dir = _first_dir(
        _candidate_dirs(
            root,
            (
                "mask_B",
                "label2",
                "Label2",
                "labels2",
                "Labels2",
                "semantic2",
                "Semantic2",
                "labels/label2",
                "labels/Label2",
            ),
            split,
        )
    )
    change_dir = _first_dir(
        _candidate_dirs(
            root,
            (
                "bcd_mask",
                "change_mask",
                "binary_change_mask",
                "label",
                "Label",
                "change",
                "Change",
                "change_label",
                "Change_Label",
                "mask",
                "masks",
                "gt",
                "GT",
            ),
            split,
        )
    )

    missing = []
    if t1_dir is None:
        missing.append("T1/img_A")
    if t2_dir is None:
        missing.append("T2/img_B")
    if label1_dir is None:
        missing.append("label1/mask_A")
    if label2_dir is None:
        missing.append("label2/mask_B")
    if missing:
        raise NotADirectoryError(
            "Cannot find required SECOND/DreamCD folders: "
            + ", ".join(missing)
            + ". DreamCD needs paired images and paired semantic masks."
        )

    return SecondDirs(
        t1_dir=t1_dir,
        t2_dir=t2_dir,
        label1_dir=label1_dir,
        label2_dir=label2_dir,
        change_dir=change_dir,
    )


def _index_files(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in exts:
            continue
        out.setdefault(path.stem, path)
        out.setdefault(path.stem.lower(), path)
    return out


def _lookup_path(index: dict[str, Path], stem: str, folder: Path) -> Path:
    for key in (stem, stem.lower()):
        found = index.get(key)
        if found is not None:
            return found
    replacements = (
        ("_t1", "_t2"),
        ("_T1", "_T2"),
        ("_1", "_2"),
        ("-t1", "-t2"),
        ("-T1", "-T2"),
        ("-1", "-2"),
        ("_A", "_B"),
        ("_a", "_b"),
    )
    for src, dst in replacements:
        if src in stem:
            key = stem.replace(src, dst)
            found = index.get(key) or index.get(key.lower())
            if found is not None:
                return found
    raise FileNotFoundError(f"cannot find matching file for stem={stem!r} under {folder}")


def _build_pairs(dirs: SecondDirs) -> list[dict[str, Path]]:
    t1_index = _index_files(dirs.t1_dir, IMAGE_EXTS)
    t2_index = _index_files(dirs.t2_dir, IMAGE_EXTS)
    label1_index = _index_files(dirs.label1_dir, MASK_EXTS)
    label2_index = _index_files(dirs.label2_dir, MASK_EXTS)
    change_index = _index_files(dirs.change_dir, MASK_EXTS) if dirs.change_dir is not None else {}

    pairs: list[dict[str, Path]] = []
    seen: set[str] = set()
    for stem, t1_path in sorted(t1_index.items()):
        if stem.lower() in seen:
            continue
        seen.add(stem.lower())
        item = {
            "stem": stem,
            "t1": t1_path,
            "t2": _lookup_path(t2_index, t1_path.stem, dirs.t2_dir),
            "label1": _lookup_path(label1_index, t1_path.stem, dirs.label1_dir),
            "label2": _lookup_path(label2_index, t1_path.stem, dirs.label2_dir),
        }
        if dirs.change_dir is not None:
            item["change"] = _lookup_path(change_index, t1_path.stem, dirs.change_dir)
        pairs.append(item)
    if not pairs:
        raise FileNotFoundError(f"no image pairs found under {dirs.t1_dir} and {dirs.t2_dir}")
    return pairs


def _selected_directions(value: str) -> list[str]:
    if value == "both":
        return list(DIRECTIONS)
    if value not in DIRECTIONS:
        raise ValueError(f"unsupported direction={value!r}")
    return [value]


def _resolve_external_style(path_text: str, *, label: str, second_root: Path) -> Path:
    if not path_text:
        raise ValueError(f"{label} is required for time-reference AdaIN")
    path = Path(_normalize_wsl_unc(path_text)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        path.relative_to(second_root)
    except ValueError:
        return path
    raise ValueError(
        f"{label} must be an external known-time reference outside SECOND_ROOT; "
        f"refusing possible target-data style leakage: {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DreamCD JSONL manifest from a SECOND-style paired image/mask directory."
    )
    parser.add_argument("--second_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", help="auto, train, test, val, or a custom split directory name")
    parser.add_argument("--direction", choices=["t1_to_t2", "t2_to_t1", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--t1_style_image", default="", help="external known-T1 style reference for t2_to_t1 AdaIN")
    parser.add_argument("--t2_style_image", default="", help="external known-T2 style reference for t1_to_t2 AdaIN")
    args = parser.parse_args()

    root = Path(_normalize_wsl_unc(args.second_root)).expanduser().resolve()
    out_path = Path(_normalize_wsl_unc(args.output)).expanduser().resolve()
    dirs = _resolve_second_dirs(root, str(args.split))
    selected_directions = _selected_directions(str(args.direction))
    t1_style_image = (
        _resolve_external_style(str(args.t1_style_image), label="T1 style image", second_root=root)
        if "t2_to_t1" in selected_directions
        else None
    )
    t2_style_image = (
        _resolve_external_style(str(args.t2_style_image), label="T2 style image", second_root=root)
        if "t1_to_t2" in selected_directions
        else None
    )
    if t1_style_image is not None and t2_style_image is not None and t1_style_image == t2_style_image:
        raise ValueError("T1_STYLE_IMAGE and T2_STYLE_IMAGE must be distinct known-time references")

    pairs = _build_pairs(dirs)
    if int(args.max_samples) > 0:
        pairs = pairs[: int(args.max_samples)]

    rows = []
    for pair in pairs:
        for direction in selected_directions:
            if direction == "t1_to_t2":
                source_image = pair["t1"]
                target_image = pair["t2"]
                source_mask = pair["label1"]
                target_mask = pair["label2"]
                style_image = t2_style_image
                style_time = "t2"
                prompt = (
                    "Synthesize the post-change remote-sensing image from the pre-change image, "
                    "post-change semantic mask, and binary change mask."
                )
            else:
                source_image = pair["t2"]
                target_image = pair["t1"]
                source_mask = pair["label2"]
                target_mask = pair["label1"]
                style_image = t1_style_image
                style_time = "t1"
                prompt = (
                    "Synthesize the pre-change remote-sensing image from the post-change image, "
                    "pre-change semantic mask, and binary change mask."
                )

            row = {
                "name": f"{pair['t1'].stem}_{direction}",
                "dataset": "SECOND",
                "split": str(args.split),
                "label_mode": "dreamcd_semantic_pair",
                "direction": direction,
                "source_image": str(source_image),
                "target_image": str(target_image),
                "source_mask": str(source_mask),
                "target_mask": str(target_mask),
                "style_image": str(style_image),
                "style_time": style_time,
                "adain_style_source": "external_known_time_reference",
                "prompt": prompt,
            }
            if "change" in pair:
                row["change_mask"] = str(pair["change"])
                row["change_mask_source"] = "explicit_bcd_mask"
            else:
                row["change_mask_source"] = "derived_from_source_target_masks"
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[build_dreamcd_second_manifest] second_root={root}")
    print(f"[build_dreamcd_second_manifest] t1_dir={dirs.t1_dir}")
    print(f"[build_dreamcd_second_manifest] t2_dir={dirs.t2_dir}")
    print(f"[build_dreamcd_second_manifest] label1_dir={dirs.label1_dir}")
    print(f"[build_dreamcd_second_manifest] label2_dir={dirs.label2_dir}")
    print(f"[build_dreamcd_second_manifest] change_dir={dirs.change_dir}")
    print(f"[build_dreamcd_second_manifest] t1_style_image={t1_style_image}")
    print(f"[build_dreamcd_second_manifest] t2_style_image={t2_style_image}")
    print(f"[build_dreamcd_second_manifest] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
