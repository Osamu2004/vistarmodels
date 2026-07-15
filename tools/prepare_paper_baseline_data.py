from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


LOVEDA_PALETTE = np.asarray(
    [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
     [159, 129, 183], [0, 255, 0], [255, 195, 128]],
    dtype=np.uint8,
)
SECOND_PALETTE = np.asarray(
    [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
     [0, 255, 0], [128, 0, 0], [255, 255, 0]],
    dtype=np.uint8,
)
SECOND_RGB_LABEL_COLORS = {
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
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def save_rgb(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    image.save(path)


def save_ids(source: Path, path: Path, size: int, remap_loveda: bool = False) -> np.ndarray:
    image = Image.open(source).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    ids = np.asarray(image, dtype=np.uint8).copy()
    if remap_loveda:
        # Official LoveDA masks use 0 as ignore and 1..7 as semantic IDs.
        ignore = ids == 0
        ids = np.clip(ids, 1, 7) - 1
        ids[ignore] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids, mode="L").save(path)
    return ids


def load_second_ids(source: Path, size: int) -> np.ndarray:
    image = Image.open(source)
    array = np.asarray(image)
    if array.ndim == 2:
        ids = array.astype(np.uint8)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        channels_equal = np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        if channels_equal and int(rgb[..., 0].max()) <= 6:
            ids = rgb[..., 0].copy()
        else:
            ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            for color, class_id in SECOND_RGB_LABEL_COLORS.items():
                ids[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
            if np.any(ids == 255):
                unknown = np.unique(rgb[ids == 255].reshape(-1, 3), axis=0)
                preview = [tuple(int(channel) for channel in color) for color in unknown[:10]]
                raise ValueError(f"{source} contains unknown SECOND RGB label colors: {preview}")
    unknown_ids = sorted(int(value) for value in np.unique(ids) if int(value) > 6)
    if unknown_ids:
        raise ValueError(f"{source} contains invalid SECOND class IDs: {unknown_ids}")
    if ids.shape != (size, size):
        ids = np.asarray(
            Image.fromarray(ids, mode="L").resize((size, size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return ids


def save_second_ids(source: Path, path: Path, size: int) -> np.ndarray:
    ids = load_second_ids(source, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids, mode="L").save(path)
    return ids


def colorize(ids: np.ndarray, palette: np.ndarray, path: Path) -> None:
    safe = ids.copy()
    ignore = safe == 255
    safe = np.clip(safe, 0, len(palette) - 1)
    rgb = palette[safe]
    rgb[ignore] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_loveda(source: Path, output: Path, size: int) -> dict:
    summary: dict[str, int] = {}
    for split_name, source_name in (("train", "Train"), ("val", "Val")):
        rows: list[dict] = []
        for domain in ("Urban", "Rural"):
            root = source / source_name / domain
            for image_path in sorted((root / "images_png").glob("*.png")):
                mask_path = root / "masks_png" / image_path.name
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                name = f"{domain.lower()}_{image_path.stem}"
                image_out = output / "loveda" / split_name / "images" / f"{name}.png"
                ids_out = output / "loveda" / split_name / "mask_ids" / f"{name}.png"
                rgb_out = output / "loveda" / split_name / "mask_rgb" / f"{name}.png"
                save_rgb(Image.open(image_path), image_out, size)
                ids = save_ids(mask_path, ids_out, size, remap_loveda=True)
                colorize(ids, LOVEDA_PALETTE, rgb_out)
                rows.append({
                    "name": name,
                    "dataset": "LoveDA",
                    "split": split_name,
                    "domain": domain.lower(),
                    "image": str(image_out),
                    "mask_ids": str(ids_out),
                    "mask_rgb": str(rgb_out),
                    "prompt": f"A high-resolution satellite image of a {domain.lower()} area.",
                })
        write_jsonl(output / "loveda" / f"{split_name}.jsonl", rows)
        summary[f"loveda_{split_name}"] = len(rows)
    return summary


def candidate_bases(root: Path, split: str) -> list[Path]:
    aliases = {
        "train": ("train", "Train", "training", "Training"),
        "test": ("test", "Test", "testing", "Testing"),
        "val": ("val", "Val", "valid", "Valid", "validation", "Validation"),
    }
    return [root, *(root / name for name in aliases.get(split, (split,)))]


def first_dir(root: Path, split: str, names: tuple[str, ...]) -> Path | None:
    for base in candidate_bases(root, split):
        for name in names:
            path = base / name
            if path.is_dir():
                return path
    return None


def index_files(folder: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in extensions:
            continue
        index.setdefault(path.stem, path)
        index.setdefault(path.stem.lower(), path)
    return index


def matching(index: dict[str, Path], source: Path, folder: Path) -> Path:
    for key in (source.stem, source.stem.lower()):
        if key in index:
            return index[key]
    replacements = (
        ("_t1", "_t2"), ("_T1", "_T2"), ("_1", "_2"),
        ("-t1", "-t2"), ("-T1", "-T2"), ("-1", "-2"),
    )
    for old, new in replacements:
        if old in source.stem:
            candidate = source.stem.replace(old, new)
            found = index.get(candidate) or index.get(candidate.lower())
            if found is not None:
                return found
    raise FileNotFoundError(f"cannot match {source.name} under {folder}")


def second_scenes_from_directories(source: Path, split: str) -> list[dict[str, str]]:
    t1_dir = first_dir(source, split, ("A", "img_A", "im1", "img1", "image1", "Image1", "T1", "t1", "images/T1", "images/A"))
    t2_dir = first_dir(source, split, ("B", "img_B", "im2", "img2", "image2", "Image2", "T2", "t2", "images/T2", "images/B"))
    label1_dir = first_dir(source, split, ("label1", "Label1", "labels1", "Labels1", "semantic1", "Semantic1", "labels/label1", "labels/Label1"))
    label2_dir = first_dir(source, split, ("label2", "Label2", "labels2", "Labels2", "semantic2", "Semantic2", "labels/label2", "labels/Label2"))
    missing = [
        name for name, value in (
            ("T1/im1/A", t1_dir), ("T2/im2/B", t2_dir),
            ("label1", label1_dir), ("label2", label2_dir),
        ) if value is None
    ]
    if missing:
        raise NotADirectoryError(f"missing SECOND {split} folders under {source}: {', '.join(missing)}")
    assert t1_dir is not None and t2_dir is not None and label1_dir is not None and label2_dir is not None
    t1_index = index_files(t1_dir, IMAGE_EXTS)
    t2_index = index_files(t2_dir, IMAGE_EXTS)
    label1_index = index_files(label1_dir, MASK_EXTS)
    label2_index = index_files(label2_dir, MASK_EXTS)
    scenes = []
    for t1_path in sorted(set(t1_index.values()), key=lambda path: path.name):
        scenes.append({
            "image_t1": str(t1_path),
            "image_t2": str(matching(t2_index, t1_path, t2_dir)),
            "label_t1": str(matching(label1_index, t1_path, label1_dir)),
            "label_t2": str(matching(label2_index, t1_path, label2_dir)),
        })
    if not scenes:
        raise FileNotFoundError(f"no SECOND {split} images found under {t1_dir}")
    print(f"[prepare_second] split={split} t1_dir={t1_dir}")
    print(f"[prepare_second] split={split} t2_dir={t2_dir}")
    print(f"[prepare_second] split={split} label1_dir={label1_dir}")
    print(f"[prepare_second] split={split} label2_dir={label2_dir}")
    return scenes


def resolve_scene_path(source: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source / path).resolve()


def prepare_second(
    source: Path,
    output: Path,
    size: int,
    splits: tuple[str, ...],
    storage: str,
) -> dict:
    summary: dict[str, int] = {}
    for split in splits:
        csv_path = source / f"{split}.csv"
        rows: list[dict] = []
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                scenes = list(csv.DictReader(handle))
            source_format = "csv"
        else:
            scenes = second_scenes_from_directories(source, split)
            source_format = "directories"
        for scene in scenes:
            scene_id = Path(scene["image_t1"]).stem
            for direction, source_key, target_key, label_key in (
                ("t1_to_t2", "image_t1", "image_t2", "label_t2"),
                ("t2_to_t1", "image_t2", "image_t1", "label_t1"),
            ):
                name = f"{scene_id}_{direction}"
                source_path = resolve_scene_path(source, scene[source_key])
                target_path = resolve_scene_path(source, scene[target_key])
                label_path = resolve_scene_path(source, scene[label_key])
                ids = load_second_ids(label_path, size)
                class_ids = sorted(int(x) for x in np.unique(ids) if int(x) > 0)
                row = {
                    "name": name,
                    "dataset": "SECOND",
                    "split": split,
                    "source_format": source_format,
                    "storage": storage,
                    "direction": direction,
                    "changed_class_ids": class_ids,
                    "prompt": f"A realistic {size} by {size} remote sensing image matching the target semantic change mask.",
                }
                if storage == "online":
                    row.update({
                        "source_image": str(source_path),
                        "target_image": str(target_path),
                        "target_mask_source": str(label_path),
                        "target_mask_encoding": "second_ids_or_palette_rgb",
                        "online_image_size": size,
                    })
                else:
                    item_root = output / "second" / split
                    source_out = item_root / "source_rgb" / f"{name}.png"
                    target_out = item_root / "target_rgb" / f"{name}.png"
                    ids_out = item_root / "target_mask_ids" / f"{name}.png"
                    rgb_out = item_root / "target_mask_rgb" / f"{name}.png"
                    save_rgb(Image.open(source_path), source_out, size)
                    save_rgb(Image.open(target_path), target_out, size)
                    ids_out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(ids, mode="L").save(ids_out)
                    colorize(ids, SECOND_PALETTE, rgb_out)
                    row.update({
                        "source_image": str(source_out),
                        "target_image": str(target_out),
                        "target_mask_ids": str(ids_out),
                        "target_mask_rgb": str(rgb_out),
                    })
                rows.append(row)
        write_jsonl(output / "second" / f"{split}.jsonl", rows)
        summary[f"second_{split}"] = len(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare common Table 3/4 baseline training and evaluation data.")
    parser.add_argument("--loveda_root", default=os.environ.get("LOVEDA_ROOT", "/data/vistar/datasets/atomic/loveda/source_data"))
    parser.add_argument("--second_root", default=os.environ.get("SECOND_ROOT", "/data/vistar/datasets/second_semantic_manifest"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/data/vistar/runs/paper_baselines/data"))
    parser.add_argument("--loveda_size", type=int, default=512)
    parser.add_argument("--second_size", type=int, default=256)
    parser.add_argument(
        "--second_splits",
        default=os.environ.get("SECOND_SPLITS", "train,test"),
        help="Comma-separated SECOND splits to prepare, e.g. train or train,test",
    )
    parser.add_argument(
        "--second_storage",
        choices=("materialized", "online"),
        default=os.environ.get("SECOND_STORAGE", "materialized"),
        help="materialized writes resized RGB/masks; online writes only raw paths in JSONL",
    )
    parser.add_argument("--dataset", choices=["all", "loveda", "second"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = resolve(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.dataset in {"all", "loveda"}:
        summary.update(prepare_loveda(resolve(args.loveda_root), output, args.loveda_size))
    if args.dataset in {"all", "second"}:
        second_splits = tuple(value.strip().lower() for value in args.second_splits.split(",") if value.strip())
        if not second_splits or any(value not in {"train", "test", "val"} for value in second_splits):
            raise ValueError(f"invalid --second_splits value: {args.second_splits!r}")
        summary.update(
            prepare_second(
                resolve(args.second_root),
                output,
                args.second_size,
                second_splits,
                args.second_storage,
            )
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
