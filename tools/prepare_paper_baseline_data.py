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


def prepare_second(source: Path, output: Path, size: int) -> dict:
    summary: dict[str, int] = {}
    for split in ("train", "test"):
        csv_path = source / f"{split}.csv"
        rows: list[dict] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            scenes = list(csv.DictReader(handle))
        for scene in scenes:
            scene_id = Path(scene["image_t1"]).stem
            for direction, source_key, target_key, label_key in (
                ("t1_to_t2", "image_t1", "image_t2", "label_t2"),
                ("t2_to_t1", "image_t2", "image_t1", "label_t1"),
            ):
                name = f"{scene_id}_{direction}"
                item_root = output / "second" / split
                source_out = item_root / "source_rgb" / f"{name}.png"
                target_out = item_root / "target_rgb" / f"{name}.png"
                ids_out = item_root / "target_mask_ids" / f"{name}.png"
                rgb_out = item_root / "target_mask_rgb" / f"{name}.png"
                source_path = source / scene[source_key]
                target_path = source / scene[target_key]
                label_path = source / scene[label_key]
                save_rgb(Image.open(source_path), source_out, size)
                save_rgb(Image.open(target_path), target_out, size)
                ids = save_ids(label_path, ids_out, size)
                unknown = sorted(int(x) for x in np.unique(ids) if int(x) > 6)
                if unknown:
                    raise ValueError(f"{label_path} has invalid SECOND IDs: {unknown}")
                colorize(ids, SECOND_PALETTE, rgb_out)
                class_ids = sorted(int(x) for x in np.unique(ids) if int(x) > 0)
                rows.append({
                    "name": name,
                    "dataset": "SECOND",
                    "split": split,
                    "direction": direction,
                    "source_image": str(source_out),
                    "target_image": str(target_out),
                    "target_mask_ids": str(ids_out),
                    "target_mask_rgb": str(rgb_out),
                    "changed_class_ids": class_ids,
                    "prompt": "A realistic 256 by 256 remote sensing image matching the target semantic change mask.",
                })
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
        summary.update(prepare_second(resolve(args.second_root), output, args.second_size))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
