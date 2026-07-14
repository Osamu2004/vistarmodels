from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


SECOND_CLASSES = {
    0: "unchanged", 1: "water", 2: "bare land", 3: "low vegetation",
    4: "tree", 5: "buildings", 6: "playgrounds",
}
SECOND_PALETTE = np.asarray([
    [255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
    [0, 255, 0], [128, 0, 0], [255, 255, 0],
], dtype=np.uint8)
OUTPUT_DIRS = ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "prompts")


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC) if image.size != (size, size) else image


def load_label(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    values = np.asarray(image, dtype=np.uint8)
    if np.any(values > 6):
        raise ValueError(f"SECOND semantic label must contain Vistar class IDs 0..6: {path}")
    return values


def load_change(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize common bidirectional SECOND eval inputs from the semantic-manifest layout.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit base scenes; both directions are retained.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root, output = resolve(args.data_root), resolve(args.output_dir)
    required = {
        "t1": root / "images" / args.split / "t1",
        "t2": root / "images" / args.split / "t2",
        "l1": root / "labels" / args.split / "t1",
        "l2": root / "labels" / args.split / "t2",
        "change": root / "masks" / args.split,
    }
    for path in required.values():
        if not path.is_dir():
            raise NotADirectoryError(path)
    folders = {name: output / name for name in OUTPUT_DIRS}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    scenes = sorted(path.stem for path in required["t1"].glob("*.png"))
    if args.max_samples > 0:
        scenes = scenes[:args.max_samples]
    manifest = []
    for scene in tqdm(scenes, desc="materialize SECOND"):
        paths = {key: folder / f"{scene}.png" for key, folder in required.items()}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing {key} for SECOND scene {scene}: {path}")
        image1, image2 = load_rgb(paths["t1"], args.size), load_rgb(paths["t2"], args.size)
        label1, label2 = load_label(paths["l1"], args.size), load_label(paths["l2"], args.size)
        changed = load_change(paths["change"], args.size)
        for direction, source, target, target_label in (
            ("t1_to_t2", image1, image2, label2),
            ("t2_to_t1", image2, image1, label1),
        ):
            name = f"{scene}_{direction}"
            ids = np.where(changed, target_label, 0).astype(np.uint8)
            changed_ids = sorted(int(value) for value in np.unique(ids) if int(value) > 0)
            prompt = (
                "Keep the remote-sensing image unchanged."
                if not changed_ids
                else "Create realistic directional changes involving "
                + ", ".join(SECOND_CLASSES[value] for value in changed_ids)
                + ", while preserving unchanged geography and image style."
            )
            targets = {
                "source_rgb": folders["source_rgb"] / f"{name}_source_rgb.png",
                "gt_rgb": folders["gt_rgb"] / f"{name}_gt_rgb.png",
                "cond_mask_ids": folders["cond_mask_ids"] / f"{name}_cond_mask_ids.png",
                "cond_mask": folders["cond_mask"] / f"{name}_cond_mask.png",
                "cond_mask_official": folders["cond_mask_official"] / f"{name}_cond_mask_official.png",
            }
            if args.overwrite or not all(path.is_file() for path in targets.values()):
                save(source, targets["source_rgb"])
                save(target, targets["gt_rgb"])
                save(Image.fromarray(ids, "L"), targets["cond_mask_ids"])
                binary = np.repeat(np.where(ids[..., None] > 0, 255, 0).astype(np.uint8), 3, axis=2)
                save(Image.fromarray(binary, "RGB"), targets["cond_mask"])
                save(Image.fromarray(SECOND_PALETTE[ids], "RGB"), targets["cond_mask_official"])
            (folders["prompts"] / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")
            manifest.append({
                "name": name, "scene": scene, "direction": direction,
                "source_image": str(targets["source_rgb"]), "target_image": str(targets["gt_rgb"]),
                "condition_mask_ids": str(targets["cond_mask_ids"]),
                "condition_mask": str(targets["cond_mask"]),
                "condition_mask_official": str(targets["cond_mask_official"]),
                "changed_class_ids": changed_ids,
                "changed_class_names": [SECOND_CLASSES[value] for value in changed_ids],
                "prompt": prompt,
            })
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps({
        "dataset": "SECOND", "split": args.split, "size": args.size,
        "directions": ["t1_to_t2", "t2_to_t1"],
        "semantic_rule": "condition ID = target Vistar semantic ID inside official binary change mask; unchanged = 0",
        "num_scenes": len(scenes), "num_eval_items": len(manifest),
    }, indent=2), encoding="utf-8")
    print(f"[materialize_second_eval] scenes={len(scenes)} items={len(manifest)} output={output}")


if __name__ == "__main__":
    main()
