from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


LOVEDA_CLASSES = ("background", "building", "road", "water", "barren", "forest", "agriculture")
LOVEDA_PALETTE = np.asarray([
    [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255],
    [255, 255, 0], [0, 255, 0], [0, 255, 255],
], dtype=np.uint8)


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def name_from(path: Path) -> str:
    suffix = "_gt_rgb.png"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def split_from(name: str) -> str:
    lowered = name.lower()
    for split in ("train", "val", "test"):
        if lowered.startswith(f"{split}_"):
            return split
    return "val"


def parse_splits(value: str) -> set[str]:
    text = value.strip().lower()
    if text in {"", "all", "both", "*"}:
        return {"train", "val", "test"}
    result = {item.strip() for item in text.replace("+", ",").split(",") if item.strip()}
    aliases = {"training": "train", "validation": "val", "valid": "val", "eval": "val"}
    result = {aliases.get(item, item) for item in result}
    unknown = result - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unsupported split(s): {sorted(unknown)}")
    return result


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def palette_to_ids(path: Path, strict: bool) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    distance = ((rgb[..., None, :] - LOVEDA_PALETTE.astype(np.int32)[None, None, :, :]) ** 2).sum(axis=-1)
    if strict and np.any(distance.min(axis=-1) != 0):
        unknown = np.unique(rgb[distance.min(axis=-1) != 0].reshape(-1, 3), axis=0)[:20].tolist()
        raise ValueError(f"{path} contains colors outside the LoveDA palette: {unknown}")
    return distance.argmin(axis=-1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Vistar LoveDA eval data for TODSynth/CRFM.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--splits", default="all", help="comma-separated train,val,test or all")
    parser.add_argument("--prompt_prefix", default="A high-resolution remote-sensing image")
    parser.add_argument("--no_strict_palette", action="store_true")
    args = parser.parse_args()

    source, output = resolve(args.eval_dir), resolve(args.output_dir)
    gt_dir = source / "gt_rgb"
    official_dir = source / "cond_mask_official"
    if not official_dir.is_dir():
        official_dir = source / "cond_mask"
    for folder in (gt_dir, official_dir):
        if not folder.is_dir():
            raise NotADirectoryError(folder)
    (output / "demo_img").mkdir(parents=True, exist_ok=True)
    (output / "demo_label").mkdir(parents=True, exist_ok=True)
    rows, mapping = [], []
    selected_splits = parse_splits(args.splits)
    for target in sorted(gt_dir.glob("*_gt_rgb.png")):
        name = name_from(target)
        split = split_from(name)
        if split not in selected_splits:
            continue
        mask = official_dir / f"{name}_{'cond_mask_official' if official_dir.name == 'cond_mask_official' else 'cond_mask'}.png"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        ids = palette_to_ids(mask, not args.no_strict_palette)
        image_rel, label_rel = f"demo_img/{name}.png", f"demo_label/{name}.png"
        link_or_copy(target.resolve(), output / image_rel, args.mode)
        Image.fromarray(ids, "L").save(output / label_rel)
        present = [LOVEDA_CLASSES[value] for value in sorted(int(value) for value in np.unique(ids)) if value > 0]
        prompt = args.prompt_prefix.rstrip(" .") + (f" containing {', '.join(present)}." if present else ".")
        rows.append({"source": label_rel, "target": image_rel, "prompt": prompt})
        mapping.append({"name": name, "split": split, "condition_image": str(mask.resolve()), "target_image": str(target.resolve()), "prompt": prompt})
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
    if not rows:
        raise FileNotFoundError(
            f"no LoveDA samples for splits={sorted(selected_splits)} under {source}; "
            "check Vistar sample-name prefixes and --splits"
        )
    for filename, records in (("index.jsonl", rows), ("vistar_manifest.jsonl", mapping)):
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "datameta.json").write_text(json.dumps({"num_cls": len(LOVEDA_CLASSES), "classes": LOVEDA_CLASSES}, indent=2), encoding="utf-8")
    print(f"[prepare_todsynth_loveda] splits={sorted(selected_splits)} wrote {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
