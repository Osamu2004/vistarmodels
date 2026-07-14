from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SECOND_CLASSES = {
    0: "unchanged", 1: "water", 2: "bare land", 3: "low vegetation",
    4: "tree", 5: "buildings", 6: "playgrounds",
}


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def name_from(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected SECOND filename: {path.name}")
    return path.name[: -len(suffix)]


def direction_from(name: str) -> str:
    if name.endswith("_t1_to_t2"):
        return "t1_to_t2"
    if name.endswith("_t2_to_t1"):
        return "t2_to_t1"
    return "unknown"


def optional_path(root: Path, folder: str, name: str, suffix: str) -> str | None:
    path = root / folder / f"{name}{suffix}"
    return str(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full-mask SECOND manifest from a Vistar eval directory.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    root = resolve(args.eval_dir)
    source_dir, ids_dir, gt_dir = root / "source_rgb", root / "cond_mask_ids", root / "gt_rgb"
    for folder in (source_dir, ids_dir, gt_dir):
        if not folder.is_dir():
            raise NotADirectoryError(f"missing required SECOND folder: {folder}")

    rows = []
    for source in sorted(source_dir.glob("*_source_rgb.png")):
        name = name_from(source, "_source_rgb.png")
        ids_path = ids_dir / f"{name}_cond_mask_ids.png"
        target = gt_dir / f"{name}_gt_rgb.png"
        if not ids_path.is_file() or not target.is_file():
            raise FileNotFoundError(f"incomplete SECOND sample {name}: mask={ids_path}, target={target}")
        ids = np.asarray(Image.open(ids_path).convert("L"), dtype=np.uint8)
        unknown = sorted(int(value) for value in np.unique(ids) if int(value) not in SECOND_CLASSES)
        if unknown:
            raise ValueError(f"{ids_path} contains invalid SECOND IDs: {unknown}")
        changed_ids = sorted(int(value) for value in np.unique(ids) if int(value) > 0)
        rows.append({
            "name": name,
            "direction": direction_from(name),
            "source_image": str(source.resolve()),
            "target_image": str(target.resolve()),
            "condition_mask_ids": str(ids_path.resolve()),
            "condition_mask": optional_path(root, "cond_mask", name, "_cond_mask.png"),
            "condition_mask_official": optional_path(root, "cond_mask_official", name, "_cond_mask_official.png"),
            "changed_class_ids": changed_ids,
            "changed_class_names": [SECOND_CLASSES[value] for value in changed_ids],
        })
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_second_manifest_from_vistar_eval] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
