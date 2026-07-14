from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SECOND_PALETTE = np.asarray([
    [255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
    [0, 255, 0], [128, 0, 0], [255, 255, 0],
], dtype=np.uint8)


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def place(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Vistar SECOND manifest to ChangeBridge A/B/label/list layout.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite_lists", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in resolve(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    root = resolve(args.output_dir)
    for folder in ("A", "B", "label", "list"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    filenames = []
    for row in rows:
        filename = f"{row['name']}.png"
        source, target = resolve(row["source_image"]), resolve(row["target_image"])
        place(source, root / "A" / filename, args.mode)
        place(target, root / "B" / filename, args.mode)
        official = row.get("condition_mask_official")
        if official and Path(official).is_file():
            place(resolve(official), root / "label" / filename, args.mode)
        else:
            ids = np.asarray(Image.open(resolve(row["condition_mask_ids"])).convert("L"), dtype=np.uint8)
            Image.fromarray(SECOND_PALETTE[ids.clip(0, 6)], "RGB").save(root / "label" / filename)
        filenames.append(filename)

    list_path = root / "list" / f"{args.split}.txt"
    if list_path.exists() and not args.overwrite_lists:
        existing = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        filenames = list(dict.fromkeys([*existing, *filenames]))
    list_path.write_text("\n".join(filenames) + ("\n" if filenames else ""), encoding="utf-8")
    for split in ("train", "val", "test"):
        (root / "list" / f"{split}.txt").touch(exist_ok=True)
    (root / f"manifest_{args.split}.jsonl").write_text(resolve(args.manifest).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[prepare_changebridge_dataset] {args.split}: {len(filenames)} samples -> {root}")


if __name__ == "__main__":
    main()
