from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


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


def index_files(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.name.startswith(".") and path.suffix.lower() in exts:
            output.setdefault(path.stem, path)
            output.setdefault(path.stem.lower(), path)
    return output


def matching(index: dict[str, Path], source: Path, folder: Path) -> Path:
    for key in (source.stem, source.stem.lower()):
        if key in index:
            return index[key]
    raise FileNotFoundError(f"cannot match {source.name} under {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an RCDGen manifest from SECOND A/B/change-label folders.")
    parser.add_argument("--second_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", choices=["t1_to_t2", "t2_to_t1", "both"], default="t1_to_t2")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.second_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    t1_dir = first_dir(root, args.split, ("A", "img_A", "im1", "T1", "t1", "images/T1", "images/A"))
    t2_dir = first_dir(root, args.split, ("B", "img_B", "im2", "T2", "t2", "images/T2", "images/B"))
    gt_dir = first_dir(root, args.split, ("gt", "GT", "label", "Label", "change", "change_label"))
    label1_dir = first_dir(root, args.split, ("label1", "Label1", "labels1", "labels/label1"))
    label2_dir = first_dir(root, args.split, ("label2", "Label2", "labels2", "labels/label2"))

    missing = []
    if t1_dir is None:
        missing.append("A/T1/img_A")
    if t2_dir is None:
        missing.append("B/T2/img_B")
    if gt_dir is None and label2_dir is None:
        missing.append("gt/change-label or label2")
    if missing:
        raise NotADirectoryError(f"Cannot find required RCDGen SECOND folders: {', '.join(missing)} under {root}")
    if args.direction in {"t2_to_t1", "both"} and label1_dir is None:
        raise NotADirectoryError(
            "t2_to_t1 requires a directional label1 folder. A standard SECOND A/B/gt layout only supports "
            "RCDGen's official t1_to_t2 protocol; use --direction t1_to_t2."
        )

    assert t1_dir is not None and t2_dir is not None
    t1_index = index_files(t1_dir, IMAGE_EXTS)
    t2_index = index_files(t2_dir, IMAGE_EXTS)
    gt_index = index_files(gt_dir, MASK_EXTS) if gt_dir else {}
    label1_index = index_files(label1_dir, MASK_EXTS) if label1_dir else {}
    label2_index = index_files(label2_dir, MASK_EXTS) if label2_dir else {}
    directions = ["t1_to_t2", "t2_to_t1"] if args.direction == "both" else [args.direction]

    # Keep dataset-facing names even when a test/deployment tree uses symlinks;
    # resolving here would replace the shared SECOND stem with the source stem.
    unique_t1 = sorted(set(t1_index.values()), key=lambda path: path.name)
    if args.max_samples > 0:
        unique_t1 = unique_t1[: args.max_samples]
    rows = []
    for t1_path in unique_t1:
        t2_path = matching(t2_index, t1_path, t2_dir)
        for direction in directions:
            if direction == "t1_to_t2":
                source, target = t1_path, t2_path
                if label2_dir is not None:
                    change_label = matching(label2_index, t1_path, label2_dir)
                    label_source = "label2"
                else:
                    assert gt_dir is not None
                    change_label = matching(gt_index, t1_path, gt_dir)
                    label_source = "gt"
            else:
                source, target = t2_path, t1_path
                assert label1_dir is not None
                change_label = matching(label1_index, t1_path, label1_dir)
                label_source = "label1"
            rows.append({
                "name": f"{t1_path.stem}_{direction}",
                "dataset": "SECOND",
                "split": args.split,
                "direction": direction,
                "source_image": str(source),
                "target_image": str(target),
                "target_change_label": str(change_label),
                "change_label_source": label_source,
                "model_inputs": ["source_image", "text_prompt"],
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_rcdgen_second_manifest] t1_dir={t1_dir}")
    print(f"[build_rcdgen_second_manifest] t2_dir={t2_dir}")
    print(f"[build_rcdgen_second_manifest] gt_dir={gt_dir}")
    print(f"[build_rcdgen_second_manifest] label1_dir={label1_dir}")
    print(f"[build_rcdgen_second_manifest] label2_dir={label2_dir}")
    print(f"[build_rcdgen_second_manifest] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
