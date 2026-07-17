from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Must stay identical to Vistar's eval_flux2_second.py.  In particular, old
# RCDGen wrapper revisions used a different class order and thus could not
# consume the shared SECOND one-class protocol safely.
SECOND_CLASSES = (
    "unchanged",
    "water",
    "bare land",
    "low vegetation",
    "tree",
    "buildings",
    "playgrounds",
)
SECOND_PALETTE = np.asarray(
    [
        [255, 255, 255],
        [0, 0, 255],
        [128, 128, 128],
        [0, 128, 0],
        [0, 255, 0],
        [128, 0, 0],
        [255, 255, 0],
    ],
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
}


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
    replacements = (
        ("_t1", "_t2"), ("_T1", "_T2"), ("_1", "_2"),
        ("-t1", "-t2"), ("-T1", "-T2"), ("-1", "-2"),
    )
    for source_suffix, target_suffix in replacements:
        if source_suffix in source.stem:
            for key in (source.stem.replace(source_suffix, target_suffix),):
                found = index.get(key) or index.get(key.lower())
                if found is not None:
                    return found
    raise FileNotFoundError(f"cannot match {source.name} under {folder}")


def selection_key(name: str, direction: str) -> str:
    return f"{name}\t{direction}"


def load_selection_records(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("protocol") != "second_oneclass_targetmask_v1":
            raise ValueError(f"Unsupported class-selection protocol in {path}: {row.get('protocol')!r}")
        name, direction = str(row.get("name", "")), str(row.get("direction", ""))
        key = selection_key(name, direction)
        if not name or direction not in {"t1_to_t2", "t2_to_t1"} or key in records:
            raise ValueError(f"Invalid or duplicate class-selection record in {path}: {row}")
        class_id = int(row.get("class_id", -1))
        if not 0 <= class_id < len(SECOND_CLASSES):
            raise ValueError(f"Invalid selected SECOND class in {path}: {row}")
        if str(row.get("class_name")) != SECOND_CLASSES[class_id]:
            raise ValueError(f"Class name/id mismatch in {path}: {row}")
        records[key] = row
    if not records:
        raise ValueError(f"Class selection file is empty: {path}")
    return records


def nearest_rgb_ids(rgb: np.ndarray) -> np.ndarray:
    rgb_i32 = rgb[..., :3].astype(np.int32)
    ids = np.full(rgb_i32.shape[:2], -1, dtype=np.int64)
    matched = np.zeros(rgb_i32.shape[:2], dtype=bool)
    for color, class_id in SECOND_RGB_LABEL_COLORS.items():
        hit = np.all(rgb_i32 == np.asarray(color, dtype=np.int32), axis=-1)
        ids[hit] = class_id
        matched |= hit
    if not np.all(matched):
        palette = SECOND_PALETTE.astype(np.int32)
        distance = ((rgb_i32[..., None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
        ids[~matched] = distance.argmin(axis=-1)[~matched]
    return ids.clip(0, len(SECOND_CLASSES) - 1)


def gray_second_ids(array: np.ndarray, zero_is_class: bool) -> np.ndarray:
    array = array.astype(np.int64, copy=False)
    unique = np.unique(array)
    output = np.zeros(array.shape, dtype=np.int64)
    valid = array != 255
    if unique.size <= 2 and set(int(value) for value in unique.tolist()).issubset({0, 255}):
        return output
    valid_values = array[valid]
    if valid_values.size == 0:
        return output
    if zero_is_class and int(valid_values.max()) <= 5 and int(valid_values.min()) >= 0 and 6 not in unique:
        output[valid] = array[valid] + 1
    elif int(valid_values.max()) < len(SECOND_CLASSES):
        output[valid] = array[valid]
    else:
        output[valid] = (array[valid] != 0).astype(np.int64)
    return output.clip(0, len(SECOND_CLASSES) - 1)


def load_ids(path: Path, zero_is_class: bool) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"RGB", "RGBA", "P"}:
        return nearest_rgb_ids(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return gray_second_ids(np.asarray(image.convert("L")), zero_is_class)


def target_change_ids(target: np.ndarray, other: np.ndarray, label_pair_mode: str) -> np.ndarray:
    if label_pair_mode == "direct_t1":
        return target.clip(0, len(SECOND_CLASSES) - 1)
    if label_pair_mode == "compare":
        changed = (target != other) & (target > 0) & (other > 0)
        return np.where(changed, target, 0).astype(np.int64)
    if label_pair_mode != "auto":
        raise ValueError(f"Unsupported label_pair_mode={label_pair_mode!r} in selection record")
    if float((target > 0).mean()) > 0.75:
        changed = (target != other) & (target > 0) & (other > 0)
        return np.where(changed, target, 0).astype(np.int64)
    return target.clip(0, len(SECOND_CLASSES) - 1)


def resized_ids(ids: np.ndarray, size: int) -> np.ndarray:
    """Match Vistar's _resize_mask_hw exactly, including nearest indices.

    PIL and ``torch.nn.functional.interpolate(mode='nearest')`` choose
    different source pixels for downsampling.  The class-selection JSONL was
    created by Vistar with the latter, so using PIL here can erase a thin
    class and falsely reject an otherwise valid shared record.
    """
    height, width = ids.shape
    source_y = np.floor(np.arange(size, dtype=np.float64) * height / size).astype(np.int64)
    source_x = np.floor(np.arange(size, dtype=np.float64) * width / size).astype(np.int64)
    return ids[source_y[:, None], source_x[None, :]].astype(np.uint8, copy=False)


def write_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), "L").save(path)


def selected_semantic_mask(
    *,
    label1_path: Path,
    label2_path: Path,
    direction: str,
    record: dict[str, Any],
) -> np.ndarray:
    zero_is_class = bool(record.get("semantic_zero_is_class", False))
    label1 = load_ids(label1_path, zero_is_class)
    label2 = load_ids(label2_path, zero_is_class)
    if label1.shape != label2.shape:
        raise ValueError(f"label shape mismatch: {label1_path}={label1.shape}, {label2_path}={label2.shape}")
    target, other = (label2, label1) if direction == "t1_to_t2" else (label1, label2)
    directional_ids = target_change_ids(target, other, str(record.get("label_pair_mode", "auto")))
    record_size = int(record.get("resize_size", 0))
    if record_size <= 0:
        raise ValueError(f"Missing/invalid resize_size in class-selection record: {record}")
    available = sorted(int(value) for value in np.unique(resized_ids(directional_ids, record_size)) if int(value) > 0)
    expected_available = [int(value) for value in record.get("available_target_class_ids", [])]
    if available != expected_available:
        raise ValueError(
            "Class selection record does not match the current SECOND directional labels for "
            f"name={record['name']!r}, direction={direction!r}: record={expected_available}, current={available}."
        )
    class_id = int(record["class_id"])
    if class_id > 0 and class_id not in available:
        raise ValueError(f"Selected class {class_id} is absent for class-selection record: {record}")
    return np.where(directional_ids == class_id, class_id, 0).astype(np.uint8) if class_id > 0 else np.zeros_like(directional_ids, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shared SECOND one-class manifest for a Vistar-model baseline.")
    parser.add_argument("--second_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class_selection_file", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--direction", choices=["t1_to_t2", "t2_to_t1", "both"], default="both")
    parser.add_argument(
        "--consumer",
        choices=["rcdgen", "flux1_fill", "anysd"],
        default="rcdgen",
        help="Declare the model input contract without changing sampled classes or masks.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.second_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    selection_file = Path(args.class_selection_file).expanduser().resolve()
    if not selection_file.is_file():
        raise FileNotFoundError(f"Shared CLASS_SELECTION_FILE does not exist: {selection_file}")
    selections = load_selection_records(selection_file)

    t1_dir = first_dir(root, args.split, ("A", "img_A", "im1", "img1", "T1", "t1", "images/T1", "images/A"))
    t2_dir = first_dir(root, args.split, ("B", "img_B", "im2", "img2", "T2", "t2", "images/T2", "images/B"))
    label1_dir = first_dir(root, args.split, ("label1", "Label1", "labels1", "Labels1", "semantic1", "Semantic1", "labels/label1", "labels/Label1"))
    label2_dir = first_dir(root, args.split, ("label2", "Label2", "labels2", "Labels2", "semantic2", "Semantic2", "labels/label2", "labels/Label2"))
    missing = [name for name, value in (("A/T1/im1", t1_dir), ("B/T2/im2", t2_dir), ("label1", label1_dir), ("label2", label2_dir)) if value is None]
    if missing:
        raise NotADirectoryError(
            "The shared one-class SECOND protocol requires directional semantic labels; missing "
            f"{', '.join(missing)} under {root}."
        )
    assert t1_dir is not None and t2_dir is not None and label1_dir is not None and label2_dir is not None
    t1_index, t2_index = index_files(t1_dir, IMAGE_EXTS), index_files(t2_dir, IMAGE_EXTS)
    label1_index, label2_index = index_files(label1_dir, MASK_EXTS), index_files(label2_dir, MASK_EXTS)
    unique_t1 = sorted(set(t1_index.values()), key=lambda path: path.name)
    if args.max_samples > 0:
        unique_t1 = unique_t1[: args.max_samples]
    directions = ["t1_to_t2", "t2_to_t1"] if args.direction == "both" else [args.direction]
    mask_dir = output.parent / f"{output.stem}_selected_masks"
    rows: list[dict[str, Any]] = []
    for t1_path in unique_t1:
        t2_path = matching(t2_index, t1_path, t2_dir)
        label1_path, label2_path = matching(label1_index, t1_path, label1_dir), matching(label2_index, t1_path, label2_dir)
        for direction in directions:
            key = selection_key(t1_path.stem, direction)
            if key not in selections:
                raise ValueError(
                    f"CLASS_SELECTION_FILE={selection_file} has no record for name={t1_path.stem!r}, direction={direction!r}."
                )
            record = selections[key]
            expected_target_label = "label2" if direction == "t1_to_t2" else "label1"
            if str(record.get("target_label")) != expected_target_label:
                raise ValueError(
                    f"CLASS_SELECTION_FILE has an inconsistent target_label for name={t1_path.stem!r}, "
                    f"direction={direction!r}: expected {expected_target_label!r}, got {record.get('target_label')!r}."
                )
            semantic_mask = selected_semantic_mask(
                label1_path=label1_path, label2_path=label2_path, direction=direction, record=record
            )
            base = f"{t1_path.stem}_{direction}"
            semantic_path = mask_dir / f"{base}_selected_semantic_mask.png"
            binary_path = mask_dir / f"{base}_selected_binary_mask.png"
            write_mask(semantic_mask, semantic_path)
            write_mask(np.where(semantic_mask > 0, 255, 0), binary_path)
            source, target = (t1_path, t2_path) if direction == "t1_to_t2" else (t2_path, t1_path)
            target_label = label2_path if direction == "t1_to_t2" else label1_path
            rows.append({
                "name": base,
                "sample_name": t1_path.stem,
                "dataset": "SECOND",
                "split": args.split,
                "direction": direction,
                "source_image": str(source),
                "target_image": str(target),
                "target_change_label": str(target_label),
                "label1": str(label1_path),
                "label2": str(label2_path),
                "selected_semantic_change_mask": str(semantic_path),
                "selected_binary_change_mask": str(binary_path),
                "selected_class_id": int(record["class_id"]),
                "selected_class_name": str(record["class_name"]),
                "class_selection_file": str(selection_file),
                "class_selection_record": record,
                "consumer": args.consumer,
                "model_inputs": (
                    ["source_image", "selected_binary_change_mask", "text_prompt"]
                    if args.consumer == "flux1_fill"
                    else ["source_image", "selected_semantic_change_mask", "text_prompt"]
                    if args.consumer == "anysd"
                    else ["source_image", "text_prompt"]
                ),
                "ground_truth_change_mask_passed_to_model": args.consumer in {"flux1_fill", "anysd"},
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_rcdgen_second_manifest] t1_dir={t1_dir}")
    print(f"[build_rcdgen_second_manifest] t2_dir={t2_dir}")
    print(f"[build_rcdgen_second_manifest] label1_dir={label1_dir}")
    print(f"[build_rcdgen_second_manifest] label2_dir={label2_dir}")
    print(f"[build_rcdgen_second_manifest] class_selection_file={selection_file}")
    print(f"[build_rcdgen_second_manifest] wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
