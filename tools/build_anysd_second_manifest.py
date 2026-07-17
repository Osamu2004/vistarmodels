from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from build_rcdgen_second_manifest import (
    IMAGE_EXTS,
    MASK_EXTS,
    SECOND_CLASSES,
    SECOND_PALETTE,
    first_dir,
    index_files,
    load_ids,
    matching,
    target_change_ids,
    write_mask,
)


PROMPT_CLASS_NAMES = (
    "unchanged",
    "changed inland water",
    "changed bare land",
    "changed grass",
    "changed forest",
    "changed building",
    "changed playground",
)
PROTOCOL = "second_full_multiclass_targetmask_v1"


def _color_map() -> list[dict[str, Any]]:
    return [
        {
            "id": class_id,
            "label_name": SECOND_CLASSES[class_id],
            "prompt_name": PROMPT_CLASS_NAMES[class_id],
            "rgb": SECOND_PALETTE[class_id].tolist(),
        }
        for class_id in range(len(SECOND_CLASSES))
    ]


def _full_directional_mask(
    *,
    label1: np.ndarray,
    label2: np.ndarray,
    direction: str,
    label_pair_mode: str,
) -> np.ndarray:
    if label1.shape != label2.shape:
        raise ValueError(f"label shape mismatch: {label1.shape} != {label2.shape}")
    target, other = (
        (label2, label1) if direction == "t1_to_t2" else (label1, label2)
    )
    return target_change_ids(target, other, label_pair_mode).astype(
        np.uint8, copy=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build full multi-class target-side SECOND change masks for AnySD. "
            "Unlike the shared one-class protocol, this builder keeps every "
            "changed category in each directional mask."
        )
    )
    parser.add_argument("--second_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--direction",
        choices=["t1_to_t2", "t2_to_t1", "both"],
        default="both",
    )
    parser.add_argument(
        "--label_pair_mode",
        choices=["auto", "direct_t1", "compare"],
        default="auto",
    )
    parser.add_argument(
        "--semantic_zero_is_class",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reuse_if_valid",
        action="store_true",
        help=(
            "Reuse an existing manifest only when its sample names, directions, "
            "protocol, and label settings match this invocation."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.second_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    t1_dir = first_dir(
        root,
        args.split,
        ("A", "img_A", "im1", "img1", "T1", "t1", "images/T1", "images/A"),
    )
    t2_dir = first_dir(
        root,
        args.split,
        ("B", "img_B", "im2", "img2", "T2", "t2", "images/T2", "images/B"),
    )
    label1_dir = first_dir(
        root,
        args.split,
        (
            "label1",
            "Label1",
            "labels1",
            "Labels1",
            "semantic1",
            "Semantic1",
            "labels/label1",
            "labels/Label1",
        ),
    )
    label2_dir = first_dir(
        root,
        args.split,
        (
            "label2",
            "Label2",
            "labels2",
            "Labels2",
            "semantic2",
            "Semantic2",
            "labels/label2",
            "labels/Label2",
        ),
    )
    missing = [
        name
        for name, value in (
            ("A/T1/im1", t1_dir),
            ("B/T2/im2", t2_dir),
            ("label1", label1_dir),
            ("label2", label2_dir),
        )
        if value is None
    ]
    if missing:
        raise NotADirectoryError(
            "Full multi-class AnySD evaluation requires SECOND's directional "
            f"semantic labels; missing {', '.join(missing)} under {root}."
        )
    assert (
        t1_dir is not None
        and t2_dir is not None
        and label1_dir is not None
        and label2_dir is not None
    )

    t1_index = index_files(t1_dir, IMAGE_EXTS)
    t2_index = index_files(t2_dir, IMAGE_EXTS)
    label1_index = index_files(label1_dir, MASK_EXTS)
    label2_index = index_files(label2_dir, MASK_EXTS)
    unique_t1 = sorted(set(t1_index.values()), key=lambda path: path.name)
    if args.max_samples > 0:
        unique_t1 = unique_t1[: args.max_samples]
    directions = (
        ["t1_to_t2", "t2_to_t1"]
        if args.direction == "both"
        else [args.direction]
    )
    mask_dir = output.parent / f"{output.stem}_full_multiclass_masks"
    color_map = _color_map()
    rows: list[dict[str, Any]] = []

    if args.reuse_if_valid and output.is_file() and output.stat().st_size > 0:
        existing = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_names = {
            f"{t1_path.stem}_{direction}"
            for t1_path in unique_t1
            for direction in directions
        }
        actual_names = {str(row.get("name", "")) for row in existing}
        settings_match = all(
            row.get("protocol") == PROTOCOL
            and row.get("split") == args.split
            and row.get("direction") in directions
            and row.get("label_pair_mode") == args.label_pair_mode
            and bool(row.get("semantic_zero_is_class", False))
            == bool(args.semantic_zero_is_class)
            and Path(str(row.get("semantic_change_mask", ""))).is_file()
            for row in existing
        )
        if (
            len(existing) == len(expected_names)
            and actual_names == expected_names
            and settings_match
        ):
            print(
                "[build_anysd_second_manifest] reusing validated manifest "
                f"with {len(existing)} rows: {output}"
            )
            return
        print(
            "[build_anysd_second_manifest] existing manifest does not match "
            "the requested data/settings; rebuilding"
        )

    for t1_path in tqdm(
        unique_t1,
        desc="Build AnySD full-multiclass masks",
        unit="pair",
        dynamic_ncols=True,
    ):
        t2_path = matching(t2_index, t1_path, t2_dir)
        label1_path = matching(label1_index, t1_path, label1_dir)
        label2_path = matching(label2_index, t1_path, label2_dir)
        label1 = load_ids(label1_path, bool(args.semantic_zero_is_class))
        label2 = load_ids(label2_path, bool(args.semantic_zero_is_class))
        if label1.shape != label2.shape:
            raise ValueError(
                f"label shape mismatch: {label1_path}={label1.shape}, "
                f"{label2_path}={label2.shape}"
            )
        for direction in directions:
            semantic_mask = _full_directional_mask(
                label1=label1,
                label2=label2,
                direction=direction,
                label_pair_mode=args.label_pair_mode,
            )
            present_class_ids = sorted(
                int(value)
                for value in np.unique(semantic_mask)
                if int(value) > 0
            )
            base = f"{t1_path.stem}_{direction}"
            semantic_path = mask_dir / f"{base}_full_multiclass_mask.png"
            write_mask(semantic_mask, semantic_path)
            source, target = (
                (t1_path, t2_path)
                if direction == "t1_to_t2"
                else (t2_path, t1_path)
            )
            target_label = (
                label2_path if direction == "t1_to_t2" else label1_path
            )
            rows.append(
                {
                    "protocol": PROTOCOL,
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
                    "semantic_change_mask": str(semantic_path),
                    "present_class_ids": present_class_ids,
                    "present_label_names": [
                        SECOND_CLASSES[value] for value in present_class_ids
                    ],
                    "present_prompt_names": [
                        PROMPT_CLASS_NAMES[value] for value in present_class_ids
                    ],
                    "label_pair_mode": args.label_pair_mode,
                    "semantic_zero_is_class": bool(
                        args.semantic_zero_is_class
                    ),
                    "color_map": color_map,
                    "consumer": "anysd",
                    "model_inputs": [
                        "source_image",
                        "full_multiclass_semantic_change_mask",
                        "text_prompt",
                    ],
                    "ground_truth_change_mask_passed_to_model": True,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_anysd_second_manifest] t1_dir={t1_dir}")
    print(f"[build_anysd_second_manifest] t2_dir={t2_dir}")
    print(f"[build_anysd_second_manifest] label1_dir={label1_dir}")
    print(f"[build_anysd_second_manifest] label2_dir={label2_dir}")
    print(
        "[build_anysd_second_manifest] "
        f"protocol={PROTOCOL} label_pair_mode={args.label_pair_mode} "
        f"semantic_zero_is_class={int(args.semantic_zero_is_class)}"
    )
    print(
        f"[build_anysd_second_manifest] wrote {len(rows)} rows to {output}"
    )


if __name__ == "__main__":
    main()
