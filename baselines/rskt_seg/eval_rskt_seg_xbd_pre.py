from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from eval_rskt_seg_chn6 import (
    IMAGE_EXTS,
    MASK_EXTS,
    _barrier,
    _ceil_to_multiple,
    _configure_model,
    _distributed_context,
    _make_output_dirs,
    _path,
    _predict_tiled,
    _save_binary_mask,
    _save_color_mask,
    _save_overlay,
    _validate_resume_protocol,
    _write_json,
    _write_jsonl,
)
from xbd_label_utils import load_xbd_building_mask


Image.MAX_IMAGE_PIXELS = None


def _resolve_xbd_test_dirs(data_root: Path) -> tuple[Path, Path, str]:
    """Resolve either an xBD root or its official ``test`` directory."""
    split_roots: list[tuple[Path, str]] = [(data_root, data_root.name)]
    if data_root.name.lower() != "test":
        split_roots.append((data_root / "test", "test"))

    checked: list[tuple[Path, Path]] = []
    for split_root, split_name in split_roots:
        image_dir = split_root / "images"
        for label_name in ("labels", "targets", "targets_cvt", "masks_building"):
            label_dir = split_root / label_name
            checked.append((image_dir, label_dir))
            if image_dir.is_dir() and label_dir.is_dir():
                return image_dir, label_dir, split_name

    details = "\n".join(
        f"  images={image_dir} labels={label_dir}"
        for image_dir, label_dir in checked
    )
    raise NotADirectoryError(
        f"Cannot find xBD test image/label folders under {data_root}. Checked:\n"
        f"{details}"
    )


def _list_pre_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
        and path.stem.endswith("_pre_disaster")
    ]
    if not images:
        raise FileNotFoundError(
            f"No *_pre_disaster images found under {image_dir}"
        )
    return images


def _find_xbd_label(label_dir: Path, image_path: Path) -> Path:
    candidates = [label_dir / f"{image_path.stem}.json"]
    for extension in MASK_EXTS:
        candidates.extend(
            [
                label_dir / f"{image_path.stem}{extension}",
                label_dir / f"{image_path.stem}_target{extension}",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find an xBD label matching {image_path.name} under {label_dir}"
    )


def _load_building_mask(
    label_path: Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if label_path.suffix.lower() == ".json":
        mask = load_xbd_building_mask(
            label_path,
            height=int(height),
            width=int(width),
        )
    else:
        with Image.open(label_path) as image:
            mask = (
                np.asarray(image.convert("L"), dtype=np.uint8) != 0
            ).astype(np.uint8)
    if mask.shape != (height, width):
        raise ValueError(
            f"xBD image/label size mismatch for {label_path}: "
            f"expected={(height, width)}, got={mask.shape}"
        )
    return mask


def _confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_building = prediction.reshape(-1) == 1
    gt_building = target.reshape(-1) == 1
    return {
        "tp": int(np.count_nonzero(pred_building & gt_building)),
        "fp": int(np.count_nonzero(pred_building & ~gt_building)),
        "fn": int(np.count_nonzero(~pred_building & gt_building)),
        "tn": int(np.count_nonzero(~pred_building & ~gt_building)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1.0e-12
    building_iou = tp / max(tp + fp + fn, eps)
    background_iou = tn / max(tn + fp + fn, eps)
    return {
        "building_iou": float(building_iou),
        "background_iou": float(background_iou),
        "miou": float((building_iou + background_iou) * 0.5),
        "building_f1": float((2.0 * tp) / max(2.0 * tp + fp + fn, eps)),
        "building_precision": float(tp / max(tp + fp, eps)),
        "building_recall": float(tp / max(tp + fn, eps)),
        "pixel_accuracy": float((tp + tn) / max(tp + fp + fn + tn, eps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-dataset RSKT-Seg evaluation on xBD-pre binary building "
            "extraction using the released DLRSD-trained checkpoint."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_rskt = repo_root / "third_party" / "RSKT-Seg"
    default_weight_root = Path("/root/data/weight/rskt_seg")
    default_checkpoint = (
        Path("/root/data/weight/RSKT-Seg-ckpt")
        / "0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5"
        / "model_final.pth"
    )
    parser.add_argument("--data_root", default="/root/data/xview2/test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rskt_root", default=str(default_rskt))
    parser.add_argument(
        "--config",
        default=str(default_rskt / "configs" / "vitl_336_DLRSD.yaml"),
    )
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument(
        "--class_json",
        default=str(
            Path(__file__).resolve().parent / "configs" / "xbd_pre_classes.json"
        ),
    )
    parser.add_argument(
        "--clip_vitl",
        default=str(default_weight_root / "pretrained" / "ViT-L-14-336px.pt"),
    )
    parser.add_argument(
        "--clip_vitb",
        default=str(default_weight_root / "pretrained" / "ViT-B-32.pt"),
    )
    parser.add_argument(
        "--remote_clip",
        default=str(
            default_weight_root / "pretrained" / "RemoteCLIP-ViT-B-32.pt"
        ),
    )
    parser.add_argument(
        "--rsib",
        default=os.environ.get("RSKT_RSIB", "/root/data/weight/rsib/RSIB.pth"),
    )
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument(
        "--prompt_ensemble",
        choices=("single", "imagenet", "imagenet_select"),
        default="single",
    )
    parser.add_argument("--amp", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--save_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    if args.input_size <= 0:
        parser.error("--input_size must be positive")
    if args.tile_size <= 0:
        parser.error("--tile_size must be positive")
    if args.num_layers <= 0:
        parser.error("--num_layers must be positive")
    if not torch.cuda.is_available():
        parser.error("RSKT-Seg evaluation requires CUDA")

    required = {
        "official source": _path(args.rskt_root) / "RSKT_Seg" / "RSKT_Seg.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        "class JSON": _path(args.class_json),
        "CLIP ViT-L/14@336": _path(args.clip_vitl),
        "RemoteCLIP ViT-B/32": _path(args.remote_clip),
        "RSIB/DINO": _path(args.rsib),
    }
    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing RSKT-Seg source or weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_rskt_seg.sh and "
            "tools/check_rskt_seg_deps.py."
        )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    _validate_resume_protocol(
        output_root,
        tile_size=args.tile_size,
        model_input_size=args.input_size,
        overwrite=args.overwrite,
    )

    data_root = _path(args.data_root)
    image_dir, label_dir, split = _resolve_xbd_test_dirs(data_root)
    images = _list_pre_images(image_dir)
    pairs = [
        (index, image_path, _find_xbd_label(label_dir, image_path))
        for index, image_path in enumerate(images)
    ]
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    local_pairs = pairs[rank::world_size]

    cfg, model, load_result = _configure_model(args, local_rank)
    if rank == 0:
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": "xBD-pre",
                "classes": [
                    {"id": 0, "name": "background", "rgb": [0, 0, 0]},
                    {"id": 1, "name": "building", "rgb": [255, 255, 255]},
                ],
                "ground_truth_mapping": (
                    "features.xy WKT polygons=building; all other pixels=background"
                ),
                "test_class_json": str(_path(args.class_json)),
                "inference_mode": "native_nonoverlap_tiled",
                "source_tile_size": args.tile_size,
                "model_input_size": args.input_size,
                "padding": "zero_right_bottom",
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                "dataset": "xBD-pre",
                "split": split,
                "method": "RSKT-Seg",
                "protocol": (
                    "DLRSD-trained cross-dataset evaluation on official xBD "
                    "test pre-disaster images with SegEarth-OV-compatible WKT "
                    "rasterization and native non-overlapping tiled inference"
                ),
                "num_samples": len(pairs),
                "inference_mode": "native_nonoverlap_tiled",
                "tile_size": args.tile_size,
                "model_input_size": args.input_size,
                "tile_resize": args.tile_size != args.input_size,
                "padding": "zero_right_bottom",
                "metric_size": "original",
                "label_protocol": "features.xy WKT rounded then cv2.fillPoly",
                "amp": args.amp,
                "prompt_ensemble": args.prompt_ensemble,
                "num_layers": args.num_layers,
                "checkpoint": str(_path(args.checkpoint)),
                "config": str(_path(args.config)),
                "world_size": world_size,
                "primary_metric": "building_iou",
                "load_result": str(load_result),
            },
        )

    rows: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    progress = tqdm(
        local_pairs,
        desc=f"RSKT-Seg xBD-pre rank {rank}",
        disable=False,
    )
    for global_index, image_path, label_path in progress:
        output_stem = f"{global_index:06d}_{image_path.stem}"
        saved_prediction = (
            output_dirs["pred_mask"] / f"{output_stem}_pred_mask.png"
        )
        with Image.open(image_path) as input_image:
            original_width, original_height = input_image.size
        gt = _load_building_mask(
            label_path,
            height=original_height,
            width=original_width,
        )
        num_tiles = (
            _ceil_to_multiple(original_height, args.tile_size) // args.tile_size
        ) * (
            _ceil_to_multiple(original_width, args.tile_size) // args.tile_size
        )

        if saved_prediction.is_file() and not args.overwrite:
            with Image.open(saved_prediction) as image:
                prediction = (
                    np.asarray(image.convert("L"), dtype=np.uint8) != 0
                ).astype(np.uint8)
            with Image.open(image_path) as input_image:
                image_rgb = np.asarray(input_image.convert("RGB"), dtype=np.uint8)
        else:
            image_rgb, prediction, num_tiles = _predict_tiled(
                model=model,
                cfg=cfg,
                image_path=image_path,
                amp=args.amp,
                tile_size=args.tile_size,
            )

        if prediction.shape != gt.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch for {image_path.name}: "
                f"prediction={prediction.shape}, gt={gt.shape}"
            )
        sample_counts = _confusion(prediction, gt)
        for key in counts:
            counts[key] += sample_counts[key]

        if args.save_images and (args.overwrite or not saved_prediction.is_file()):
            Image.fromarray(image_rgb, mode="RGB").save(
                output_dirs["input"] / f"{output_stem}_input.png"
            )
            _save_binary_mask(prediction, saved_prediction)
            _save_color_mask(
                prediction,
                output_dirs["pred_rgb"] / f"{output_stem}_pred_rgb.png",
            )
            _save_binary_mask(
                gt,
                output_dirs["gt_mask"] / f"{output_stem}_gt_mask.png",
            )
            _save_color_mask(
                gt,
                output_dirs["gt_rgb"] / f"{output_stem}_gt_rgb.png",
            )
            _save_overlay(
                image_rgb,
                prediction,
                output_dirs["overlay"] / f"{output_stem}_overlay.png",
                args.overlay_alpha,
            )

        rows.append(
            {
                "index": global_index,
                "image": str(image_path),
                "ground_truth": str(label_path),
                "prediction": str(saved_prediction),
                "original_height": int(original_height),
                "original_width": int(original_width),
                "num_tiles": int(num_tiles),
                **sample_counts,
                **_metrics(sample_counts),
            }
        )

    _write_json(
        output_root / f"rank_{rank:05d}.json",
        {"rank": rank, "counts": counts, "rows": rows},
    )
    _barrier(world_size)

    if rank == 0:
        merged_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        merged_rows: list[dict[str, Any]] = []
        for process_rank in range(world_size):
            rank_file = output_root / f"rank_{process_rank:05d}.json"
            with rank_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for key in merged_counts:
                merged_counts[key] += int(payload["counts"][key])
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["index"]))
        result = {
            "dataset": "xBD-pre",
            "split": split,
            "method": "RSKT-Seg",
            "training_dataset": "DLRSD",
            "evaluation_setting": "cross-dataset/out-of-domain",
            "label_protocol": "SegEarth-OV-compatible xBD WKT rasterization",
            "inference_mode": "native_nonoverlap_tiled",
            "tile_size": args.tile_size,
            "model_input_size": args.input_size,
            "tile_resize": args.tile_size != args.input_size,
            "padding": "zero_right_bottom",
            "metric_size": "original",
            "num_samples": len(merged_rows),
            "num_tiles": sum(int(row["num_tiles"]) for row in merged_rows),
            **merged_counts,
            **_metrics(merged_counts),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[eval_rskt_seg_xbd_pre] saved outputs to: {output_root}")

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
