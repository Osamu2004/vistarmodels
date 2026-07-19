from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ROAD_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [part for part in text.strip("\\").split("\\") if part]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _path(value: str) -> Path:
    return Path(_normalize_wsl_unc(value)).expanduser().resolve()


def _resolve_chn6_dirs(data_root: Path) -> tuple[Path, Path, str]:
    checked: list[tuple[Path, Path]] = []
    split_roots = [(data_root, data_root.name), (data_root / "val", "val")]
    layouts = (
        ("images", "gt"),
        ("images", "labels"),
        ("images", "masks"),
        ("image", "label"),
        ("imgs", "gt"),
        ("imgs", "masks"),
    )
    for split_root, split_name in split_roots:
        for image_name, mask_name in layouts:
            image_dir = split_root / image_name
            mask_dir = split_root / mask_name
            checked.append((image_dir, mask_dir))
            if image_dir.is_dir() and mask_dir.is_dir():
                return image_dir, mask_dir, split_name
    details = "\n".join(f"  images={images} masks={masks}" for images, masks in checked)
    raise NotADirectoryError(
        f"Cannot find CHN6-CUG image/mask folders under {data_root}. Checked:\n{details}"
    )


def _list_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found under {image_dir}")
    return images


def _find_mask(mask_dir: Path, image_path: Path) -> Path:
    stems: list[str] = []
    if "_sat" in image_path.stem:
        stems.append(image_path.stem.replace("_sat", "_mask"))
    stems.append(image_path.stem)
    for stem in stems:
        for extension in MASK_EXTS:
            candidate = mask_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Cannot find a CHN6-CUG mask matching {image_path.name} under {mask_dir}"
    )


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return (array != 0).astype(np.uint8)


def _save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_color_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(ROAD_PALETTE[mask.clip(0, 1)], mode="RGB").save(path)


def _save_overlay(image_rgb: np.ndarray, mask: np.ndarray, path: Path, alpha: float) -> None:
    output = image_rgb.astype(np.float32).copy()
    road = mask == 1
    output[road] = (1.0 - alpha) * output[road] + alpha * ROAD_PALETTE[1]
    Image.fromarray(output.clip(0, 255).astype(np.uint8), mode="RGB").save(path)


def _confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_road = prediction.reshape(-1) == 1
    gt_road = target.reshape(-1) == 1
    return {
        "tp": int(np.count_nonzero(pred_road & gt_road)),
        "fp": int(np.count_nonzero(pred_road & ~gt_road)),
        "fn": int(np.count_nonzero(~pred_road & gt_road)),
        "tn": int(np.count_nonzero(~pred_road & ~gt_road)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1.0e-12
    road_iou = tp / max(tp + fp + fn, eps)
    background_iou = tn / max(tn + fp + fn, eps)
    return {
        "road_iou": float(road_iou),
        "background_iou": float(background_iou),
        "miou": float((road_iou + background_iou) * 0.5),
        "road_f1": float((2.0 * tp) / max(2.0 * tp + fp + fn, eps)),
        "road_precision": float(tp / max(tp + fp, eps)),
        "road_recall": float(tp / max(tp + fn, eps)),
        "pixel_accuracy": float((tp + tn) / max(tp + fp + fn + tn, eps)),
    }


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # RSKT-Seg inference is data parallel without DDP. Gloo is used only
        # for a CPU barrier and metadata synchronization; no NCCL collectives
        # are needed.
        torch.distributed.init_process_group(backend="gloo")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def _make_output_dirs(root: Path) -> dict[str, Path]:
    directories = {
        "input": root / "input",
        "pred_mask": root / "pred_mask",
        "pred_rgb": root / "pred_rgb",
        "gt_mask": root / "gt_mask",
        "gt_rgb": root / "gt_rgb",
        "overlay": root / "overlay",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _configure_model(args: argparse.Namespace, local_rank: int):
    source_root = _path(args.rskt_root)
    sys.path.insert(0, str(source_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    from RSKT_Seg import add_RSKT_seg_config
    from RSKT_Seg.third_party import clip as official_clip

    # The official release hard-codes ./pretrained paths inside clip.py.
    # Override the lookup table in memory so large weights remain outside Git.
    official_clip.pretrained["ViT-L/14@336px"] = str(_path(args.clip_vitl))
    clip_vitb = _path(args.clip_vitb)
    if clip_vitb.is_file():
        official_clip.pretrained["ViT-B/32"] = str(clip_vitb)

    config_path = _path(args.config)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_RSKT_seg_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = f"cuda:{local_rank}"
    cfg.MODEL.WEIGHTS = str(_path(args.checkpoint))
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(
        source_root / "datasets" / "DLRSD.json"
    )
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(_path(args.class_json))
    cfg.MODEL.SEM_SEG_HEAD.DINO_WEIGHTS = str(_path(args.rsib))
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS_REMOTE = str(
        _path(args.remote_clip)
    )
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = int(args.num_layers)
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = args.prompt_ensemble
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_size)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_size)
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()

    # RSKT-Seg's released RSIB and model checkpoints predate PyTorch 2.6.
    # Their upstream loaders omit weights_only, whose default changed to True
    # in PyTorch 2.6. These paths are explicitly supplied official weights, so
    # retain the legacy loading behavior only while constructing/loading this
    # model instead of modifying the vendored upstream source.
    original_torch_load = torch.load

    def load_trusted_release(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = load_trusted_release
    try:
        model = build_model(cfg)
        model.eval()
        load_result = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    finally:
        torch.load = original_torch_load
    return cfg, model, load_result


def _predict(
    *,
    model: torch.nn.Module,
    cfg: Any,
    image_path: Path,
    amp: str,
) -> tuple[np.ndarray, np.ndarray]:
    from detectron2.data import detection_utils as utils
    from detectron2.data import transforms as transforms

    image = utils.read_image(str(image_path), format=cfg.INPUT.FORMAT)
    original_height, original_width = image.shape[:2]
    resize = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    transformed = resize.get_transform(image).apply_image(image)
    tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)
    )
    model_input = {
        "image": tensor,
        "height": original_height,
        "width": original_width,
        "file_name": str(image_path),
    }
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[amp]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=amp != "fp32",
    ):
        output = model([model_input])[0]["sem_seg"]
    if output.ndim != 3 or output.shape[0] != 2:
        raise RuntimeError(
            "RSKT-Seg did not return two CHN6-CUG class maps: "
            f"shape={tuple(output.shape)}"
        )
    prediction = output.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
    return image, prediction


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-dataset RSKT-Seg evaluation on CHN6-CUG binary road "
            "segmentation using the released DLRSD-trained checkpoint."
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
    parser.add_argument("--data_root", default="/root/data/CHN6-CUG/val")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rskt_root", default=str(default_rskt))
    parser.add_argument(
        "--config",
        default=str(default_rskt / "configs" / "vitl_336_DLRSD.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint),
    )
    parser.add_argument(
        "--class_json",
        default=str(
            Path(__file__).resolve().parent / "configs" / "chn6_cug_classes.json"
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
        default=os.environ.get(
            "RSKT_RSIB",
            "/root/data/weight/rsib/RSIB.pth",
        ),
    )
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument(
        "--prompt_ensemble",
        choices=("single", "imagenet", "imagenet_select"),
        default="single",
    )
    parser.add_argument("--amp", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    if args.input_size <= 0:
        parser.error("--input_size must be positive")
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
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing RSKT-Seg source or weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_rskt_seg.sh and tools/check_rskt_seg_deps.py."
        )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)

    data_root = _path(args.data_root)
    image_dir, mask_dir, split = _resolve_chn6_dirs(data_root)
    images = _list_images(image_dir)
    pairs = [(index, image, _find_mask(mask_dir, image)) for index, image in enumerate(images)]
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    local_pairs = pairs[rank::world_size]

    cfg, model, load_result = _configure_model(args, local_rank)
    if rank == 0:
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": "CHN6-CUG",
                "classes": [
                    {"id": 0, "name": "background", "rgb": [0, 0, 0]},
                    {"id": 1, "name": "road", "rgb": [255, 255, 255]},
                ],
                "ground_truth_mapping": "zero=background, nonzero=road",
                "test_class_json": str(_path(args.class_json)),
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                "dataset": "CHN6-CUG",
                "split": split,
                "method": "RSKT-Seg",
                "protocol": "DLRSD-trained cross-dataset evaluation",
                "num_samples": len(pairs),
                "input_size": args.input_size,
                "amp": args.amp,
                "prompt_ensemble": args.prompt_ensemble,
                "num_layers": args.num_layers,
                "checkpoint": str(_path(args.checkpoint)),
                "config": str(_path(args.config)),
                "world_size": world_size,
                "primary_metric": "road_iou",
                "load_result": str(load_result),
            },
        )

    rows: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    progress = tqdm(
        local_pairs,
        desc=f"RSKT-Seg CHN6-CUG rank {rank}",
        disable=False,
    )
    for global_index, image_path, mask_path in progress:
        output_stem = f"{global_index:06d}_{image_path.stem}"
        saved_prediction = output_dirs["pred_mask"] / f"{output_stem}_pred_mask.png"
        gt = _load_binary_mask(mask_path)

        if saved_prediction.is_file() and not args.overwrite:
            prediction = _load_binary_mask(saved_prediction)
            with Image.open(image_path) as input_image:
                image_rgb = np.asarray(input_image.convert("RGB"), dtype=np.uint8)
        else:
            image_rgb, prediction = _predict(
                model=model,
                cfg=cfg,
                image_path=image_path,
                amp=args.amp,
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
                "ground_truth": str(mask_path),
                "prediction": str(saved_prediction),
                **sample_counts,
                **_metrics(sample_counts),
            }
        )

    rank_payload = {
        "rank": rank,
        "counts": counts,
        "rows": rows,
    }
    _write_json(output_root / f"rank_{rank:05d}.json", rank_payload)
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
            "dataset": "CHN6-CUG",
            "split": split,
            "method": "RSKT-Seg",
            "training_dataset": "DLRSD",
            "evaluation_setting": "cross-dataset/out-of-domain",
            "num_samples": len(merged_rows),
            **merged_counts,
            **_metrics(merged_counts),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[eval_rskt_seg_chn6] saved outputs to: {output_root}")

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
