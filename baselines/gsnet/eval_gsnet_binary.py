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


BASELINES_DIR = Path(__file__).resolve().parents[1]
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_binary_boundary_wfm,
)


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BINARY_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
DATASET_SPECS = {
    "chn6_cug": {
        "display_name": "CHN6-CUG",
        "foreground": "road",
        "model_foreground": "road",
        "default_root": "/root/data/CHN6-CUG/val",
        "class_json": "chn6_cug_classes.json",
        "primary_metric": "road_iou",
    },
    "xbd_pre": {
        "display_name": "xBD-pre",
        "foreground": "building",
        "model_foreground": "building",
        "default_root": "/root/data/xview2/test",
        "class_json": "xbd_pre_classes.json",
        "primary_metric": "building_iou",
    },
}
PREDICTION_PROTOCOL = (
    "complete_nonbackground_taxonomy_argmax_then_target_collapse"
)


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


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Class JSON must contain a non-empty list: {path}")
    class_names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Class JSON entry {index} must be a non-empty string: {path}"
            )
        class_names.append(item.strip())
    normalized = [name.casefold() for name in class_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Class JSON contains duplicate names: {path}")
    if "background" in normalized:
        raise ValueError(
            "GSNet's LandDiscover50K label 0 is ignored during training, so "
            "the text class 'background' is not a valid binary comparator. "
            f"Use a complete non-background taxonomy instead: {path}"
        )
    return class_names


def _resolve_foreground_indices(
    class_names: list[str],
    foreground_class: str,
) -> list[int]:
    normalized_foreground = foreground_class.strip().casefold()
    indices = [
        index
        for index, class_name in enumerate(class_names)
        if class_name.casefold() == normalized_foreground
    ]
    if not indices:
        raise ValueError(
            f"Foreground class {foreground_class!r} is absent from the test "
            f"taxonomy: {class_names}"
        )
    return indices


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # Each rank owns an independent GSNet instance. Gloo is used only for
        # barriers and metadata synchronization, not model collectives.
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
        "pred_class_id": root / "pred_class_id",
        "pred_rgb": root / "pred_rgb",
        "gt_mask": root / "gt_mask",
        "gt_rgb": root / "gt_rgb",
        "overlay": root / "overlay",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _save_id_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_color_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(BINARY_PALETTE[mask.clip(0, 1)], mode="RGB").save(path)


def _save_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    path: Path,
    alpha: float,
) -> None:
    output = image_rgb.astype(np.float32).copy()
    foreground = mask == 1
    output[foreground] = (
        (1.0 - alpha) * output[foreground]
        + alpha * BINARY_PALETTE[1]
    )
    Image.fromarray(output.clip(0, 255).astype(np.uint8), mode="RGB").save(path)


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
    details = "\n".join(
        f"  images={image_dir} masks={mask_dir}"
        for image_dir, mask_dir in checked
    )
    raise NotADirectoryError(
        f"Cannot find CHN6-CUG image/mask folders under {data_root}. "
        f"Checked:\n{details}"
    )


def _resolve_xbd_dirs(data_root: Path) -> tuple[Path, Path, str]:
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
        f"Cannot find xBD test image/label folders under {data_root}. "
        f"Checked:\n{details}"
    )


def _list_images(image_dir: Path, dataset: str) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
        and (dataset != "xbd_pre" or path.stem.endswith("_pre_disaster"))
    ]
    if not images:
        suffix = " matching *_pre_disaster" if dataset == "xbd_pre" else ""
        raise FileNotFoundError(
            f"No supported images{suffix} found under {image_dir}"
        )
    return images


def _find_chn6_mask(mask_dir: Path, image_path: Path) -> Path:
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
        f"Cannot find a CHN6-CUG mask matching {image_path.name} "
        f"under {mask_dir}"
    )


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


def _load_target(
    dataset: str,
    label_path: Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if dataset == "xbd_pre" and label_path.suffix.lower() == ".json":
        # Reuse the exact rasterizer used by the RSKT-Seg adapter so both
        # baselines share the SegEarth-OV-compatible xBD label protocol.
        rskt_dir = Path(__file__).resolve().parents[1] / "rskt_seg"
        if str(rskt_dir) not in sys.path:
            sys.path.insert(0, str(rskt_dir))
        from xbd_label_utils import load_xbd_building_mask

        target = load_xbd_building_mask(
            label_path,
            height=height,
            width=width,
        )
    else:
        with Image.open(label_path) as image:
            target = (
                np.asarray(image.convert("L"), dtype=np.uint8) != 0
            ).astype(np.uint8)
    if target.shape != (height, width):
        raise ValueError(
            f"Image/label size mismatch for {label_path}: "
            f"expected={(height, width)}, got={target.shape}"
        )
    return target


def _confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    predicted_foreground = prediction.reshape(-1) == 1
    target_foreground = target.reshape(-1) == 1
    return {
        "tp": int(np.count_nonzero(predicted_foreground & target_foreground)),
        "fp": int(np.count_nonzero(predicted_foreground & ~target_foreground)),
        "fn": int(np.count_nonzero(~predicted_foreground & target_foreground)),
        "tn": int(np.count_nonzero(~predicted_foreground & ~target_foreground)),
    }


def _metrics(counts: dict[str, int], foreground: str) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1.0e-12
    foreground_iou = tp / max(tp + fp + fn, eps)
    background_iou = tn / max(tn + fp + fn, eps)
    return {
        f"{foreground}_iou": float(foreground_iou),
        "background_iou": float(background_iou),
        "miou": float((foreground_iou + background_iou) * 0.5),
        f"{foreground}_f1": float(
            (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
        ),
        f"{foreground}_precision": float(tp / max(tp + fp, eps)),
        f"{foreground}_recall": float(tp / max(tp + fn, eps)),
        "pixel_accuracy": float(
            (tp + tn) / max(tp + fp + fn + tn, eps)
        ),
    }


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _tile_coords(height: int, width: int, tile_size: int) -> list[tuple[int, int]]:
    return [
        (y, x)
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]


def _configure_model(
    args: argparse.Namespace,
    local_rank: int,
    *,
    num_test_classes: int,
):
    source_root = _path(args.gsnet_root)
    sys.path.insert(0, str(source_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    from gs_net import add_cat_seg_config
    from gs_net.third_party import clip as official_clip

    config_path = _path(args.config)
    clip_path = _path(args.clip_vitb)
    rsib_path = _path(args.rsib)
    os.environ["RSIB_CKPT"] = str(rsib_path)

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = f"cuda:{local_rank}"
    cfg.MODEL.WEIGHTS = str(_path(args.checkpoint))
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(
        source_root / "datasets" / "landdiscover.json"
    )
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(_path(args.class_json))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_test_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = int(args.num_layers)
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    cfg.MODEL.SEM_SEG_HEAD.DINO_WEIGHTS = str(rsib_path)
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = args.prompt_ensemble
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_size)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_size)
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()

    # The official CLIP helper always resolves ViT-B/16 through ~/.cache/clip.
    # Redirect only that download to the explicitly managed, checksum-verified
    # file while leaving all other official behavior unchanged.
    original_download = official_clip._download

    def use_managed_clip(url: str, root: str | None = None) -> str:
        if Path(url).name == "ViT-B-16.pt":
            return str(clip_path)
        if root is None:
            return original_download(url)
        return original_download(url, root)

    # These released checkpoints predate the PyTorch 2.6 weights_only default.
    # They are explicitly supplied official release files, so preserve the
    # legacy loader only during model construction and checkpoint loading.
    original_torch_load = torch.load

    def load_trusted_release(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    official_clip._download = use_managed_clip
    torch.load = load_trusted_release
    try:
        model = build_model(cfg)
        model.eval()
        load_result = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    finally:
        official_clip._download = original_download
        torch.load = original_torch_load
    return cfg, model, load_result


def _predict_tiled(
    *,
    model: torch.nn.Module,
    cfg: Any,
    image_path: Path,
    amp: str,
    tile_size: int,
    num_test_classes: int,
    foreground_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from detectron2.data import detection_utils as utils
    from detectron2.data import transforms as transforms

    image = utils.read_image(str(image_path), format=cfg.INPUT.FORMAT)
    original_height, original_width = image.shape[:2]
    image_rgb = (
        image.copy()
        if cfg.INPUT.FORMAT == "RGB"
        else utils.read_image(str(image_path), format="RGB")
    )

    padded_height = _ceil_to_multiple(original_height, tile_size)
    padded_width = _ceil_to_multiple(original_width, tile_size)
    padded_image = np.zeros(
        (padded_height, padded_width, image.shape[2]),
        dtype=image.dtype,
    )
    padded_image[:original_height, :original_width] = image
    class_prediction_padded = np.zeros(
        (padded_height, padded_width),
        dtype=np.uint8,
    )
    coordinates = _tile_coords(padded_height, padded_width, tile_size)

    resize = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[amp]

    # The official non-sliding forward path returns one item, so process each
    # native tile independently. GSNet then performs its official internal
    # 384x384 CLIP/RSIB resampling before returning a tile-sized prediction.
    for y, x in coordinates:
        tile = padded_image[y : y + tile_size, x : x + tile_size]
        transformed = resize.get_transform(tile).apply_image(tile)
        tensor = torch.as_tensor(
            transformed.astype("float32").transpose(2, 0, 1)
        )
        model_input = {
            "image": tensor,
            "height": tile_size,
            "width": tile_size,
            "file_name": str(image_path),
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=amp != "fp32",
        ):
            output = model([model_input])[0]["sem_seg"]
        if output.ndim != 3 or output.shape[0] != num_test_classes:
            raise RuntimeError(
                "GSNet output-channel count does not match the configured "
                f"test taxonomy: shape={tuple(output.shape)}, "
                f"expected_classes={num_test_classes}"
            )
        class_prediction = (
            output.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
        )
        if class_prediction.shape != (tile_size, tile_size):
            class_prediction = cv2.resize(
                class_prediction,
                (tile_size, tile_size),
                interpolation=cv2.INTER_NEAREST,
            )
        class_prediction_padded[
            y : y + tile_size,
            x : x + tile_size,
        ] = class_prediction

    class_prediction = class_prediction_padded[
        :original_height,
        :original_width,
    ].copy()
    binary_prediction = np.isin(
        class_prediction,
        np.asarray(foreground_indices, dtype=np.uint8),
    ).astype(np.uint8)

    return (
        image_rgb,
        binary_prediction,
        class_prediction,
        len(coordinates),
    )


def _validate_resume_protocol(
    output_root: Path,
    *,
    dataset: str,
    tile_size: int,
    model_input_size: int,
    class_names: list[str],
    foreground_class: str,
    prompt_ensemble: str,
    num_layers: int,
    checkpoint: Path,
    config: Path,
    clip_vitb: Path,
    rsib: Path,
    amp: str,
    overwrite: bool,
) -> None:
    predictions = list((output_root / "pred_mask").glob("*_pred_mask.png"))
    if overwrite or not predictions:
        return
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"{output_root} contains predictions without run_config.json. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    expected = {
        "dataset_key": dataset,
        "method": "GSNet",
        "inference_mode": "native_nonoverlap_tiled",
        "tile_size": int(tile_size),
        "model_input_size": int(model_input_size),
        "padding": "zero_right_bottom",
        "metric_size": "original",
        "prediction_protocol": PREDICTION_PROTOCOL,
        "test_classes": class_names,
        "foreground_model_class": foreground_class,
        "prompt_ensemble": prompt_ensemble,
        "num_layers": int(num_layers),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "clip_vitb": str(clip_vitb),
        "rsib": str(rsib),
        "amp": amp,
        "pooling_sizes": [1, 1],
    }
    mismatches = {
        key: {"existing": existing.get(key), "expected": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing predictions use a different evaluation protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official GSNet checkpoint evaluation on CHN6-CUG road or "
            "xBD-pre building segmentation with native 512 tiled inference."
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_source = repo_root / "third_party" / "GSNet"
    default_weight_root = Path("/root/data/weight/gsnet")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        required=True,
    )
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gsnet_root", default=str(default_source))
    parser.add_argument(
        "--config",
        default=str(default_source / "configs" / "vitb_384.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(default_weight_root / "GSNet_base.pth"),
    )
    parser.add_argument("--class_json", default="")
    parser.add_argument("--foreground_class", default="")
    parser.add_argument(
        "--clip_vitb",
        default=str(default_weight_root / "pretrained" / "ViT-B-16.pt"),
    )
    parser.add_argument(
        "--rsib",
        default=os.environ.get(
            "GSNET_RSIB",
            "/root/data/weight/rsib/RSIB.pth",
        ),
    )
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
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

    spec = DATASET_SPECS[args.dataset]
    if not args.data_root:
        args.data_root = spec["default_root"]
    if not args.class_json:
        args.class_json = str(
            Path(__file__).resolve().parent
            / "configs"
            / spec["class_json"]
        )
    if not args.foreground_class:
        args.foreground_class = str(spec["model_foreground"])
    if args.input_size <= 0:
        parser.error("--input_size must be positive")
    if args.tile_size <= 0:
        parser.error("--tile_size must be positive")
    if args.num_layers <= 0:
        parser.error("--num_layers must be positive")
    if not torch.cuda.is_available():
        parser.error("GSNet evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    spec = DATASET_SPECS[args.dataset]
    foreground = str(spec["foreground"])
    class_json_path = _path(args.class_json)
    class_names = _load_class_names(class_json_path)
    foreground_indices = _resolve_foreground_indices(
        class_names,
        args.foreground_class,
    )

    required = {
        "official source": _path(args.gsnet_root) / "gs_net" / "GSNet.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        "class JSON": class_json_path,
        "CLIP ViT-B/16": _path(args.clip_vitb),
        "RSIB/DINO": _path(args.rsib),
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing GSNet source or weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_gsnet.sh and "
            "tools/check_gsnet_deps.py."
        )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    _validate_resume_protocol(
        output_root,
        dataset=args.dataset,
        tile_size=args.tile_size,
        model_input_size=args.input_size,
        class_names=class_names,
        foreground_class=args.foreground_class,
        prompt_ensemble=args.prompt_ensemble,
        num_layers=args.num_layers,
        checkpoint=_path(args.checkpoint),
        config=_path(args.config),
        clip_vitb=_path(args.clip_vitb),
        rsib=_path(args.rsib),
        amp=args.amp,
        overwrite=args.overwrite,
    )

    data_root = _path(args.data_root)
    if args.dataset == "chn6_cug":
        image_dir, label_dir, split = _resolve_chn6_dirs(data_root)
    else:
        image_dir, label_dir, split = _resolve_xbd_dirs(data_root)
    images = _list_images(image_dir, args.dataset)
    pairs: list[tuple[int, Path, Path]] = []
    for index, image_path in enumerate(images):
        label_path = (
            _find_chn6_mask(label_dir, image_path)
            if args.dataset == "chn6_cug"
            else _find_xbd_label(label_dir, image_path)
        )
        pairs.append((index, image_path, label_path))
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    local_pairs = pairs[rank::world_size]

    cfg, model, load_result = _configure_model(
        args,
        local_rank,
        num_test_classes=len(class_names),
    )
    if rank == 0:
        label_protocol = (
            "zero=background, nonzero=road"
            if args.dataset == "chn6_cug"
            else "features.xy WKT rounded then cv2.fillPoly"
        )
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": spec["display_name"],
                "classes": [
                    {"id": 0, "name": "background", "rgb": [0, 0, 0]},
                    {"id": 1, "name": foreground, "rgb": [255, 255, 255]},
                ],
                "ground_truth_mapping": label_protocol,
                "test_class_json": str(class_json_path),
                "test_classes": class_names,
                "foreground_model_class": args.foreground_class,
                "foreground_class_indices": foreground_indices,
                "prediction_protocol": PREDICTION_PROTOCOL,
                "inference_mode": "native_nonoverlap_tiled",
                "source_tile_size": args.tile_size,
                "model_input_size": args.input_size,
                "encoder_internal_size": 384,
                "padding": "zero_right_bottom",
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                "dataset": spec["display_name"],
                "dataset_key": args.dataset,
                "split": split,
                "method": "GSNet",
                "training_dataset": "LandDiscover50K",
                "evaluation_setting": "cross-dataset/out-of-domain",
                "num_samples": len(pairs),
                "inference_mode": "native_nonoverlap_tiled",
                "tile_size": args.tile_size,
                "model_input_size": args.input_size,
                "tile_resize": args.tile_size != args.input_size,
                "encoder_internal_size": 384,
                "padding": "zero_right_bottom",
                "metric_size": "original",
                "prediction_protocol": PREDICTION_PROTOCOL,
                "label_protocol": label_protocol,
                "amp": args.amp,
                "prompt_ensemble": args.prompt_ensemble,
                "num_layers": args.num_layers,
                "pooling_sizes": [1, 1],
                "test_classes": class_names,
                "num_test_classes": len(class_names),
                "foreground_model_class": args.foreground_class,
                "foreground_class_indices": foreground_indices,
                "checkpoint": str(_path(args.checkpoint)),
                "config": str(_path(args.config)),
                "class_json": str(class_json_path),
                "clip_vitb": str(_path(args.clip_vitb)),
                "rsib": str(_path(args.rsib)),
                "world_size": world_size,
                "primary_metric": spec["primary_metric"],
                "load_result": str(load_result),
            },
        )

    rows: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    progress = tqdm(
        local_pairs,
        desc=f"GSNet {spec['display_name']} rank {rank}",
        disable=False,
    )
    for global_index, image_path, label_path in progress:
        output_stem = f"{global_index:06d}_{image_path.stem}"
        saved_prediction = (
            output_dirs["pred_mask"] / f"{output_stem}_pred_mask.png"
        )
        saved_class_prediction = (
            output_dirs["pred_class_id"] / f"{output_stem}_pred_class_id.png"
        )
        with Image.open(image_path) as input_image:
            original_width, original_height = input_image.size
        target = _load_target(
            args.dataset,
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
            if not saved_class_prediction.is_file():
                raise RuntimeError(
                    f"Binary prediction exists without its taxonomy class-ID "
                    f"map: {saved_class_prediction}. Use OVERWRITE=1 or a new "
                    "OUTPUT_DIR."
                )
            with Image.open(saved_prediction) as image:
                cached_binary_prediction = (
                    np.asarray(image.convert("L"), dtype=np.uint8) != 0
                ).astype(np.uint8)
            with Image.open(saved_class_prediction) as image:
                class_prediction = np.asarray(
                    image.convert("L"),
                    dtype=np.uint8,
                )
            if class_prediction.size and int(class_prediction.max()) >= len(
                class_names
            ):
                raise ValueError(
                    f"Cached class-ID map contains an invalid class index: "
                    f"{saved_class_prediction}"
                )
            prediction = np.isin(
                class_prediction,
                np.asarray(foreground_indices, dtype=np.uint8),
            ).astype(np.uint8)
            if not np.array_equal(prediction, cached_binary_prediction):
                raise ValueError(
                    "Cached binary prediction is inconsistent with its "
                    f"taxonomy class-ID map: {saved_prediction}. Use "
                    "OVERWRITE=1 or a new OUTPUT_DIR."
                )
            with Image.open(image_path) as input_image:
                image_rgb = np.asarray(
                    input_image.convert("RGB"),
                    dtype=np.uint8,
                )
        else:
            image_rgb, prediction, class_prediction, num_tiles = _predict_tiled(
                model=model,
                cfg=cfg,
                image_path=image_path,
                amp=args.amp,
                tile_size=args.tile_size,
                num_test_classes=len(class_names),
                foreground_indices=foreground_indices,
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch for {image_path.name}: "
                f"prediction={prediction.shape}, gt={target.shape}"
            )
        if class_prediction.shape != target.shape:
            raise ValueError(
                f"Class prediction/GT shape mismatch for {image_path.name}: "
                f"prediction={class_prediction.shape}, gt={target.shape}"
            )
        sample_counts = _confusion(prediction, target)
        sample_wfm = score_binary_boundary_wfm(prediction, target)
        class_histogram = np.bincount(
            class_prediction.reshape(-1),
            minlength=len(class_names),
        )
        for key in counts:
            counts[key] += sample_counts[key]

        if args.save_images and (args.overwrite or not saved_prediction.is_file()):
            Image.fromarray(image_rgb, mode="RGB").save(
                output_dirs["input"] / f"{output_stem}_input.png"
            )
            _save_id_mask(prediction, saved_prediction)
            _save_id_mask(class_prediction, saved_class_prediction)
            _save_color_mask(
                prediction,
                output_dirs["pred_rgb"] / f"{output_stem}_pred_rgb.png",
            )
            _save_id_mask(
                target,
                output_dirs["gt_mask"] / f"{output_stem}_gt_mask.png",
            )
            _save_color_mask(
                target,
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
                "class_prediction": str(saved_class_prediction),
                "original_height": int(original_height),
                "original_width": int(original_width),
                "num_tiles": int(num_tiles),
                "ground_truth_foreground_pixels": int(np.count_nonzero(target)),
                "predicted_foreground_pixels": int(
                    np.count_nonzero(prediction)
                ),
                "predicted_class_pixels": {
                    class_name: int(class_histogram[index])
                    for index, class_name in enumerate(class_names)
                    if int(class_histogram[index]) > 0
                },
                **sample_counts,
                **_metrics(sample_counts, foreground),
                **sample_wfm,
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
        merged_class_histogram = {
            class_name: 0 for class_name in class_names
        }
        for process_rank in range(world_size):
            rank_file = output_root / f"rank_{process_rank:05d}.json"
            with rank_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for key in merged_counts:
                merged_counts[key] += int(payload["counts"][key])
            for row in payload["rows"]:
                for class_name, count in row["predicted_class_pixels"].items():
                    merged_class_histogram[class_name] += int(count)
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["index"]))
        total_pixels = sum(merged_counts.values())
        ground_truth_foreground_pixels = (
            merged_counts["tp"] + merged_counts["fn"]
        )
        predicted_foreground_pixels = (
            merged_counts["tp"] + merged_counts["fp"]
        )
        result = {
            "dataset": spec["display_name"],
            "split": split,
            "method": "GSNet",
            "training_dataset": "LandDiscover50K",
            "evaluation_setting": "cross-dataset/out-of-domain",
            "label_protocol": (
                "SegEarth-OV-compatible xBD WKT rasterization"
                if args.dataset == "xbd_pre"
                else "zero=background, nonzero=road"
            ),
            "inference_mode": "native_nonoverlap_tiled",
            "tile_size": args.tile_size,
            "model_input_size": args.input_size,
            "tile_resize": args.tile_size != args.input_size,
            "encoder_internal_size": 384,
            "padding": "zero_right_bottom",
            "metric_size": "original",
            "prediction_protocol": PREDICTION_PROTOCOL,
            "test_classes": class_names,
            "num_test_classes": len(class_names),
            "foreground_model_class": args.foreground_class,
            "foreground_class_indices": foreground_indices,
            "num_samples": len(merged_rows),
            "num_tiles": sum(int(row["num_tiles"]) for row in merged_rows),
            "num_pixels": total_pixels,
            "ground_truth_foreground_pixels": ground_truth_foreground_pixels,
            "predicted_foreground_pixels": predicted_foreground_pixels,
            "ground_truth_foreground_fraction": (
                ground_truth_foreground_pixels / max(total_pixels, 1)
            ),
            "predicted_foreground_fraction": (
                predicted_foreground_pixels / max(total_pixels, 1)
            ),
            "predicted_class_pixels": {
                class_name: count
                for class_name, count in merged_class_histogram.items()
                if count > 0
            },
            **merged_counts,
            **_metrics(merged_counts, foreground),
            **aggregate_binary_boundary_wfm(merged_rows),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[eval_gsnet_binary] saved outputs to: {output_root}")

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
