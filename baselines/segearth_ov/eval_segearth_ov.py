from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)
from baselines.segearth_ov.protocols import (  # noqa: E402
    DATASET_SPECS,
    colorize_mask,
    confusion_matrix,
    discover_dataset,
    load_rgb,
    load_target,
    metrics_for_dataset,
    normalize_path,
    read_class_groups,
    resize_keep_ratio,
    validate_class_groups,
    write_json,
)


SEGEARTH_OV_SOURCE_REVISION = "3e22a969b32c6d751bdbba64a88a0b670e630f55"
SIMFEATUP_SOURCE_REVISION = "78a0ba70b1d6ea7283684a88c98ce338af4593ca"
CLIP_MEAN_255 = np.asarray([122.771, 116.746, 104.094], dtype=np.float32)
CLIP_STD_255 = np.asarray([68.501, 66.632, 70.323], dtype=np.float32)

DEFAULT_DATA_ROOTS = {
    "loveda": "/root/data/LoveDA",
    "flair": "/root/data/FLAIR-1-2/data/flair#1-test",
    "uavid": "/root/data/OVSISBenchDataset/uavid",
    "xbd_pre": "/root/data/xview2/test",
    "chn6_cug": "/root/data/CHN6-CUG/val",
}
DEFAULT_CLASS_FILES = {
    "loveda": "loveda.txt",
    "flair": "flair_12.txt",
    "uavid": "uavid_8.txt",
    "xbd_pre": "xbd_pre.txt",
    "chn6_cug": "chn6_cug.txt",
}


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # Model instances are independent; Gloo only synchronizes result files.
        torch.distributed.init_process_group(backend="gloo")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def _source_revision(source_root: Path) -> str:
    if not (source_root / ".git").is_dir():
        return "unavailable-no-git-metadata"
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen and not isinstance(row[key], (dict, list)):
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_id_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(path)


def _save_visualizations(
    output_dirs: dict[str, Path],
    name: str,
    image_rgb: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    palette: np.ndarray,
    *,
    ignore_index: int | None,
    prediction_ignore_index: int | None = None,
    overlay_alpha: float,
) -> None:
    pred_rgb = colorize_mask(
        prediction,
        palette,
        ignore_index=prediction_ignore_index,
    )
    gt_rgb = colorize_mask(target, palette, ignore_index=ignore_index)
    valid = np.ones(target.shape, dtype=bool)
    if ignore_index is not None:
        valid = target != int(ignore_index)
    overlay = np.asarray(image_rgb, dtype=np.float32).copy()
    overlay[valid] = (
        (1.0 - overlay_alpha) * overlay[valid]
        + overlay_alpha * pred_rgb[valid].astype(np.float32)
    )
    Image.fromarray(image_rgb, mode="RGB").save(output_dirs["input"] / f"{name}.png")
    Image.fromarray(pred_rgb, mode="RGB").save(output_dirs["pred_rgb"] / f"{name}.png")
    Image.fromarray(gt_rgb, mode="RGB").save(output_dirs["gt_rgb"] / f"{name}.png")
    _save_id_mask(target, output_dirs["gt_mask"] / f"{name}.png")
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8), mode="RGB").save(
        output_dirs["overlay"] / f"{name}.png"
    )


def _build_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    probability_threshold: float,
    cls_token_lambda: float,
):
    """Instantiate the pinned official model while using managed weights."""

    source_root = normalize_path(args.segearth_root)
    simfeatup_root = normalize_path(args.simfeatup_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    if args.feature_up:
        featup_module = importlib.import_module("featup")
        featup_file = Path(str(getattr(featup_module, "__file__", ""))).resolve()
        if simfeatup_root not in featup_file.parents:
            raise RuntimeError(
                "The imported featup package does not come from the pinned "
                f"SimFeatUp checkout: imported={featup_file}, "
                f"expected_root={simfeatup_root}. Re-run the bootstrap in the "
                "active environment."
            )

    # ``segearth_segmentor`` imports the repository-vendored ``open_clip``.
    # Ensure a previously imported site-package cannot silently win resolution.
    loaded_open_clip = sys.modules.get("open_clip")
    if loaded_open_clip is not None:
        module_file = Path(str(getattr(loaded_open_clip, "__file__", ""))).resolve()
        if source_root not in module_file.parents:
            raise RuntimeError(
                "A non-SegEarth open_clip module was imported before model setup: "
                f"{module_file}. Use a clean SegEarth-OV environment."
            )

    official_factory = importlib.import_module("open_clip.factory")
    original_download_pretrained = official_factory.download_pretrained
    original_torch_load = torch.load
    clip_path = normalize_path(args.clip_vitb)

    def use_managed_clip(pretrained_cfg, cache_dir=None):
        _ = pretrained_cfg, cache_dir
        return str(clip_path)

    def load_trusted_official(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    official_factory.download_pretrained = use_managed_clip
    torch.load = load_trusted_official
    try:
        module = importlib.import_module("segearth_segmentor")
        model = module.SegEarthSegmentation(
            clip_type="CLIP",
            vit_type="ViT-B/16",
            model_type="SegEarth",
            name_path=str(normalize_path(args.class_file)),
            device=device,
            ignore_residual=True,
            prob_thd=float(probability_threshold),
            logit_scale=50,
            slide_stride=int(args.slide_stride),
            slide_crop=int(args.slide_crop),
            cls_token_lambda=float(cls_token_lambda),
            bg_idx=0,
            feature_up=bool(args.feature_up),
            feature_up_cfg={
                "model_name": "jbu_one",
                "model_path": str(normalize_path(args.simfeatup_checkpoint)),
            },
        )
    finally:
        official_factory.download_pretrained = original_download_pretrained
        torch.load = original_torch_load
    model.eval()
    return model


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    *,
    device: torch.device,
    input_size: int,
    slide_crop: int,
    slide_stride: int,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, int]]:
    original_height, original_width = image_rgb.shape[:2]
    resized = resize_keep_ratio(image_rgb, input_size)
    resized_height, resized_width = resized.shape[:2]
    normalized = (
        resized.astype(np.float32)
        - CLIP_MEAN_255.reshape(1, 1, 3)
    ) / CLIP_STD_255.reshape(1, 1, 3)
    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))
    ).unsqueeze(0).to(device=device, dtype=torch.float32)

    from mmseg.structures import SegDataSample

    sample = SegDataSample()
    sample.set_metainfo(
        {
            "ori_shape": (original_height, original_width, 3),
            "img_shape": (resized_height, resized_width, 3),
            "pad_shape": (resized_height, resized_width, 3),
            "padding_size": [0, 0, 0, 0],
        }
    )
    output = model.predict(tensor, [sample])
    prediction = output[0].pred_sem_seg.data.squeeze(0).detach().cpu().numpy()
    prediction = np.asarray(prediction, dtype=np.int64)
    if prediction.shape != (original_height, original_width):
        raise RuntimeError(
            "SegEarth-OV failed to restore the native image extent: "
            f"expected={(original_height, original_width)}, got={prediction.shape}"
        )
    if prediction.size and (prediction.min() < 0 or prediction.max() >= num_classes):
        raise RuntimeError(
            f"SegEarth-OV produced class IDs outside 0..{num_classes - 1}: "
            f"range=({prediction.min()},{prediction.max()})"
        )
    h_grids = max(resized_height - slide_crop + slide_stride - 1, 0) // slide_stride + 1
    w_grids = max(resized_width - slide_crop + slide_stride - 1, 0) // slide_stride + 1
    return np.ascontiguousarray(prediction), {
        "model_input_height": int(resized_height),
        "model_input_width": int(resized_width),
        "num_slide_windows": int(h_grids * w_grids),
    }


def _validate_cached_prediction(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    num_classes: int,
    prediction_ignore_index: int | None = None,
) -> np.ndarray:
    with Image.open(path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.int64)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"Cached prediction has shape {prediction.shape}, expected {expected_shape}: {path}"
        )
    valid = (prediction >= 0) & (prediction < num_classes)
    if prediction_ignore_index is not None:
        valid |= prediction == int(prediction_ignore_index)
    if prediction.size and not bool(np.all(valid)):
        values = np.unique(prediction[~valid])[:8].tolist()
        raise ValueError(
            f"Cached prediction contains invalid class IDs {values}: {path}"
        )
    return prediction


def _validate_resume_protocol(output_root: Path, expected: dict[str, Any], overwrite: bool) -> None:
    predictions = list((output_root / "pred_mask").glob("*.png"))
    if overwrite or not predictions:
        return
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"{output_root} contains cached predictions without run_config.json. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"existing": existing.get(key), "expected": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing predictions use a different SegEarth-OV protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone SegEarth-OV inference on LoveDA, FLAIR #1, UAVid, "
            "xBD-pre, or CHN6-CUG with Vistar-compatible saved outputs."
        )
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--segearth_root",
        default=str(REPO_ROOT / "third_party" / "SegEarth-OV"),
    )
    parser.add_argument(
        "--simfeatup_root",
        default=str(REPO_ROOT / "third_party" / "SimFeatUp"),
    )
    parser.add_argument("--class_file", default="")
    parser.add_argument(
        "--simfeatup_checkpoint",
        default="",
    )
    parser.add_argument(
        "--clip_vitb",
        default="/root/data/weight/segearth_ov/pretrained/ViT-B-16.pt",
    )
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--slide_crop", type=int, default=224)
    parser.add_argument("--slide_stride", type=int, default=112)
    parser.add_argument("--probability_threshold", type=float, default=None)
    parser.add_argument("--cls_token_lambda", type=float, default=None)
    parser.add_argument(
        "--feature_up",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--mask_id_base",
        choices=("auto", "zero", "one"),
        default="auto",
        help="UAVid indexed-mask base; ignored by the other datasets",
    )
    parser.add_argument(
        "--strict_protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_unpinned_source",
        action="store_true",
        help="debug only: allow a SegEarth-OV checkout other than the pinned revision",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    if not args.data_root:
        args.data_root = DEFAULT_DATA_ROOTS[args.dataset]
    if not args.class_file:
        args.class_file = str(
            Path(__file__).resolve().parent
            / "configs"
            / DEFAULT_CLASS_FILES[args.dataset]
        )
    if not args.simfeatup_checkpoint:
        args.simfeatup_checkpoint = str(
            normalize_path(args.segearth_root)
            / "simfeatup_dev"
            / "weights"
            / "xclip_jbu_one_million_aid.ckpt"
        )
    if args.probability_threshold is None:
        args.probability_threshold = float(spec["probability_threshold"])
    if args.cls_token_lambda is None:
        args.cls_token_lambda = float(spec["cls_token_lambda"])
    if args.input_size <= 0 or args.slide_crop <= 0 or args.slide_stride <= 0:
        parser.error("--input_size, --slide_crop, and --slide_stride must be positive")
    if args.slide_crop > args.input_size:
        parser.error("--slide_crop cannot exceed --input_size")
    if args.max_samples < 0:
        parser.error("--max_samples cannot be negative")
    if not 0.0 <= args.probability_threshold <= 1.0:
        parser.error("--probability_threshold must be in [0,1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0,1]")
    if not torch.cuda.is_available():
        parser.error("Standalone SegEarth-OV evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    spec = DATASET_SPECS[args.dataset]
    class_file = normalize_path(args.class_file)
    class_groups = read_class_groups(class_file)
    validate_class_groups(args.dataset, class_groups)
    class_names = tuple(str(name) for name in spec["classes"])
    num_classes = len(class_names)
    palette = np.asarray(spec["palette"], dtype=np.uint8)
    ignore_index = spec["ignore_index"]

    source_root = normalize_path(args.segearth_root)
    simfeatup_root = normalize_path(args.simfeatup_root)
    simfeatup_checkpoint = normalize_path(args.simfeatup_checkpoint)
    clip_vitb = normalize_path(args.clip_vitb)
    required = {
        "official SegEarth-OV source": source_root / "segearth_segmentor.py",
        "official SegEarth-OV open_clip": source_root / "open_clip" / "factory.py",
        "class vocabulary": class_file,
        "managed CLIP ViT-B/16": clip_vitb,
    }
    if args.feature_up:
        required["SegEarth-OV SimFeatUp checkpoint"] = simfeatup_checkpoint
        required["official SimFeatUp source"] = simfeatup_root / "setup.py"
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing standalone SegEarth-OV source or weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_segearth_ov.sh and tools/check_segearth_ov_deps.py."
        )
    source_revision = _source_revision(source_root)
    simfeatup_revision = _source_revision(simfeatup_root)
    if (
        source_revision != SEGEARTH_OV_SOURCE_REVISION
        and not args.allow_unpinned_source
    ):
        raise RuntimeError(
            "Standalone paper evaluation requires the pinned official "
            f"SegEarth-OV revision {SEGEARTH_OV_SOURCE_REVISION}, got "
            f"{source_revision!r}. Run scripts/bootstrap_segearth_ov.sh, or "
            "pass --allow_unpinned_source only for an explicitly labeled debug run."
        )
    if (
        args.feature_up
        and simfeatup_revision != SIMFEATUP_SOURCE_REVISION
        and not args.allow_unpinned_source
    ):
        raise RuntimeError(
            "The official feature-up protocol requires pinned SimFeatUp "
            f"revision {SIMFEATUP_SOURCE_REVISION}, got "
            f"{simfeatup_revision!r}. Run scripts/bootstrap_segearth_ov.sh, "
            "or disable feature-up only for a non-paper debug run."
        )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    data_root = normalize_path(args.data_root)
    records, dataset_audit = discover_dataset(
        args.dataset,
        data_root,
        strict=bool(args.strict_protocol),
        mask_id_base=str(args.mask_id_base),
    )
    indexed_records = list(enumerate(records))
    if args.max_samples > 0:
        indexed_records = indexed_records[: args.max_samples]
    if not indexed_records:
        raise RuntimeError("No evaluation samples were selected")
    local_records = indexed_records[rank::world_size]

    output_root = normalize_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    protocol_identity = {
        "dataset_key": args.dataset,
        "method": "SegEarth-OV",
        "source_revision": source_revision,
        "class_groups": [list(group) for group in class_groups],
        "input_size": int(args.input_size),
        "resize": "MMSeg keep_ratio bilinear",
        "slide_crop": int(args.slide_crop),
        "slide_stride": int(args.slide_stride),
        "probability_threshold": float(args.probability_threshold),
        "cls_token_lambda": float(args.cls_token_lambda),
        "feature_up": bool(args.feature_up),
        "simfeatup_source_revision": simfeatup_revision,
        "simfeatup_checkpoint": str(simfeatup_checkpoint),
        "clip_vitb": str(clip_vitb),
        "metric_size": "original",
    }
    if args.dataset == "uavid":
        protocol_identity.update(
            {
                "label_space": "VISTAR/OVSISBench UAVid eight-class",
                "mask_id_base": dataset_audit["mask_encoding"][
                    "resolved_mask_id_base"
                ],
                "population": "all 270 one-to-one pairs under Images/Labels",
            }
        )
    _validate_resume_protocol(output_root, protocol_identity, args.overwrite)

    if rank == 0:
        write_json(
            output_root / "class_map.json",
            {
                "dataset": spec["display_name"],
                "classes": [
                    {
                        "id": class_id,
                        "name": class_name,
                        "text_aliases": list(class_groups[class_id]),
                        "rgb": palette[class_id].tolist(),
                    }
                    for class_id, class_name in enumerate(class_names)
                ],
                "ignore_index": ignore_index,
                "label_protocol": spec["label_protocol"],
            },
        )
        write_json(
            output_root / "run_config.json",
            {
                **protocol_identity,
                "dataset": spec["display_name"],
                "split": spec["split"],
                "training_setting": "training-free",
                "clip_type": "CLIP",
                "vit_type": "ViT-B/16",
                "model_type": "SegEarth",
                "ignore_residual": True,
                "logit_scale": 50,
                "background_index": 0,
                "num_classes": num_classes,
                "class_file": str(class_file),
                "segearth_root": str(source_root),
                "expected_source_revision": SEGEARTH_OV_SOURCE_REVISION,
                "source_is_pinned": source_revision == SEGEARTH_OV_SOURCE_REVISION,
                "allow_unpinned_source": bool(args.allow_unpinned_source),
                "simfeatup_root": str(simfeatup_root),
                "expected_simfeatup_source_revision": SIMFEATUP_SOURCE_REVISION,
                "simfeatup_source_is_pinned": (
                    not args.feature_up
                    or simfeatup_revision == SIMFEATUP_SOURCE_REVISION
                ),
                "data_root": str(data_root),
                "requested_mask_id_base": str(args.mask_id_base),
                "strict_protocol": bool(args.strict_protocol),
                "num_discovered_samples": len(records),
                "num_selected_samples": len(indexed_records),
                "world_size": world_size,
                "save_images": bool(args.save_images),
                "primary_metric": spec["primary_metric"],
                "dataset_audit": dataset_audit,
                "flair_protocol_note": (
                    "FLAIR uses the shared GSNet/RSKT-Seg 12-class protocol; "
                    "the original SegEarth-OV repository has no FLAIR config."
                    if args.dataset == "flair"
                    else None
                ),
                "uavid_protocol_note": (
                    "Repository-local VISTAR-compatible eight-class adaptation; "
                    "moving car and static car remain separate. This is not the "
                    "official SegEarth-OV seven-class merged-car protocol."
                    if args.dataset == "uavid"
                    else None
                ),
            },
        )

    model = _build_model(
        args,
        device=device,
        probability_threshold=float(args.probability_threshold),
        cls_token_lambda=float(args.cls_token_lambda),
    )
    local_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    local_domain_confusions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        local_records,
        desc=f"SegEarth-OV {spec['display_name']} rank {rank}",
        disable=False,
    )
    for global_index, record in progress:
        image_rgb = load_rgb(record, args.dataset)
        height, width = image_rgb.shape[:2]
        target = load_target(record, args.dataset, height=height, width=width)
        pred_path = output_dirs["pred_mask"] / f"{record.name}.png"
        if pred_path.is_file() and not args.overwrite:
            prediction = _validate_cached_prediction(
                pred_path,
                expected_shape=(height, width),
                num_classes=num_classes,
            )
            resized = resize_keep_ratio(image_rgb, args.input_size)
            resized_height, resized_width = resized.shape[:2]
            h_grids = max(
                resized_height - args.slide_crop + args.slide_stride - 1,
                0,
            ) // args.slide_stride + 1
            w_grids = max(
                resized_width - args.slide_crop + args.slide_stride - 1,
                0,
            ) // args.slide_stride + 1
            inference_meta = {
                "model_input_height": resized_height,
                "model_input_width": resized_width,
                "num_slide_windows": int(h_grids * w_grids),
                "resumed_from_saved_prediction": True,
            }
        else:
            prediction, inference_meta = _predict(
                model,
                image_rgb,
                device=device,
                input_size=args.input_size,
                slide_crop=args.slide_crop,
                slide_stride=args.slide_stride,
                num_classes=num_classes,
            )
            _save_id_mask(prediction, pred_path)
            inference_meta["resumed_from_saved_prediction"] = False

        if args.save_images:
            _save_visualizations(
                output_dirs,
                record.name,
                image_rgb,
                prediction,
                target,
                palette,
                ignore_index=ignore_index,
                overlay_alpha=args.overlay_alpha,
            )
        matrix = confusion_matrix(
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        local_confusion += matrix
        if record.domain:
            local_domain_confusions.setdefault(
                record.domain,
                np.zeros_like(local_confusion),
            )
            local_domain_confusions[record.domain] += matrix
        if ignore_index is None:
            boundary_metrics = score_binary_boundary_wfm(prediction, target)
        else:
            boundary_metrics = score_semantic_boundary_wfm(
                prediction,
                target,
                ignore_index=int(ignore_index),
                ignore_margin=2,
                num_classes=num_classes,
            )
        rows.append(
            {
                "global_index": int(global_index),
                "name": record.name,
                "domain": record.domain,
                "zone": record.zone,
                "sample_id": record.sample_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "prediction_path": str(pred_path),
                "height": int(height),
                "width": int(width),
                **inference_meta,
                **metrics_for_dataset(matrix, args.dataset),
                **boundary_metrics,
            }
        )

    write_json(
        output_root / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "confusion_matrix": local_confusion.tolist(),
            "domain_confusions": {
                domain: matrix.tolist()
                for domain, matrix in local_domain_confusions.items()
            },
            "rows": rows,
        },
    )
    _barrier(world_size)

    if rank == 0:
        merged_confusion = np.zeros_like(local_confusion)
        merged_domain_confusions: dict[str, np.ndarray] = {}
        merged_rows: list[dict[str, Any]] = []
        for process_rank in range(world_size):
            rank_path = output_root / f"rank_{process_rank:05d}.json"
            payload = json.loads(rank_path.read_text(encoding="utf-8"))
            merged_confusion += np.asarray(payload["confusion_matrix"], dtype=np.int64)
            for domain, values in payload["domain_confusions"].items():
                merged_domain_confusions.setdefault(
                    domain,
                    np.zeros_like(merged_confusion),
                )
                merged_domain_confusions[domain] += np.asarray(values, dtype=np.int64)
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        if len(merged_rows) != len(indexed_records):
            raise RuntimeError(
                "Distributed result count mismatch: "
                f"expected {len(indexed_records)}, found {len(merged_rows)}"
            )
        expected_num_samples = int(dataset_audit["expected_num_samples"])
        result = {
            "dataset": spec["display_name"],
            "dataset_key": args.dataset,
            "split": spec["split"],
            "method": "SegEarth-OV",
            "training_setting": "training-free",
            "source_revision": source_revision,
            "expected_source_revision": SEGEARTH_OV_SOURCE_REVISION,
            "source_is_pinned": source_revision == SEGEARTH_OV_SOURCE_REVISION,
            "simfeatup_source_revision": simfeatup_revision,
            "expected_simfeatup_source_revision": SIMFEATUP_SOURCE_REVISION,
            "num_samples": len(merged_rows),
            "expected_num_samples": expected_num_samples,
            "complete_protocol_population": len(merged_rows) == expected_num_samples,
            "num_classes": num_classes,
            "classes": list(class_names),
            "class_groups": [list(group) for group in class_groups],
            "label_protocol": spec["label_protocol"],
            "inference_mode": "official_keep_ratio_resize_then_overlapping_slide",
            "input_size": int(args.input_size),
            "slide_crop": int(args.slide_crop),
            "slide_stride": int(args.slide_stride),
            "metric_size": "original",
            "num_slide_windows": sum(int(row["num_slide_windows"]) for row in merged_rows),
            "feature_up": bool(args.feature_up),
            "probability_threshold": float(args.probability_threshold),
            "cls_token_lambda": float(args.cls_token_lambda),
            "logit_scale": 50,
            "test_time_augmentation": "none",
            "world_size": world_size,
            "save_pred_mask": True,
            "save_visualizations": bool(args.save_images),
            "primary_metric": spec["primary_metric"],
            "confusion_matrix": merged_confusion.tolist(),
            "domain_metrics": {
                domain: {
                    "num_samples": sum(1 for row in merged_rows if row["domain"] == domain),
                    "confusion_matrix": matrix.tolist(),
                    **metrics_for_dataset(matrix, args.dataset),
                }
                for domain, matrix in sorted(merged_domain_confusions.items())
            },
            "dataset_audit": dataset_audit,
            **metrics_for_dataset(merged_confusion, args.dataset),
            **aggregate_binary_boundary_wfm(merged_rows),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_csv(output_root / "per_image_metrics.csv", merged_rows)
        write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[eval_segearth_ov] saved outputs to: {output_root}")

    _barrier(world_size)
    del model
    torch.cuda.empty_cache()
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
