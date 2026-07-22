from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


BASELINES_DIR = Path(__file__).resolve().parents[1]
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)
from flair_protocol import (  # noqa: E402
    FLAIR1_EXPECTED_SAMPLES,
    FLAIR1_EXPECTED_ZONES,
    FLAIR1_TEST_DOMAIN_COUNTS,
    FLAIR_GSNET_CLASSES,
    FLAIR_GSNET_MODEL_CLASSES,
    FLAIR_IGNORE_VISUALIZATION_RGB,
    FLAIR_VISUAL_PALETTE_U8,
    IGNORE_INDEX,
    discover_flair1_test,
    flair_confusion_matrix,
    flair_metrics_from_confusion,
    load_flair_mask_array,
    load_flair_rgb_u8,
)


Image.MAX_IMAGE_PIXELS = None

METHOD = "GSNet"
TRAINING_DATASET = "LandDiscover50K"
PROTOCOL = "GSNet_FLAIR_12class"
INFERENCE_MODE = "official_gsnet_flair_sliding_window"
NATIVE_PATCH_SIZE = 512
OFFICIAL_TEST_RESIZE = 640
OFFICIAL_LOCAL_WINDOW = 384
IGNORE_VISUALIZATION_RGB = np.asarray(
    FLAIR_IGNORE_VISUALIZATION_RGB,
    dtype=np.uint8,
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model_classes(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"GSNet FLAIR class JSON must be a non-empty string list: {path}"
        )
    classes = [item.strip() for item in value]
    expected = list(FLAIR_GSNET_MODEL_CLASSES)
    if classes != expected:
        raise ValueError(
            "GSNet FLAIR text classes must exactly match the official "
            f"datasets/flair.json order. expected={expected}, got={classes}"
        )
    return classes


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # Models are independent across ranks; Gloo synchronizes metadata only.
        torch.distributed.init_process_group(backend="gloo")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


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
    if str(cfg.INPUT.FORMAT).upper() != "RGB":
        raise ValueError(
            "The FLAIR loader returns RGB bands 1--3, but the selected "
            f"GSNet config requests INPUT.FORMAT={cfg.INPUT.FORMAT!r}."
        )
    # Match the released GSNet FLAIR evaluation: a native 512 patch is first
    # resized to 640, then the model uses 384 windows plus a 384 global view.
    cfg.INPUT.MIN_SIZE_TEST = OFFICIAL_TEST_RESIZE
    cfg.INPUT.MAX_SIZE_TEST = OFFICIAL_TEST_RESIZE
    cfg.TEST.SLIDING_WINDOW = True
    cfg.freeze()

    original_download = official_clip._download

    def use_managed_clip(url: str, root: str | None = None) -> str:
        if Path(url).name == "ViT-B-16.pt":
            return str(clip_path)
        if root is None:
            return original_download(url)
        return original_download(url, root)

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


def _predict(
    *,
    model: torch.nn.Module,
    image_rgb_u8: np.ndarray,
    image_path: Path,
    amp: str,
    num_classes: int,
) -> np.ndarray:
    from detectron2.data import transforms as transforms

    if image_rgb_u8.shape != (NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE, 3):
        raise ValueError(
            "Official FLAIR #1 RGB patches must be 512x512: "
            f"{image_path} has shape={image_rgb_u8.shape}"
        )
    resize = transforms.ResizeShortestEdge(
        [OFFICIAL_TEST_RESIZE, OFFICIAL_TEST_RESIZE],
        OFFICIAL_TEST_RESIZE,
    )
    transformed = resize.get_transform(image_rgb_u8).apply_image(image_rgb_u8)
    if transformed.shape[:2] != (OFFICIAL_TEST_RESIZE, OFFICIAL_TEST_RESIZE):
        raise RuntimeError(
            "Unexpected GSNet FLAIR test resize: "
            f"{transformed.shape[:2]}"
        )
    tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)
    )
    model_input = {
        "image": tensor,
        "height": NATIVE_PATCH_SIZE,
        "width": NATIVE_PATCH_SIZE,
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
    if output.ndim != 3 or output.shape[0] != num_classes:
        raise RuntimeError(
            "GSNet output-channel count does not match the official FLAIR "
            f"taxonomy: shape={tuple(output.shape)}, classes={num_classes}"
        )
    prediction = output.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
    if prediction.shape != (NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE):
        raise RuntimeError(
            "GSNet must restore each FLAIR prediction to its native 512x512 "
            f"extent, got={prediction.shape}"
        )
    return prediction


def _make_output_dirs(root: Path) -> dict[str, Path]:
    directories = {
        name: root / name
        for name in ("pred_mask", "gt_mask", "pred_rgb", "gt_rgb")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _colorize(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.int64)
    palette = np.asarray(FLAIR_VISUAL_PALETTE_U8, dtype=np.uint8)
    output = np.empty((*values.shape, 3), dtype=np.uint8)
    output[...] = IGNORE_VISUALIZATION_RGB
    valid = (values >= 0) & (values < len(palette))
    output[valid] = palette[values[valid]]
    return output


def _save_masks(
    directories: dict[str, Path],
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    Image.fromarray(prediction.astype(np.uint8), mode="L").save(
        directories["pred_mask"] / f"{name}.png"
    )
    Image.fromarray(target.astype(np.uint8), mode="L").save(
        directories["gt_mask"] / f"{name}.png"
    )
    Image.fromarray(_colorize(prediction), mode="RGB").save(
        directories["pred_rgb"] / f"{name}.png"
    )
    Image.fromarray(_colorize(target), mode="RGB").save(
        directories["gt_rgb"] / f"{name}.png"
    )


def _validate_prediction(
    prediction: np.ndarray,
    *,
    num_classes: int,
    path: Path,
) -> np.ndarray:
    values = np.asarray(prediction)
    if values.shape != (NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE):
        raise ValueError(f"Invalid cached prediction shape at {path}: {values.shape}")
    invalid = (values < 0) | (values >= num_classes)
    if bool(np.any(invalid)):
        raise ValueError(
            f"Invalid cached class IDs at {path}: "
            f"{np.unique(values[invalid]).tolist()}"
        )
    return values.astype(np.uint8, copy=False)


def _validate_resume_protocol(
    output_root: Path,
    *,
    expected: dict[str, Any],
    overwrite: bool,
) -> None:
    cached = list((output_root / "pred_mask").glob("*.png"))
    if overwrite or not cached:
        return
    path = output_root / "run_config.json"
    if not path.is_file():
        raise RuntimeError(
            f"{output_root} contains cached predictions without run_config.json. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )
    existing = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"existing": existing.get(key), "expected": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Cached predictions use a different GSNet FLAIR protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_source = repo_root / "third_party" / "GSNet"
    default_weight_root = Path("/root/data/weight/gsnet")
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the released LandDiscover50K GSNet checkpoint on the "
            "complete official FLAIR #1 test set using its 12-class protocol."
        )
    )
    parser.add_argument(
        "--data_root",
        default="/root/data/FLAIR-1-2/data/flair#1-test",
    )
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
    parser.add_argument(
        "--class_json",
        default=str(
            Path(__file__).resolve().parent
            / "configs"
            / "flair_12_classes.json"
        ),
    )
    parser.add_argument(
        "--clip_vitb",
        default=str(default_weight_root / "pretrained" / "ViT-B-16.pt"),
    )
    parser.add_argument(
        "--rsib",
        default=os.environ.get("GSNET_RSIB", "/root/data/weight/rsib/RSIB.pth"),
    )
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
    parser.add_argument("--no_strict_protocol", action="store_true")
    args = parser.parse_args()
    if args.num_layers <= 0:
        parser.error("--num_layers must be positive")
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if not torch.cuda.is_available():
        parser.error("GSNet FLAIR evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    class_json = _path(args.class_json)
    model_classes = _load_model_classes(class_json)
    num_classes = len(model_classes)
    required = {
        "official source": _path(args.gsnet_root) / "gs_net" / "GSNet.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        "class JSON": class_json,
        "CLIP ViT-B/16": _path(args.clip_vitb),
        "RSIB/DINO": _path(args.rsib),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing GSNet source or weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_gsnet.sh and tools/check_gsnet_deps.py."
        )

    strict_protocol = not bool(args.no_strict_protocol)
    records, dataset_audit = discover_flair1_test(
        args.data_root,
        strict=strict_protocol,
    )
    indexed_records = list(enumerate(records))
    if args.max_samples:
        indexed_records = indexed_records[: args.max_samples]
    comparable_full_test = bool(
        strict_protocol and len(indexed_records) == FLAIR1_EXPECTED_SAMPLES
    )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    local_records = indexed_records[rank::world_size]

    resume_protocol = {
        "dataset": "FLAIR#1",
        "split": "flair#1-test",
        "data_root": str(_path(args.data_root)),
        "method": METHOD,
        "training_dataset": TRAINING_DATASET,
        "protocol": PROTOCOL,
        "selected_num_samples": len(indexed_records),
        "strict_protocol": strict_protocol,
        "inference_mode": INFERENCE_MODE,
        "native_patch_size": NATIVE_PATCH_SIZE,
        "model_input_size": OFFICIAL_TEST_RESIZE,
        "local_window_size": OFFICIAL_LOCAL_WINDOW,
        "test_classes": model_classes,
        "checkpoint": str(_path(args.checkpoint)),
        "config": str(_path(args.config)),
        "class_json": str(class_json),
        "clip_vitb": str(_path(args.clip_vitb)),
        "rsib": str(_path(args.rsib)),
        "prompt_ensemble": args.prompt_ensemble,
        "num_layers": int(args.num_layers),
        "amp": args.amp,
    }
    _validate_resume_protocol(
        output_root,
        expected=resume_protocol,
        overwrite=args.overwrite,
    )

    _cfg, model, load_result = _configure_model(
        args,
        local_rank,
        num_test_classes=num_classes,
    )
    if rank == 0:
        _write_json(
            output_root / "dataset_audit.json",
            {
                **dataset_audit,
                "protocol": PROTOCOL,
                "model_text_classes": model_classes,
                "evaluation_classes": list(FLAIR_GSNET_CLASSES),
                "selected_rgb_band_indices_one_based": [1, 2, 3],
            },
        )
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": "FLAIR#1",
                "split": "flair#1-test",
                "protocol": PROTOCOL,
                "classes": [
                    {
                        "id": index,
                        "name": FLAIR_GSNET_CLASSES[index],
                        "model_text": model_classes[index],
                        "rgb": np.asarray(FLAIR_VISUAL_PALETTE_U8)[index].tolist(),
                    }
                    for index in range(num_classes)
                ],
                "ignore_index": IGNORE_INDEX,
                "ignore_visualization_rgb": IGNORE_VISUALIZATION_RGB.tolist(),
                "raw_to_eval_mapping": {
                    str(raw_id): raw_id - 1 for raw_id in range(1, 13)
                },
                "ignored_raw_ids": [0, 13, 14, 15, 16, 17, 18, 19, 255],
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                **resume_protocol,
                "evaluation_setting": "cross-dataset/out-of-domain",
                "paper_table_ovrsisbenchv2_comparable": False,
                "num_samples": len(indexed_records),
                "expected_num_samples": FLAIR1_EXPECTED_SAMPLES,
                "expected_num_zones": FLAIR1_EXPECTED_ZONES,
                "strict_protocol": strict_protocol,
                "comparable_full_test": comparable_full_test,
                "source_bands": ["red", "green", "blue", "NIR", "nDSM"],
                "selected_rgb_band_indices_one_based": [1, 2, 3],
                "test_resize": "shortest_edge_640",
                "sliding_window": True,
                "sliding_window_overlap": 1.0 / 3.0,
                "global_view_size": OFFICIAL_LOCAL_WINDOW,
                "pooling_sizes": [1, 1],
                "world_size": world_size,
                "save_images": bool(args.save_images),
                "load_result": str(load_result),
            },
        )
        print(
            f"[eval_gsnet_flair] samples={len(indexed_records)} "
            f"strict={strict_protocol} comparable_full_test={comparable_full_test}"
        )
        print(
            "[eval_gsnet_flair] preprocessing=native RGB bands 1-3 -> "
            "resize 640 -> 384 local windows + 384 global view -> native 512"
        )
    _barrier(world_size)

    rows: list[dict[str, Any]] = []
    local_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    local_domain_confusions = {
        domain: np.zeros((num_classes, num_classes), dtype=np.int64)
        for domain in FLAIR1_TEST_DOMAIN_COUNTS
    }
    progress = tqdm(
        local_records,
        desc=f"GSNet FLAIR rank {rank}",
        disable=False,
    )
    for global_index, record in progress:
        name = record.output_name
        pred_path = output_dirs["pred_mask"] / f"{name}.png"
        target = load_flair_mask_array(record.mask_path, ignore_index=IGNORE_INDEX)
        if target.shape != (NATIVE_PATCH_SIZE, NATIVE_PATCH_SIZE):
            raise ValueError(
                f"Invalid FLAIR target shape for {record.mask_path}: {target.shape}"
            )
        if pred_path.is_file() and not args.overwrite:
            with Image.open(pred_path) as image:
                prediction = _validate_prediction(
                    np.asarray(image.convert("L")),
                    num_classes=num_classes,
                    path=pred_path,
                )
        else:
            image_rgb_u8 = load_flair_rgb_u8(record.image_path)
            prediction = _predict(
                model=model,
                image_rgb_u8=image_rgb_u8,
                image_path=record.image_path,
                amp=args.amp,
                num_classes=num_classes,
            )
            # The class-ID map is the restart cache and is therefore always
            # saved, even when visualization output is disabled.
            Image.fromarray(prediction, mode="L").save(pred_path)

        if args.save_images:
            _save_masks(output_dirs, name, prediction, target)

        confusion = flair_confusion_matrix(
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
        )
        local_confusion += confusion
        local_domain_confusions[record.domain] += confusion
        image_metrics = flair_metrics_from_confusion(confusion)
        image_wfm = score_semantic_boundary_wfm(
            prediction,
            target,
            ignore_index=IGNORE_INDEX,
            ignore_margin=2,
            num_classes=num_classes,
        )
        rows.append(
            {
                "global_index": int(global_index),
                "name": name,
                "domain": record.domain,
                "zone": record.zone,
                "sample_id": record.sample_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "prediction_path": str(pred_path),
                "height": NATIVE_PATCH_SIZE,
                "width": NATIVE_PATCH_SIZE,
                **image_metrics,
                **image_wfm,
            }
        )

    _write_json(
        output_root / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "confusion": local_confusion.tolist(),
            "domain_confusions": {
                domain: matrix.tolist()
                for domain, matrix in local_domain_confusions.items()
            },
            "rows": rows,
        },
    )
    _barrier(world_size)

    if rank == 0:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        domain_confusions = {
            domain: np.zeros((num_classes, num_classes), dtype=np.int64)
            for domain in FLAIR1_TEST_DOMAIN_COUNTS
        }
        merged_rows: list[dict[str, Any]] = []
        for process_rank in range(world_size):
            rank_path = output_root / f"rank_{process_rank:05d}.json"
            payload = json.loads(rank_path.read_text(encoding="utf-8"))
            confusion += np.asarray(payload["confusion"], dtype=np.int64)
            for domain in FLAIR1_TEST_DOMAIN_COUNTS:
                domain_confusions[domain] += np.asarray(
                    payload["domain_confusions"][domain],
                    dtype=np.int64,
                )
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        if len(merged_rows) != len(indexed_records):
            raise RuntimeError(
                f"Distributed result count mismatch: expected {len(indexed_records)}, "
                f"found {len(merged_rows)}"
            )
        global_metrics = flair_metrics_from_confusion(confusion)
        summary = {
            "dataset": "FLAIR#1",
            "split": "flair#1-test",
            "method": METHOD,
            "training_dataset": TRAINING_DATASET,
            "evaluation_setting": "cross-dataset/out-of-domain",
            "paper_table_ovrsisbenchv2_comparable": False,
            "protocol": PROTOCOL,
            "checkpoint": str(_path(args.checkpoint)),
            "num_samples": len(merged_rows),
            "expected_num_samples": FLAIR1_EXPECTED_SAMPLES,
            "expected_num_zones": FLAIR1_EXPECTED_ZONES,
            "comparable_full_test": comparable_full_test,
            "official_test_domains": list(FLAIR1_TEST_DOMAIN_COUNTS),
            "num_classes": num_classes,
            "classes": list(FLAIR_GSNET_CLASSES),
            "model_text_classes": model_classes,
            "ignored_raw_ids": [0, 13, 14, 15, 16, 17, 18, 19, 255],
            "inference_mode": INFERENCE_MODE,
            "native_patch_size": NATIVE_PATCH_SIZE,
            "model_input_size": OFFICIAL_TEST_RESIZE,
            "local_window_size": OFFICIAL_LOCAL_WINDOW,
            "global_view_size": OFFICIAL_LOCAL_WINDOW,
            "sliding_window_overlap": 1.0 / 3.0,
            "num_tiles": len(merged_rows),
            "num_native_patches": len(merged_rows),
            "local_windows_per_patch": 4,
            "num_local_windows": 4 * len(merged_rows),
            "num_global_views": len(merged_rows),
            "total_pixels": len(merged_rows)
            * NATIVE_PATCH_SIZE
            * NATIVE_PATCH_SIZE,
            "ignored_pixels": len(merged_rows)
            * NATIVE_PATCH_SIZE
            * NATIVE_PATCH_SIZE
            - int(confusion.sum()),
            "world_size": world_size,
            "amp": args.amp,
            "prompt_ensemble": args.prompt_ensemble,
            "num_layers": args.num_layers,
            "test_time_augmentation": "none",
            "source_band_indices_one_based": [1, 2, 3],
            "save_pred_mask": True,
            "save_gt_mask": bool(args.save_images),
            "save_pred_rgb": bool(args.save_images),
            "save_gt_rgb": bool(args.save_images),
            "confusion_matrix": confusion.tolist(),
            "domain_metrics": {
                domain: {
                    "num_samples": sum(
                        1 for row in merged_rows if row["domain"] == domain
                    ),
                    "confusion_matrix": domain_confusions[domain].tolist(),
                    **flair_metrics_from_confusion(domain_confusions[domain]),
                }
                for domain in FLAIR1_TEST_DOMAIN_COUNTS
            },
            **global_metrics,
            **aggregate_binary_boundary_wfm(merged_rows),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_csv(output_root / "per_image_metrics.csv", merged_rows)
        _write_json(output_root / "metrics.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[eval_gsnet_flair] saved outputs to: {output_root}")

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
