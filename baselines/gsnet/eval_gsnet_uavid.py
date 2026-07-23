"""Standalone GSNet evaluation on VISTAR's eight-class UAVid protocol."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "baselines"
for import_root in (REPO_ROOT, BASELINES_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)
from baselines.gsnet.eval_gsnet_binary import (  # noqa: E402
    _barrier,
    _ceil_to_multiple,
    _configure_model,
    _distributed_context,
    _path,
    _tile_coords,
)
from baselines.gsnet.gsnet_uavid_protocol import (  # noqa: E402
    UAVID_AMP,
    UAVID_EXPECTED_SAMPLES,
    UAVID_MODEL_CLASSES,
    UAVID_MODEL_INPUT_SIZE,
    UAVID_NUM_LAYERS,
    UAVID_PROMPT_ENSEMBLE,
    UAVID_TILE_SIZE,
    strict_protocol_errors,
    tile_grid_count,
    validate_uavid_model_classes,
)
from baselines.segearth_ov.protocols import (  # noqa: E402
    DATASET_SPECS,
    IGNORE_INDEX,
    colorize_mask,
    confusion_matrix,
    discover_uavid,
    load_rgb,
    load_target,
    metrics_for_dataset,
)


Image.MAX_IMAGE_PIXELS = None

METHOD = "GSNet"
TRAINING_DATASET = "LandDiscover50K"
PROTOCOL = "GSNet_UAVid_VISTAR_8class"
INFERENCE_MODE = "native_nonoverlap_512_tiled"
EVALUATION_SETTING = "cross-dataset/out-of-domain"


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    _write_text_atomic(path, text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def _save_id_mask(mask: np.ndarray, path: Path) -> None:
    _save_png_atomic(
        Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L"),
        path,
    )


def _make_output_dirs(output_root: Path) -> dict[str, Path]:
    directories = {
        name: output_root / name
        for name in ("pred_mask", "gt_mask", "pred_rgb", "gt_rgb")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _load_model_classes(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"GSNet UAVid class JSON must be a non-empty string list: {path}"
        )
    classes = [item.strip() for item in value]
    validate_uavid_model_classes(classes)
    return classes


def _predict_tiled(
    *,
    model: torch.nn.Module,
    cfg: Any,
    image_rgb: np.ndarray,
    image_path: Path,
    amp: str,
    tile_size: int,
    num_classes: int,
) -> tuple[np.ndarray, int]:
    """Run square non-overlapping tiles and restore the native image extent."""

    from detectron2.data import transforms as transforms

    if str(cfg.INPUT.FORMAT).upper() != "RGB":
        raise ValueError(
            "The UAVid loader returns RGB images, but GSNet requests "
            f"INPUT.FORMAT={cfg.INPUT.FORMAT!r}."
        )
    original_height, original_width = image_rgb.shape[:2]
    padded_height = _ceil_to_multiple(original_height, tile_size)
    padded_width = _ceil_to_multiple(original_width, tile_size)
    padded_image = np.zeros(
        (padded_height, padded_width, 3),
        dtype=np.uint8,
    )
    padded_image[:original_height, :original_width] = image_rgb
    padded_prediction = np.zeros(
        (padded_height, padded_width),
        dtype=np.uint8,
    )
    coordinates = _tile_coords(padded_height, padded_width, tile_size)
    expected_tiles = tile_grid_count(
        original_height,
        original_width,
        tile_size,
    )
    if len(coordinates) != expected_tiles:
        raise RuntimeError(
            f"UAVid tile-grid mismatch for {image_path}: "
            f"coordinates={len(coordinates)}, expected={expected_tiles}"
        )

    resize = transforms.ResizeShortestEdge(
        [int(cfg.INPUT.MIN_SIZE_TEST), int(cfg.INPUT.MIN_SIZE_TEST)],
        int(cfg.INPUT.MAX_SIZE_TEST),
    )
    autocast_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[amp]
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
            dtype=autocast_dtype,
            enabled=amp != "fp32",
        ):
            output = model([model_input])[0]["sem_seg"]
        if output.ndim != 3 or output.shape[0] != num_classes:
            raise RuntimeError(
                "GSNet output-channel count does not match the VISTAR UAVid "
                f"taxonomy: shape={tuple(output.shape)}, classes={num_classes}"
            )
        prediction = output.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
        if prediction.shape != (tile_size, tile_size):
            prediction = np.asarray(
                Image.fromarray(prediction, mode="L").resize(
                    (tile_size, tile_size),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        padded_prediction[
            y : y + tile_size,
            x : x + tile_size,
        ] = prediction

    return (
        padded_prediction[:original_height, :original_width].copy(),
        len(coordinates),
    )


def _load_cached_prediction(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    with Image.open(path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.uint8)
    if prediction.shape != expected_shape:
        raise ValueError(
            f"Cached UAVid prediction has shape {prediction.shape}, expected "
            f"{expected_shape}: {path}"
        )
    invalid = prediction >= num_classes
    if bool(np.any(invalid)):
        raise ValueError(
            f"Cached UAVid prediction contains invalid class IDs "
            f"{np.unique(prediction[invalid]).tolist()}: {path}"
        )
    return np.ascontiguousarray(prediction)


def _validate_resume_protocol(
    output_root: Path,
    *,
    expected: dict[str, Any],
    overwrite: bool,
) -> None:
    cached = list((output_root / "pred_mask").glob("*.png"))
    if overwrite or not cached:
        return
    config_path = output_root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"{output_root} contains cached predictions without "
            "run_config.json. Use OVERWRITE=1 or a new OUTPUT_DIR."
        )
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"existing": existing.get(key), "expected": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Cached predictions use a different GSNet UAVid protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _validate_cache_inventory(
    output_root: Path,
    expected_names: list[str],
) -> dict[str, Any]:
    expected = {f"{name}.png" for name in expected_names}
    actual = {
        path.name
        for path in (output_root / "pred_mask").glob("*.png")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "expected_predictions": len(expected),
        "actual_predictions": len(actual),
        "missing_predictions": missing,
        "extra_predictions": extra,
        "complete": not missing and not extra,
    }


def _parse_args() -> argparse.Namespace:
    default_source = REPO_ROOT / "third_party" / "GSNet"
    default_weight_root = Path("/root/data/weight/gsnet")
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the released LandDiscover50K GSNet checkpoint on the "
            "complete VISTAR UAVid eight-class protocol."
        )
    )
    parser.add_argument(
        "--data_root",
        default="/root/data/OVSISBenchDataset/uavid",
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
            / "uavid_8_classes.json"
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
    parser.add_argument("--input_size", type=int, default=UAVID_MODEL_INPUT_SIZE)
    parser.add_argument("--tile_size", type=int, default=UAVID_TILE_SIZE)
    parser.add_argument("--num_layers", type=int, default=UAVID_NUM_LAYERS)
    parser.add_argument(
        "--prompt_ensemble",
        choices=("single", "imagenet", "imagenet_select"),
        default=UAVID_PROMPT_ENSEMBLE,
    )
    parser.add_argument(
        "--amp",
        choices=("fp32", "fp16", "bf16"),
        default=UAVID_AMP,
    )
    parser.add_argument(
        "--mask_id_base",
        choices=("auto", "zero", "one"),
        default="auto",
    )
    parser.add_argument(
        "--expected_samples",
        type=int,
        default=UAVID_EXPECTED_SAMPLES,
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--save_images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compute_wfm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_strict_protocol", action="store_true")
    args = parser.parse_args()
    if args.input_size <= 0 or args.tile_size <= 0:
        parser.error("--input_size and --tile_size must be positive")
    if args.num_layers <= 0:
        parser.error("--num_layers must be positive")
    if args.expected_samples <= 0:
        parser.error("--expected_samples must be positive")
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if not torch.cuda.is_available():
        parser.error("GSNet UAVid evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    class_json = _path(args.class_json)
    model_classes = _load_model_classes(class_json)
    num_classes = len(model_classes)
    spec = DATASET_SPECS["uavid"]
    evaluation_classes = tuple(str(value) for value in spec["classes"])
    if tuple(model_classes) != evaluation_classes:
        raise ValueError(
            "GSNet model text classes and VISTAR UAVid evaluation classes "
            f"must match exactly: {model_classes} vs {evaluation_classes}"
        )
    required = {
        "official source": _path(args.gsnet_root) / "gs_net" / "GSNet.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        "class JSON": class_json,
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
            + "\nRun scripts/bootstrap_gsnet.sh and tools/check_gsnet_deps.py."
        )

    strict = not bool(args.no_strict_protocol)
    records, dataset_audit = discover_uavid(
        _path(args.data_root),
        mask_id_base=args.mask_id_base,
    )
    selected_records = (
        records[: args.max_samples] if args.max_samples else records
    )
    if strict:
        errors = strict_protocol_errors(
            args,
            full_count=len(records),
            selected_count=len(selected_records),
        )
        if errors:
            raise RuntimeError(
                "Invalid strict GSNet UAVid protocol: " + "; ".join(errors)
            )
    complete_population = (
        len(records) == len(selected_records) == UAVID_EXPECTED_SAMPLES
    )
    complete_vistar_protocol = bool(
        strict
        and complete_population
        and args.tile_size == UAVID_TILE_SIZE
        and args.input_size == UAVID_MODEL_INPUT_SIZE
        and args.num_layers == UAVID_NUM_LAYERS
        and args.prompt_ensemble == UAVID_PROMPT_ENSEMBLE
        and args.amp == UAVID_AMP
    )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    sample_names = [record.name for record in selected_records]
    resume_protocol = {
        "dataset": "UAVid",
        "dataset_key": "uavid",
        "split": str(spec["split"]),
        "data_root": str(_path(args.data_root)),
        "method": METHOD,
        "training_dataset": TRAINING_DATASET,
        "protocol": PROTOCOL,
        "selected_num_samples": len(selected_records),
        "selected_sample_names": sample_names,
        "strict_protocol": strict,
        "inference_mode": INFERENCE_MODE,
        "tile_size": int(args.tile_size),
        "model_input_size": int(args.input_size),
        "padding": "zero_right_bottom",
        "metric_size": "original",
        "test_classes": model_classes,
        "checkpoint": str(_path(args.checkpoint)),
        "config": str(_path(args.config)),
        "class_json": str(class_json),
        "clip_vitb": str(_path(args.clip_vitb)),
        "rsib": str(_path(args.rsib)),
        "prompt_ensemble": args.prompt_ensemble,
        "num_layers": int(args.num_layers),
        "amp": args.amp,
        "mask_id_base": args.mask_id_base,
        "resolved_mask_id_base": dataset_audit["mask_encoding"][
            "resolved_mask_id_base"
        ],
        "compute_wfm": bool(args.compute_wfm),
    }
    _validate_resume_protocol(
        output_root,
        expected=resume_protocol,
        overwrite=args.overwrite,
    )

    cfg, model, load_result = _configure_model(
        args,
        local_rank,
        num_test_classes=num_classes,
    )
    palette = np.asarray(spec["palette"], dtype=np.uint8)
    if rank == 0:
        _write_json(
            output_root / "dataset_audit.json",
            {
                **dataset_audit,
                "protocol": PROTOCOL,
                "strict_protocol": strict,
                "expected_num_samples": UAVID_EXPECTED_SAMPLES,
                "selected_num_samples": len(selected_records),
                "model_text_classes": model_classes,
                "evaluation_classes": list(evaluation_classes),
            },
        )
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": "UAVid",
                "split": str(spec["split"]),
                "protocol": PROTOCOL,
                "classes": [
                    {
                        "id": index,
                        "name": evaluation_classes[index],
                        "model_text": model_classes[index],
                        "rgb": palette[index].tolist(),
                    }
                    for index in range(num_classes)
                ],
                "ignore_index": IGNORE_INDEX,
                "background_clutter_is_evaluated_class": True,
                "extra_negative_or_support_class": False,
                "label_protocol": str(spec["label_protocol"]),
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                **resume_protocol,
                "evaluation_setting": EVALUATION_SETTING,
                "official_gsnet_uavid_protocol": False,
                "paper_table_ovrsisbenchv2_comparable": False,
                "expected_num_samples": UAVID_EXPECTED_SAMPLES,
                "complete_population": complete_population,
                "complete_vistar_protocol": complete_vistar_protocol,
                "pooling_sizes": [1, 1],
                "sliding_window": False,
                "tile_overlap": 0,
                "encoder_internal_size": 384,
                "world_size": world_size,
                "save_images": bool(args.save_images),
                "load_result": str(load_result),
            },
        )
        print(
            f"[eval_gsnet_uavid] samples={len(selected_records)} "
            f"strict={strict} complete={complete_vistar_protocol}",
            flush=True,
        )
        print(
            "[eval_gsnet_uavid] preprocessing=native RGB -> non-overlap "
            f"{args.tile_size} tiles -> GSNet internal 384 encoders -> "
            "native reassembly",
            flush=True,
        )
    _barrier(world_size)

    local_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    local_rows: list[dict[str, Any]] = []
    local_forward_tiles = 0
    local_total_tiles = 0
    indexed_records = list(enumerate(selected_records))[rank::world_size]
    for global_index, record in tqdm(
        indexed_records,
        desc=f"GSNet UAVid rank {rank}",
    ):
        image_rgb = load_rgb(record, "uavid")
        height, width = image_rgb.shape[:2]
        target = load_target(
            record,
            "uavid",
            height=height,
            width=width,
        )
        pred_mask_path = output_dirs["pred_mask"] / f"{record.name}.png"
        gt_mask_path = output_dirs["gt_mask"] / f"{record.name}.png"
        pred_rgb_path = output_dirs["pred_rgb"] / f"{record.name}.png"
        gt_rgb_path = output_dirs["gt_rgb"] / f"{record.name}.png"

        cache_hit = pred_mask_path.is_file() and not args.overwrite
        tile_count = tile_grid_count(height, width, args.tile_size)
        if cache_hit:
            try:
                prediction = _load_cached_prediction(
                    pred_mask_path,
                    expected_shape=(height, width),
                    num_classes=num_classes,
                )
            except (OSError, ValueError) as error:
                print(
                    f"[eval_gsnet_uavid] invalid cache, recomputing "
                    f"{record.name}: {error}",
                    flush=True,
                )
                cache_hit = False
        if not cache_hit:
            prediction, observed_tiles = _predict_tiled(
                model=model,
                cfg=cfg,
                image_rgb=image_rgb,
                image_path=record.image_path,
                amp=args.amp,
                tile_size=args.tile_size,
                num_classes=num_classes,
            )
            if observed_tiles != tile_count:
                raise RuntimeError(
                    f"Tile count changed for {record.name}: "
                    f"{observed_tiles} vs {tile_count}"
                )
            _save_id_mask(prediction, pred_mask_path)
            local_forward_tiles += tile_count
        local_total_tiles += tile_count

        if args.overwrite or not gt_mask_path.is_file():
            _save_id_mask(target, gt_mask_path)
        if args.save_images and (
            args.overwrite or not cache_hit or not pred_rgb_path.is_file()
        ):
            _save_png_atomic(
                Image.fromarray(
                    colorize_mask(
                        prediction,
                        palette,
                        ignore_index=IGNORE_INDEX,
                    ),
                    mode="RGB",
                ),
                pred_rgb_path,
            )
        if args.save_images and (
            args.overwrite or not gt_rgb_path.is_file()
        ):
            _save_png_atomic(
                Image.fromarray(
                    colorize_mask(
                        target,
                        palette,
                        ignore_index=IGNORE_INDEX,
                    ),
                    mode="RGB",
                ),
                gt_rgb_path,
            )

        matrix = confusion_matrix(
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
        )
        local_confusion += matrix
        per_image_metrics = metrics_for_dataset(matrix, "uavid")
        boundary_metrics: dict[str, Any] = {}
        if args.compute_wfm:
            boundary_metrics = score_semantic_boundary_wfm(
                prediction,
                target,
                ignore_index=IGNORE_INDEX,
                ignore_margin=2,
                num_classes=num_classes,
            )
        local_rows.append(
            {
                "global_index": int(global_index),
                "name": record.name,
                "sample_id": record.sample_id,
                "domain": record.domain,
                "sequence": record.domain,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "prediction_path": str(pred_mask_path),
                "height": int(height),
                "width": int(width),
                "tile_size": int(args.tile_size),
                "num_tiles": int(tile_count),
                "cache_hit": bool(cache_hit),
                "ignored_pixels": int(target.size - matrix.sum()),
                **per_image_metrics,
                **boundary_metrics,
            }
        )

    _write_json(
        output_root / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "confusion_matrix": local_confusion.tolist(),
            "forward_tiles": local_forward_tiles,
            "total_tiles": local_total_tiles,
            "rows": local_rows,
        },
    )
    _barrier(world_size)

    if rank == 0:
        merged_confusion = np.zeros_like(local_confusion)
        merged_rows: list[dict[str, Any]] = []
        forward_tiles = 0
        total_tiles = 0
        for process_rank in range(world_size):
            payload = json.loads(
                (output_root / f"rank_{process_rank:05d}.json").read_text(
                    encoding="utf-8"
                )
            )
            merged_confusion += np.asarray(
                payload["confusion_matrix"],
                dtype=np.int64,
            )
            forward_tiles += int(payload["forward_tiles"])
            total_tiles += int(payload["total_tiles"])
            merged_rows.extend(payload["rows"])
        indices = sorted(int(row["global_index"]) for row in merged_rows)
        if indices != list(range(len(selected_records))):
            raise RuntimeError(
                "Distributed GSNet UAVid merge has missing or duplicate "
                f"samples: expected 0..{len(selected_records) - 1}, "
                f"found {indices[:20]}"
            )
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        cache_audit = _validate_cache_inventory(
            output_root,
            sample_names,
        )
        if not cache_audit["complete"]:
            raise RuntimeError(
                "GSNet UAVid prediction-cache inventory is incomplete: "
                f"{cache_audit}"
            )
        result = {
            "dataset": "UAVid",
            "dataset_key": "uavid",
            "split": str(spec["split"]),
            "method": METHOD,
            "training_dataset": TRAINING_DATASET,
            "evaluation_setting": EVALUATION_SETTING,
            "official_gsnet_uavid_protocol": False,
            "paper_table_ovrsisbenchv2_comparable": False,
            "protocol": PROTOCOL,
            "checkpoint": str(_path(args.checkpoint)),
            "num_samples": len(merged_rows),
            "expected_num_samples": UAVID_EXPECTED_SAMPLES,
            "complete_population": complete_population,
            "complete_vistar_protocol": complete_vistar_protocol,
            "num_classes": num_classes,
            "classes": list(evaluation_classes),
            "model_text_classes": model_classes,
            "background_clutter_is_evaluated_class": True,
            "extra_negative_or_support_class": False,
            "inference_mode": INFERENCE_MODE,
            "tile_size": int(args.tile_size),
            "model_input_size": int(args.input_size),
            "encoder_internal_size": 384,
            "padding": "zero_right_bottom",
            "tile_overlap": 0,
            "metric_size": "original",
            "num_native_images": len(merged_rows),
            "num_tiles": total_tiles,
            "num_forward_tiles": forward_tiles,
            "world_size": world_size,
            "amp": args.amp,
            "prompt_ensemble": args.prompt_ensemble,
            "num_layers": int(args.num_layers),
            "pooling_sizes": [1, 1],
            "test_time_augmentation": "none",
            "compute_wfm": bool(args.compute_wfm),
            "save_pred_mask": True,
            "save_gt_mask": True,
            "save_pred_rgb": bool(args.save_images),
            "save_gt_rgb": bool(args.save_images),
            "confusion_matrix": merged_confusion.tolist(),
            "total_pixels": int(
                sum(int(row["height"]) * int(row["width"]) for row in merged_rows)
            ),
            "ignored_pixels": int(
                sum(int(row["ignored_pixels"]) for row in merged_rows)
            ),
            "cache_audit": cache_audit,
            **metrics_for_dataset(merged_confusion, "uavid"),
        }
        if args.compute_wfm:
            result.update(aggregate_binary_boundary_wfm(merged_rows))
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_csv(output_root / "per_image_metrics.csv", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(
            f"[eval_gsnet_uavid] saved outputs to: {output_root}",
            flush=True,
        )

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
