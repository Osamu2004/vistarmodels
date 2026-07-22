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
    FLAIR_VISUAL_PALETTE_U8,
    IGNORE_INDEX,
    FlairRecord,
    discover_flair1_test,
    flair_confusion_matrix,
    flair_metrics_from_confusion,
    load_flair_mask_array,
    load_flair_rgb_u8,
)


Image.MAX_IMAGE_PIXELS = None
VISUAL_PALETTE_U8 = np.asarray(FLAIR_VISUAL_PALETTE_U8, dtype=np.uint8)


def _normalize_wsl_unc(path: str) -> str:
    text = str(path).strip().strip("\"'").replace("\\", "/")
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if text.casefold().startswith(prefix.casefold()):
            parts = [part for part in text.strip("/").split("/") if part]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _path(value: str) -> Path:
    return Path(_normalize_wsl_unc(value)).expanduser().resolve()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_string_list(path: Path, *, name: str) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{name} must be a non-empty JSON list: {path}")
    values: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} entry {index} must be a non-empty string: {path}"
            )
        values.append(value.strip())
    return values


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # Each rank owns an independent RSKT-Seg model. Gloo synchronizes only
        # CPU metadata; no model tensors are communicated.
        torch.distributed.init_process_group(backend="gloo")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def _make_output_dirs(root: Path) -> dict[str, Path]:
    directories = {
        "pred_mask": root / "pred_mask",
        "gt_mask": root / "gt_mask",
        "pred_rgb": root / "pred_rgb",
        "gt_rgb": root / "gt_rgb",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _save_id_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(path)


def _load_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.uint8)
    num_classes = len(FLAIR_GSNET_CLASSES)
    if prediction.size and int(prediction.max()) >= num_classes:
        raise ValueError(
            f"Cached FLAIR prediction contains IDs outside 0..{num_classes - 1}: "
            f"{path}"
        )
    return prediction


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    rgb = np.full((*values.shape, 3), 127, dtype=np.uint8)
    valid = (values >= 0) & (values < len(FLAIR_GSNET_CLASSES))
    rgb[valid] = VISUAL_PALETTE_U8[values[valid].astype(np.int64)]
    return rgb


def _configure_model(
    args: argparse.Namespace,
    local_rank: int,
    *,
    num_classes: int,
):
    source_root = _path(args.rskt_root)
    sys.path.insert(0, str(source_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config

    from RSKT_Seg import add_RSKT_seg_config
    from RSKT_Seg.third_party import clip as official_clip

    official_clip.pretrained["ViT-L/14@336px"] = str(_path(args.clip_vitl))
    clip_vitb = _path(args.clip_vitb)
    if clip_vitb.is_file():
        official_clip.pretrained["ViT-B/32"] = str(clip_vitb)

    source_train_json = source_root / "datasets" / "DLRSD.json"
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_RSKT_seg_config(cfg)
    cfg.merge_from_file(str(_path(args.config)))
    cfg.MODEL.DEVICE = f"cuda:{local_rank}"
    cfg.MODEL.WEIGHTS = str(_path(args.checkpoint))
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = str(source_train_json)
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(_path(args.class_json))
    cfg.MODEL.SEM_SEG_HEAD.DINO_WEIGHTS = str(_path(args.rsib))
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED_WEIGHTS_REMOTE = str(
        _path(args.remote_clip)
    )
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = int(args.num_layers)
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = args.prompt_ensemble
    if str(cfg.INPUT.FORMAT).upper() != "RGB":
        raise ValueError(
            "The FLAIR loader returns RGB bands 1--3, but the selected "
            f"RSKT-Seg config requests INPUT.FORMAT={cfg.INPUT.FORMAT!r}."
        )
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_size)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_size)
    cfg.TEST.SLIDING_WINDOW = False
    cfg.freeze()

    # These trusted official releases predate the PyTorch 2.6 weights_only
    # default. Limit the compatibility override to model construction/loading.
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


def _predict_patch(
    *,
    model: torch.nn.Module,
    cfg: Any,
    image_rgb_u8: np.ndarray,
    image_path: Path,
    amp: str,
) -> np.ndarray:
    from detectron2.data import transforms as transforms

    if image_rgb_u8.shape != (512, 512, 3) or image_rgb_u8.dtype != np.uint8:
        raise ValueError(
            "RSKT-Seg FLAIR input must be native 512x512 RGB uint8; "
            f"got shape={image_rgb_u8.shape}, dtype={image_rgb_u8.dtype}"
        )
    resize = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    transformed = resize.get_transform(image_rgb_u8).apply_image(image_rgb_u8)
    tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)
    )
    model_input = {
        "image": tensor,
        "height": 512,
        "width": 512,
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
        scores = model([model_input])[0]["sem_seg"]
    num_classes = len(FLAIR_GSNET_CLASSES)
    if scores.ndim != 3 or scores.shape[0] != num_classes:
        raise RuntimeError(
            "RSKT-Seg did not return the expected 12 FLAIR class maps: "
            f"shape={tuple(scores.shape)}"
        )
    prediction = scores.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
    if prediction.shape != (512, 512):
        raise RuntimeError(
            f"RSKT-Seg returned spatial shape {prediction.shape}, expected (512, 512)"
        )
    return prediction


def _validate_resume_protocol(
    output_root: Path,
    *,
    args: argparse.Namespace,
    selected_num_samples: int,
    strict_protocol: bool,
) -> None:
    predictions = list((output_root / "pred_mask").glob("*.png"))
    if args.overwrite or not predictions:
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
        "dataset": "FLAIR#1",
        "split": "flair#1-test",
        "data_root": str(_path(args.data_root)),
        "method": "RSKT-Seg",
        "checkpoint": str(_path(args.checkpoint)),
        "config": str(_path(args.config)),
        "class_json": str(_path(args.class_json)),
        "test_model_classes": list(FLAIR_GSNET_MODEL_CLASSES),
        "selected_num_samples": int(selected_num_samples),
        "strict_protocol": bool(strict_protocol),
        "inference_mode": "whole_patch_shortest_edge_resize",
        "native_patch_size": 512,
        "model_input_size": int(args.input_size),
        "rgb_source_band_indices_one_based": [1, 2, 3],
        "prompt_ensemble": args.prompt_ensemble,
        "num_layers": int(args.num_layers),
        "amp": args.amp,
    }
    mismatches = {
        key: {"existing": existing.get(key), "expected": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing predictions use a different RSKT-Seg FLAIR protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the DLRSD-trained RSKT-Seg ViT-L checkpoint on the "
            "official FLAIR #1 test set under GSNet's 12-class label protocol."
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
    parser.add_argument(
        "--data_root", default="/root/data/FLAIR-1-2/data/flair#1-test"
    )
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
            Path(__file__).resolve().parent
            / "configs"
            / "flair_12_classes.json"
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
    parser.add_argument(
        "--input_size",
        type=int,
        default=640,
        help=(
            "Detectron2 shortest-edge test size. The released DLRSD ViT-L "
            "configuration uses 640; predictions are restored to native 512."
        ),
    )
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument(
        "--prompt_ensemble",
        choices=("single", "imagenet", "imagenet_select"),
        default="single",
    )
    parser.add_argument("--amp", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--save_pred_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_gt_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_strict_protocol", action="store_true")
    args = parser.parse_args()

    if args.input_size <= 0:
        parser.error("--input_size must be positive")
    if args.num_layers != 5:
        parser.error("The released DLRSD + ViT-L checkpoint requires --num_layers 5")
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if not torch.cuda.is_available():
        parser.error("RSKT-Seg FLAIR evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    strict_protocol = not bool(args.no_strict_protocol)
    records, dataset_audit = discover_flair1_test(
        args.data_root,
        strict=strict_protocol,
    )
    indexed_records = list(enumerate(records))
    if args.max_samples > 0:
        indexed_records = indexed_records[: args.max_samples]
    if not indexed_records:
        raise RuntimeError("No FLAIR samples were selected")
    comparable_full_test = bool(
        strict_protocol and len(indexed_records) == FLAIR1_EXPECTED_SAMPLES
    )

    source_root = _path(args.rskt_root)
    train_class_json = source_root / "datasets" / "DLRSD.json"
    model_classes = _load_string_list(_path(args.class_json), name="FLAIR classes")
    if tuple(model_classes) != tuple(FLAIR_GSNET_MODEL_CLASSES):
        raise ValueError(
            "RSKT-Seg FLAIR class JSON must exactly match GSNet's official "
            f"prompt order: expected={list(FLAIR_GSNET_MODEL_CLASSES)}, "
            f"found={model_classes}"
        )
    training_classes = _load_string_list(train_class_json, name="DLRSD classes")

    required = {
        "official source": source_root / "RSKT_Seg" / "RSKT_Seg.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        "FLAIR class JSON": _path(args.class_json),
        "DLRSD training taxonomy": train_class_json,
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
    _validate_resume_protocol(
        output_root,
        args=args,
        selected_num_samples=len(indexed_records),
        strict_protocol=strict_protocol,
    )
    local_records = indexed_records[rank::world_size]

    cfg, model, load_result = _configure_model(
        args,
        local_rank,
        num_classes=len(model_classes),
    )
    if rank == 0:
        dataset_audit = {
            **dataset_audit,
            "protocol": "GSNet_FLAIR_12class",
            "model_class_names": model_classes,
            "display_class_names": list(FLAIR_GSNET_CLASSES),
            "raw_to_eval_mapping": {str(raw): raw - 1 for raw in range(1, 13)},
            "ignored_raw_ids": [0, *range(13, 20), 255],
            "selected_num_samples": len(indexed_records),
            "comparable_full_test": comparable_full_test,
        }
        _write_json(output_root / "dataset_audit.json", dataset_audit)
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": "FLAIR#1",
                "split": "flair#1-test",
                "protocol": "GSNet_FLAIR_12class",
                "classes": [
                    {
                        "id": index,
                        "name": display_name,
                        "model_text": model_classes[index],
                        "rgb": VISUAL_PALETTE_U8[index].tolist(),
                    }
                    for index, display_name in enumerate(FLAIR_GSNET_CLASSES)
                ],
                "ignore_index": IGNORE_INDEX,
                "ignore_visualization_rgb": [127, 127, 127],
                "raw_to_eval_mapping": {str(raw): raw - 1 for raw in range(1, 13)},
                "ignored_raw_ids": [0, *range(13, 20), 255],
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                "dataset": "FLAIR#1",
                "split": "flair#1-test",
                "data_root": str(_path(args.data_root)),
                "method": "RSKT-Seg",
                "training_dataset": "DLRSD",
                "training_class_json": str(train_class_json),
                "training_classes": training_classes,
                "evaluation_setting": "cross-dataset/out-of-domain",
                "official_rskt_flair_reproduction": False,
                "paper_table_ovrsisbenchv2_comparable": False,
                "protocol": "GSNet_FLAIR_12class",
                "strict_protocol": strict_protocol,
                "selected_num_samples": len(indexed_records),
                "expected_num_samples": FLAIR1_EXPECTED_SAMPLES,
                "checkpoint": str(_path(args.checkpoint)),
                "config": str(_path(args.config)),
                "class_json": str(_path(args.class_json)),
                "test_model_classes": model_classes,
                "display_classes": list(FLAIR_GSNET_CLASSES),
                "inference_mode": "whole_patch_shortest_edge_resize",
                "native_patch_size": 512,
                "model_input_size": args.input_size,
                "tile_size": 512,
                "tile_resize": args.input_size != 512,
                "rgb_source_band_indices_one_based": [1, 2, 3],
                "five_band_channels_ignored": [4, 5],
                "ground_truth_used_as_model_input": False,
                "same_dataset_and_metric_protocol_as_gsnet": True,
                "same_model_preprocessing_as_gsnet": False,
                "rskt_official_default_min_size_test": 640,
                "prompt_ensemble": args.prompt_ensemble,
                "num_layers": args.num_layers,
                "pooling_sizes": [1, 1],
                "amp": args.amp,
                "clip_vitl": str(_path(args.clip_vitl)),
                "remote_clip": str(_path(args.remote_clip)),
                "rsib": str(_path(args.rsib)),
                "world_size": world_size,
                "save_pred_mask": True,
                "save_gt_mask": True,
                "save_pred_rgb": bool(args.save_pred_rgb),
                "save_gt_rgb": bool(args.save_gt_rgb),
                "load_result": str(load_result),
            },
        )
    _barrier(world_size)

    num_classes = len(FLAIR_GSNET_CLASSES)
    local_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    local_domain_confusions = {
        domain: np.zeros((num_classes, num_classes), dtype=np.int64)
        for domain in FLAIR1_TEST_DOMAIN_COUNTS
    }
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        local_records,
        desc=f"RSKT-Seg FLAIR rank {rank}",
        disable=False,
    )
    for global_index, record in progress:
        assert isinstance(record, FlairRecord)
        output_name = record.output_name
        pred_mask_path = output_dirs["pred_mask"] / f"{output_name}.png"
        gt_mask_path = output_dirs["gt_mask"] / f"{output_name}.png"
        pred_rgb_path = output_dirs["pred_rgb"] / f"{output_name}.png"
        gt_rgb_path = output_dirs["gt_rgb"] / f"{output_name}.png"

        target = load_flair_mask_array(record.mask_path, ignore_index=IGNORE_INDEX)
        if target.shape != (512, 512):
            raise ValueError(
                f"FLAIR target must be native 512x512: {record.mask_path} "
                f"shape={target.shape}"
            )
        if pred_mask_path.is_file() and not args.overwrite:
            prediction = _load_prediction(pred_mask_path)
        else:
            image_rgb_u8 = load_flair_rgb_u8(record.image_path)
            prediction = _predict_patch(
                model=model,
                cfg=cfg,
                image_rgb_u8=image_rgb_u8,
                image_path=record.image_path,
                amp=args.amp,
            )
            _save_id_mask(prediction, pred_mask_path)

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch for {record.output_name}: "
                f"prediction={prediction.shape}, gt={target.shape}"
            )
        if args.overwrite or not gt_mask_path.is_file():
            _save_id_mask(target, gt_mask_path)
        if args.save_pred_rgb and (args.overwrite or not pred_rgb_path.is_file()):
            Image.fromarray(_colorize_mask(prediction), mode="RGB").save(pred_rgb_path)
        if args.save_gt_rgb and (args.overwrite or not gt_rgb_path.is_file()):
            Image.fromarray(_colorize_mask(target), mode="RGB").save(gt_rgb_path)

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
                "name": output_name,
                "domain": record.domain,
                "zone": record.zone,
                "sample_id": record.sample_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "prediction_path": str(pred_mask_path),
                "height": 512,
                "width": 512,
                **image_metrics,
                **image_wfm,
            }
        )

    _write_json(
        output_root / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "confusion_matrix": local_confusion.tolist(),
            "domain_confusion_matrices": {
                domain: confusion.tolist()
                for domain, confusion in local_domain_confusions.items()
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
            with rank_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            confusion += np.asarray(payload["confusion_matrix"], dtype=np.int64)
            for domain in domain_confusions:
                domain_confusions[domain] += np.asarray(
                    payload["domain_confusion_matrices"][domain],
                    dtype=np.int64,
                )
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        if len(merged_rows) != len(indexed_records):
            raise RuntimeError(
                f"Distributed result count mismatch: expected {len(indexed_records)}, "
                f"found {len(merged_rows)}"
            )

        total_pixels = len(merged_rows) * 512 * 512
        valid_pixels = int(confusion.sum())
        result = {
            "dataset": "FLAIR#1",
            "split": "flair#1-test",
            "method": "RSKT-Seg",
            "training_dataset": "DLRSD",
            "training_class_json": str(train_class_json),
            "training_classes": training_classes,
            "evaluation_setting": "cross-dataset/out-of-domain",
            "official_rskt_flair_reproduction": False,
            "paper_table_ovrsisbenchv2_comparable": False,
            "protocol": "GSNet_FLAIR_12class",
            "checkpoint": str(_path(args.checkpoint)),
            "config": str(_path(args.config)),
            "test_model_classes": model_classes,
            "classes": list(FLAIR_GSNET_CLASSES),
            "ignored_raw_ids": [0, *range(13, 20), 255],
            "num_classes": num_classes,
            "num_samples": len(merged_rows),
            "expected_num_samples": FLAIR1_EXPECTED_SAMPLES,
            "expected_num_zones": FLAIR1_EXPECTED_ZONES,
            "comparable_full_test": comparable_full_test,
            "official_test_domains": list(FLAIR1_TEST_DOMAIN_COUNTS),
            "inference_mode": "whole_patch_shortest_edge_resize",
            "native_patch_size": 512,
            "model_input_size": args.input_size,
            "tile_size": 512,
            "tile_resize": args.input_size != 512,
            "num_tiles": len(merged_rows),
            "rgb_source_band_indices_one_based": [1, 2, 3],
            "five_band_channels_ignored": [4, 5],
            "ground_truth_used_as_model_input": False,
            "same_dataset_and_metric_protocol_as_gsnet": True,
            "same_model_preprocessing_as_gsnet": False,
            "world_size": world_size,
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "ignored_pixels": total_pixels - valid_pixels,
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
            **flair_metrics_from_confusion(confusion),
            **aggregate_binary_boundary_wfm(merged_rows),
        }
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_csv(output_root / "per_image_metrics.csv", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[eval_rskt_seg_flair] saved outputs to: {output_root}")

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
