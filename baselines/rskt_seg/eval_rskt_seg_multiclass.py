"""Shared standalone RSKT-Seg engine for VISTAR multiclass protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # CPU-only protocol tests do not require PyTorch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)
from baselines.segearth_ov.protocols import (  # noqa: E402
    DATASET_SPECS,
    IGNORE_INDEX,
    EvalSample,
    colorize_mask,
    confusion_matrix,
    discover_dataset,
    load_rgb,
    load_target,
    metrics_for_dataset,
    normalize_path,
)


Image.MAX_IMAGE_PIXELS = None
OFFICIAL_RSKT_COMMIT = "7b84091598e1edc3236dfbf45cc27e7e3436ffcb"
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class DatasetContract:
    """Dataset-specific declarations consumed by the shared evaluator."""

    dataset_key: str
    model_classes: tuple[str, ...]
    default_data_root: str
    class_filename: str
    complete_protocol_name: str
    partial_protocol_name: str
    primary_metric_description: str
    auxiliary_metric_description: str | None
    official_rskt_taxonomy_protocol: bool
    taxonomy_note: str
    background_is_class_zero: bool = True

    @property
    def spec(self) -> dict[str, Any]:
        return DATASET_SPECS[self.dataset_key]

    @property
    def display_name(self) -> str:
        return str(self.spec["display_name"])

    @property
    def expected_samples(self) -> int:
        return int(self.spec["expected_samples"])

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.spec["classes"])

    @property
    def palette(self) -> np.ndarray:
        return np.asarray(self.spec["palette"], dtype=np.uint8)


def _path(value: str | os.PathLike[str]) -> Path:
    return normalize_path(value)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(path, handle.getvalue())


def _save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )
    try:
        image.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_id_mask(mask: np.ndarray, path: Path) -> None:
    _save_image_atomic(
        Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L"),
        path,
    )


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


def validate_model_classes(
    contract: DatasetContract,
    model_classes: Sequence[str],
) -> None:
    """Require the exact model-channel order declared by the dataset contract."""

    actual = tuple(str(value) for value in model_classes)
    if actual != contract.model_classes:
        raise ValueError(
            f"RSKT-Seg {contract.display_name} must use exactly "
            f"{len(contract.model_classes)} declared output classes; no extra "
            "negative class or no-data/background-support class is allowed. "
            f"expected={list(contract.model_classes)}, "
            f"found={list(actual)}"
        )


def shortest_edge_shape(
    height: int,
    width: int,
    min_size: int,
    max_size: int,
) -> tuple[int, int]:
    """Match Detectron2 ``ResizeShortestEdge`` dimension rounding."""

    if min(height, width, min_size, max_size) <= 0:
        raise ValueError("Image and resize dimensions must be positive")
    scale = float(min_size) / float(min(height, width))
    if scale * float(max(height, width)) > float(max_size):
        scale = float(max_size) / float(max(height, width))
    return max(1, int(height * scale + 0.5)), max(
        1, int(width * scale + 0.5)
    )


def strict_protocol_errors(
    contract: DatasetContract,
    args: argparse.Namespace,
    full_count: int,
) -> list[str]:
    errors: list[str] = []
    if int(args.expected_samples) != contract.expected_samples:
        errors.append(f"expected_samples must be {contract.expected_samples}")
    if int(full_count) != contract.expected_samples:
        errors.append(
            f"full {contract.display_name} population must contain "
            f"{contract.expected_samples} pairs"
        )
    if int(args.min_size_test) != 640:
        errors.append("min_size_test must be 640")
    if int(args.max_size_test) != 2560:
        errors.append("max_size_test must be 2560")
    if int(args.num_layers) != 5:
        errors.append("num_layers must be 5")
    if str(args.prompt_ensemble) != "single":
        errors.append("prompt_ensemble must be single")
    if str(args.amp) != "fp32":
        errors.append("amp must be fp32")
    return errors


def _sample_manifest_sha256(records: Sequence[EvalSample]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for value in (record.name, record.image_path, record.mask_path):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not torch.distributed.is_initialized():
        # Each rank owns an independent model. Gloo only synchronizes metadata.
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

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_RSKT_seg_config(cfg)
    cfg.merge_from_file(str(_path(args.config)))
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
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(num_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = int(args.num_layers)
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = args.prompt_ensemble
    cfg.INPUT.MIN_SIZE_TEST = int(args.min_size_test)
    cfg.INPUT.MAX_SIZE_TEST = int(args.max_size_test)
    cfg.TEST.SLIDING_WINDOW = False
    if str(cfg.INPUT.FORMAT).upper() != "RGB":
        raise ValueError(
            "The multiclass adapter returns RGB, but RSKT-Seg requests "
            f"INPUT.FORMAT={cfg.INPUT.FORMAT!r}"
        )
    cfg.freeze()

    # These trusted official checkpoints predate PyTorch's weights_only default.
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


def _predict_whole_image(
    *,
    model: torch.nn.Module,
    cfg: Any,
    image_rgb: np.ndarray,
    image_path: Path,
    amp: str,
    num_classes: int,
    dataset_name: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    from detectron2.data import transforms as transforms

    height, width = image_rgb.shape[:2]
    resize = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )
    transformed = resize.get_transform(image_rgb).apply_image(image_rgb)
    expected_shape = shortest_edge_shape(
        height,
        width,
        int(cfg.INPUT.MIN_SIZE_TEST),
        int(cfg.INPUT.MAX_SIZE_TEST),
    )
    if transformed.shape[:2] != expected_shape:
        raise RuntimeError(
            f"Detectron2 resize disagrees with the recorded {dataset_name} protocol: "
            f"actual={transformed.shape[:2]}, expected={expected_shape}"
        )
    tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)
    )
    model_input = {
        "image": tensor,
        "height": height,
        "width": width,
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
        if scores.ndim != 3 or scores.shape[0] != int(num_classes):
            raise RuntimeError(
                f"RSKT-Seg must return exactly {num_classes} {dataset_name} "
                "class maps without an extra support channel, got "
                f"{tuple(scores.shape)}"
            )
        if tuple(scores.shape[-2:]) != (height, width):
            scores = F.interpolate(
                scores.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        prediction = scores.argmax(dim=0).cpu().numpy().astype(np.uint8)
    if prediction.shape != (height, width):
        raise RuntimeError(
            f"Prediction shape {prediction.shape} does not match {(height, width)}"
        )
    return prediction, expected_shape


def _load_cached_prediction(
    path: Path,
    shape: tuple[int, int],
    *,
    num_classes: int,
    dataset_name: str,
) -> np.ndarray:
    with Image.open(path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.uint8)
    if prediction.shape != shape:
        raise ValueError(
            f"Cached prediction shape {prediction.shape} does not match {shape}: {path}"
        )
    if prediction.size and int(prediction.max()) >= int(num_classes):
        raise ValueError(
            f"Cached prediction contains a non-{dataset_name} class ID: {path}"
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


def _prediction_fingerprint(
    contract: DatasetContract,
    args: argparse.Namespace,
    full_records: Sequence[EvalSample],
    selected_records: Sequence[EvalSample],
    model_classes: Sequence[str],
    mask_id_base: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": contract.display_name,
        "dataset_key": contract.dataset_key,
        "method": "RSKT-Seg",
        "data_root": str(_path(args.data_root)),
        "checkpoint": str(_path(args.checkpoint)),
        "config": str(_path(args.config)),
        "class_json": str(_path(args.class_json)),
        "test_model_classes": list(model_classes),
        "negative_support_class": False,
        "background_is_class_zero": contract.background_is_class_zero,
        "official_rskt_taxonomy_protocol": (
            contract.official_rskt_taxonomy_protocol
        ),
        "taxonomy_note": contract.taxonomy_note,
        "full_num_samples": len(full_records),
        "full_manifest_sha256": _sample_manifest_sha256(full_records),
        "selected_num_samples": len(selected_records),
        "selected_manifest_sha256": _sample_manifest_sha256(selected_records),
        "mask_id_base": mask_id_base,
        "inference_mode": "whole_image_shortest_edge_resize",
        "min_size_test": int(args.min_size_test),
        "max_size_test": int(args.max_size_test),
        "sliding_window": False,
        "pooling_sizes": [1, 1],
        "output_size": "original_image_extent",
        "prompt_ensemble": args.prompt_ensemble,
        "num_layers": int(args.num_layers),
        "amp": args.amp,
    }


def _validate_resume(
    contract: DatasetContract,
    output_root: Path,
    fingerprint: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    cached = list((output_root / "pred_mask").glob("*.png"))
    if overwrite or not cached:
        return
    run_config_path = output_root / "run_config.json"
    if not run_config_path.is_file():
        raise RuntimeError(
            "Cached predictions lack run_config.json; use OVERWRITE=1 or a new OUTPUT_DIR"
        )
    with run_config_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle).get("prediction_fingerprint")
    if previous != fingerprint:
        raise RuntimeError(
            f"Cached predictions use another {contract.display_name} protocol; "
            "use OVERWRITE=1 or a new OUTPUT_DIR"
        )


def _validate_cache_inventory(
    output_root: Path,
    records: Sequence[EvalSample],
) -> dict[str, Any]:
    expected = {f"{record.name}.png" for record in records}
    actual = {
        path.name
        for path in (output_root / "pred_mask").glob("*.png")
        if path.is_file()
    }
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        raise RuntimeError(
            "Prediction cache contains files outside this manifest; use a new "
            f"OUTPUT_DIR. examples={extra[:10]}"
        )
    return {
        "expected_predictions": len(expected),
        "present_predictions": len(actual),
        "missing_predictions": len(missing),
        "complete": not missing,
    }


def _parse_args(contract: DatasetContract) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the released DLRSD-trained RSKT-Seg ViT-L checkpoint "
            f"on the VISTAR-compatible {contract.display_name} population."
        )
    )
    default_rskt = REPO_ROOT / "third_party" / "RSKT-Seg"
    default_weight_root = Path("/root/data/weight/rskt_seg")
    default_checkpoint = (
        Path("/root/data/weight/RSKT-Seg-ckpt")
        / "0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5"
        / "model_final.pth"
    )
    parser.add_argument("--data_root", default=contract.default_data_root)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rskt_root", default=str(default_rskt))
    parser.add_argument(
        "--config", default=str(default_rskt / "configs" / "vitl_336_DLRSD.yaml")
    )
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument(
        "--class_json",
        default=str(Path(__file__).parent / "configs" / contract.class_filename),
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
        default=str(default_weight_root / "pretrained" / "RemoteCLIP-ViT-B-32.pt"),
    )
    parser.add_argument(
        "--rsib",
        default=os.environ.get("RSKT_RSIB", "/root/data/weight/rsib/RSIB.pth"),
    )
    parser.add_argument("--min_size_test", type=int, default=640)
    parser.add_argument("--max_size_test", type=int, default=2560)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument(
        "--prompt_ensemble",
        choices=("single", "imagenet", "imagenet_select"),
        default="single",
    )
    parser.add_argument("--amp", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--mask_id_base",
        choices=("auto", "zero", "one"),
        default="auto",
        help=(
            None
            if contract.dataset_key == "uavid"
            else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--expected_samples", type=int, default=contract.expected_samples
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--compute_wfm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_pred_rgb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_gt_rgb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_strict_protocol", action="store_true")
    args = parser.parse_args()

    if args.min_size_test <= 0 or args.max_size_test < args.min_size_test:
        parser.error("Resize sizes must satisfy 0 < min_size_test <= max_size_test")
    if args.expected_samples <= 0 or args.max_samples < 0:
        parser.error("expected_samples must be positive and max_samples non-negative")
    if args.num_layers != 5:
        parser.error("The released DLRSD + ViT-L checkpoint requires num_layers=5")
    if torch is None or not torch.cuda.is_available():
        parser.error(f"RSKT-Seg {contract.display_name} evaluation requires CUDA")
    return args


def run_evaluation(contract: DatasetContract) -> None:
    args = _parse_args(contract)
    spec = contract.spec
    classes = contract.classes
    palette = contract.palette
    strict = not bool(args.no_strict_protocol)
    full_records, dataset_audit = discover_dataset(
        contract.dataset_key,
        _path(args.data_root),
        strict=strict,
        mask_id_base=args.mask_id_base,
    )
    if strict:
        errors = strict_protocol_errors(contract, args, len(full_records))
        if errors:
            raise ValueError(
                f"Strict RSKT-Seg {contract.display_name} protocol violation: "
                + "; ".join(errors)
            )
    selected_records = list(full_records)
    if args.max_samples:
        selected_records = selected_records[: args.max_samples]
    if not selected_records:
        raise RuntimeError(f"No {contract.display_name} samples selected")

    source_root = _path(args.rskt_root)
    train_class_json = source_root / "datasets" / "DLRSD.json"
    required = {
        "official source": source_root / "RSKT_Seg" / "RSKT_Seg.py",
        "config": _path(args.config),
        "checkpoint": _path(args.checkpoint),
        f"{contract.display_name} class JSON": _path(args.class_json),
        "DLRSD training taxonomy": train_class_json,
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
            + "\nRun scripts/bootstrap_rskt_seg.sh and tools/check_rskt_seg_deps.py."
        )
    model_classes = _load_string_list(
        _path(args.class_json),
        name=f"{contract.display_name} classes",
    )
    validate_model_classes(contract, model_classes)
    training_classes = _load_string_list(train_class_json, name="DLRSD classes")
    mask_encoding = dataset_audit.get("mask_encoding")
    resolved_mask_id_base = (
        str(mask_encoding["resolved_mask_id_base"])
        if isinstance(mask_encoding, dict)
        else "not_applicable"
    )

    rank, world_size, local_rank, initialized_here = _distributed_context()
    torch.cuda.set_device(local_rank)
    output_root = _path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    fingerprint = _prediction_fingerprint(
        contract,
        args,
        full_records,
        selected_records,
        model_classes,
        resolved_mask_id_base,
    )
    _validate_resume(
        contract,
        output_root,
        fingerprint,
        overwrite=args.overwrite,
    )
    if rank == 0:
        _validate_cache_inventory(output_root, selected_records)
    _barrier(world_size)

    cfg, model, load_result = _configure_model(
        args,
        local_rank,
        num_classes=len(classes),
    )
    official_spatial_protocol = bool(
        int(args.min_size_test) == 640
        and int(args.max_size_test) == 2560
        and args.prompt_ensemble == "single"
        and args.amp == "fp32"
        and int(args.num_layers) == 5
    )
    complete_population = bool(
        strict
        and len(full_records) == contract.expected_samples
        and len(selected_records) == len(full_records)
    )
    complete_vistar_protocol = bool(
        official_spatial_protocol and complete_population
    )
    official_reproduction_protocol = bool(
        complete_vistar_protocol and contract.official_rskt_taxonomy_protocol
    )
    protocol_prefix = f"official_rskt_{contract.dataset_key}"
    if rank == 0:
        _write_json(
            output_root / "dataset_audit.json",
            {
                **dataset_audit,
                "selected_num_samples": len(selected_records),
                "selected_manifest_sha256": _sample_manifest_sha256(
                    selected_records
                ),
                "complete_population": complete_population,
            },
        )
        _write_json(
            output_root / "class_map.json",
            {
                "dataset": contract.display_name,
                "protocol": (
                    f"VISTAR_{contract.display_name}_{len(classes)}class_taxonomy"
                ),
                "classes": [
                    {
                        "id": class_id,
                        "name": classes[class_id],
                        "model_text": model_classes[class_id],
                        "rgb": palette[class_id].tolist(),
                    }
                    for class_id in range(len(classes))
                ],
                "ignore_index": IGNORE_INDEX,
                "label_protocol": spec["label_protocol"],
                "class_zero_included_in_macro_metrics": True,
                "background_is_class_zero": contract.background_is_class_zero,
                "negative_support_class": False,
                "official_rskt_taxonomy_protocol": (
                    contract.official_rskt_taxonomy_protocol
                ),
                "taxonomy_note": contract.taxonomy_note,
                "resolved_mask_id_base": resolved_mask_id_base,
            },
        )
        _write_json(
            output_root / "run_config.json",
            {
                **fingerprint,
                "prediction_fingerprint": fingerprint,
                "official_rskt_source_commit": OFFICIAL_RSKT_COMMIT,
                "training_dataset": "DLRSD",
                "training_class_json": str(train_class_json),
                "training_classes": training_classes,
                "evaluation_setting": "cross-dataset/out-of-domain",
                "strict_protocol": strict,
                "complete_population": complete_population,
                f"{protocol_prefix}_spatial_protocol": official_spatial_protocol,
                f"{protocol_prefix}_population_protocol": complete_population,
                f"{protocol_prefix}_reproduction_protocol": (
                    official_reproduction_protocol
                ),
                "complete_vistar_protocol": complete_vistar_protocol,
                "compute_wfm": bool(args.compute_wfm),
                "world_size": world_size,
                "save_pred_rgb": bool(args.save_pred_rgb),
                "save_gt_rgb": bool(args.save_gt_rgb),
                "clip_vitl": str(_path(args.clip_vitl)),
                "remote_clip": str(_path(args.remote_clip)),
                "rsib": str(_path(args.rsib)),
                "load_result": str(load_result),
            },
        )
    _barrier(world_size)

    local_confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    local_rows: list[dict[str, Any]] = []
    local_forward_passes = 0
    indexed_records = list(enumerate(selected_records))[rank::world_size]
    for global_index, record in tqdm(
        indexed_records,
        desc=f"RSKT-Seg {contract.display_name} rank {rank}",
    ):
        image_rgb = load_rgb(record, contract.dataset_key)
        height, width = image_rgb.shape[:2]
        target = load_target(
            record,
            contract.dataset_key,
            height=height,
            width=width,
        )
        pred_mask_path = output_dirs["pred_mask"] / f"{record.name}.png"
        gt_mask_path = output_dirs["gt_mask"] / f"{record.name}.png"
        pred_rgb_path = output_dirs["pred_rgb"] / f"{record.name}.png"
        gt_rgb_path = output_dirs["gt_rgb"] / f"{record.name}.png"

        cache_hit = pred_mask_path.is_file() and not args.overwrite
        if cache_hit:
            try:
                prediction = _load_cached_prediction(
                    pred_mask_path,
                    (height, width),
                    num_classes=len(classes),
                    dataset_name=contract.display_name,
                )
            except (OSError, ValueError) as error:
                print(
                    f"[eval_rskt_seg_{contract.dataset_key}] invalid cache, "
                    f"recomputing {record.name}: {error}",
                    flush=True,
                )
                cache_hit = False
            else:
                resized_shape = shortest_edge_shape(
                    height,
                    width,
                    args.min_size_test,
                    args.max_size_test,
                )
        if not cache_hit:
            prediction, resized_shape = _predict_whole_image(
                model=model,
                cfg=cfg,
                image_rgb=image_rgb,
                image_path=record.image_path,
                amp=args.amp,
                num_classes=len(classes),
                dataset_name=contract.display_name,
            )
            _save_id_mask(prediction, pred_mask_path)
            local_forward_passes += 1

        if args.overwrite or not gt_mask_path.is_file():
            _save_id_mask(target, gt_mask_path)
        if args.save_pred_rgb and (
            args.overwrite or not cache_hit or not pred_rgb_path.is_file()
        ):
            _save_image_atomic(
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
        if args.save_gt_rgb and (args.overwrite or not gt_rgb_path.is_file()):
            _save_image_atomic(
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
            num_classes=len(classes),
            ignore_index=IGNORE_INDEX,
        )
        local_confusion += matrix
        per_image_metrics = metrics_for_dataset(matrix, contract.dataset_key)
        boundary_metrics: dict[str, Any] = {}
        if args.compute_wfm:
            boundary_metrics = score_semantic_boundary_wfm(
                prediction,
                target,
                ignore_index=IGNORE_INDEX,
                ignore_margin=2,
                num_classes=len(classes),
            )
        local_rows.append(
            {
                "global_index": int(global_index),
                "sample_id": record.sample_id,
                "domain": record.domain,
                "sequence": record.domain,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "prediction_path": str(pred_mask_path),
                "height": int(height),
                "width": int(width),
                "model_input_height": int(resized_shape[0]),
                "model_input_width": int(resized_shape[1]),
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
            "forward_passes": local_forward_passes,
            "rows": local_rows,
        },
    )
    _barrier(world_size)

    if rank == 0:
        merged_confusion = np.zeros_like(local_confusion)
        merged_rows: list[dict[str, Any]] = []
        forward_passes = 0
        for process_rank in range(world_size):
            with (output_root / f"rank_{process_rank:05d}.json").open(
                "r", encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            merged_confusion += np.asarray(
                payload["confusion_matrix"], dtype=np.int64
            )
            forward_passes += int(payload["forward_passes"])
            merged_rows.extend(payload["rows"])
        indices = sorted(int(row["global_index"]) for row in merged_rows)
        if indices != list(range(len(selected_records))):
            raise RuntimeError(
                f"Distributed {contract.display_name} merge has missing or "
                "duplicate samples"
            )
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        cache_audit = _validate_cache_inventory(output_root, selected_records)
        if not cache_audit["complete"]:
            raise RuntimeError(
                f"{contract.display_name} inference has missing predictions: "
                f"{cache_audit}"
            )

        total_pixels = sum(
            int(row["height"]) * int(row["width"]) for row in merged_rows
        )
        result: dict[str, Any] = {
            "dataset": contract.display_name,
            "dataset_key": contract.dataset_key,
            "split": spec["split"],
            "method": "RSKT-Seg",
            "training_dataset": "DLRSD",
            "evaluation_setting": "cross-dataset/out-of-domain",
            "protocol": (
                contract.complete_protocol_name
                if complete_vistar_protocol
                else contract.partial_protocol_name
            ),
            "official_rskt_source_commit": OFFICIAL_RSKT_COMMIT,
            f"{protocol_prefix}_spatial_protocol": official_spatial_protocol,
            f"{protocol_prefix}_population_protocol": complete_population,
            f"{protocol_prefix}_reproduction_protocol": (
                official_reproduction_protocol
            ),
            "complete_vistar_protocol": complete_vistar_protocol,
            "official_rskt_taxonomy_protocol": (
                contract.official_rskt_taxonomy_protocol
            ),
            "taxonomy_note": contract.taxonomy_note,
            "label_protocol": spec["label_protocol"],
            "classes": list(classes),
            "test_model_classes": model_classes,
            "palette": palette.tolist(),
            "num_classes": len(classes),
            "background_is_class_zero": contract.background_is_class_zero,
            "negative_support_class": False,
            "class_zero_included_in_macro_metrics": True,
            "primary_metric": contract.primary_metric_description,
            "resolved_mask_id_base": resolved_mask_id_base,
            "num_samples": len(merged_rows),
            "full_population_num_samples": len(full_records),
            "expected_num_samples": contract.expected_samples,
            "complete_coverage": complete_vistar_protocol,
            "sample_manifest_sha256": _sample_manifest_sha256(selected_records),
            "inference_mode": "whole_image_shortest_edge_resize",
            "min_size_test": int(args.min_size_test),
            "max_size_test": int(args.max_size_test),
            "sliding_window": False,
            "pooling_sizes": [1, 1],
            "output_size": "original_image_extent",
            "prompt_ensemble": args.prompt_ensemble,
            "num_layers": int(args.num_layers),
            "amp": args.amp,
            "world_size": world_size,
            "forward_passes": forward_passes,
            "cache_hits": sum(bool(row["cache_hit"]) for row in merged_rows),
            "cache_inventory": cache_audit,
            "compute_wfm": bool(args.compute_wfm),
            "total_pixels": int(total_pixels),
            "valid_pixels": int(merged_confusion.sum()),
            "ignored_pixels": int(total_pixels - merged_confusion.sum()),
            "confusion_matrix": merged_confusion.tolist(),
            **metrics_for_dataset(merged_confusion, contract.dataset_key),
        }
        if contract.auxiliary_metric_description is not None:
            result["auxiliary_metric"] = contract.auxiliary_metric_description
        if args.compute_wfm:
            result.update(aggregate_binary_boundary_wfm(merged_rows))
        _write_jsonl(output_root / "predictions.jsonl", merged_rows)
        _write_csv(output_root / "per_image_metrics.csv", merged_rows)
        _write_json(output_root / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            f"[eval_rskt_seg_{contract.dataset_key}] saved outputs to: "
            f"{output_root}"
        )

    _barrier(world_size)
    if initialized_here:
        torch.distributed.destroy_process_group()
