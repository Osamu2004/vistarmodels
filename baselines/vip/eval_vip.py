"""VIP evaluation on the five Vistar open-vocabulary segmentation datasets."""

from __future__ import annotations

import argparse
import json
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
from baselines.segearth_ov.eval_segearth_ov import (  # noqa: E402
    _barrier,
    _distributed_context,
    _make_output_dirs,
    _save_id_mask,
    _save_visualizations,
    _validate_cached_prediction,
    _write_csv,
    _write_jsonl,
)
from baselines.segearth_ov.protocols import (  # noqa: E402
    DATASET_SPECS,
    confusion_matrix,
    discover_dataset,
    load_rgb,
    load_target,
    metrics_for_dataset,
    normalize_path,
    resize_keep_ratio,
    write_json,
)
from baselines.vip.protocols import (  # noqa: E402
    read_vip_class_groups,
    resolve_low_confidence_policy,
    sliding_window_boxes,
)
from baselines.vip.vip_model import VIP_SOURCE_REVISION, VIPSegmenter  # noqa: E402


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


def _source_revision(source_root: Path) -> str:
    if not (source_root / ".git").is_dir():
        return "unavailable-no-git-metadata"
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def _canonical_name(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _validate_class_order(
    dataset: str,
    groups: tuple[tuple[str, ...], ...],
) -> None:
    expected = tuple(
        _canonical_name(name) for name in DATASET_SPECS[dataset]["classes"]
    )
    actual = tuple(_canonical_name(group[0]) for group in groups)
    if actual != expected:
        raise ValueError(
            f"VIP class order for {dataset} must be {list(expected)}, got {list(actual)}"
        )


def _resize_short_side(rgb: np.ndarray, short_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = float(short_size) / float(min(height, width))
    resized_width = max(1, int(width * scale + 0.5))
    resized_height = max(1, int(height * scale + 0.5))
    return np.asarray(
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def _resize_input(
    rgb: np.ndarray,
    *,
    policy: str,
    input_size: int,
) -> np.ndarray:
    if policy == "release_max_side":
        return resize_keep_ratio(rgb, input_size)
    if policy == "paper_short_side":
        return _resize_short_side(rgb, input_size)
    raise ValueError(f"Unknown VIP resize policy: {policy}")


def _validate_resume_protocol(
    output_root: Path,
    expected: dict[str, Any],
    overwrite: bool,
) -> None:
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
            "Existing predictions use a different VIP protocol: "
            f"{json.dumps(mismatches, ensure_ascii=False)}. "
            "Use OVERWRITE=1 or a new OUTPUT_DIR."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pinned official VIP inference on LoveDA, FLAIR #1, UAVid, "
            "xBD-pre, or CHN6-CUG with Vistar-compatible metrics and WFm."
        )
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--vip_root", default=str(REPO_ROOT / "third_party" / "VIP")
    )
    parser.add_argument("--class_file", default="")
    parser.add_argument(
        "--backbone_checkpoint",
        default="/root/data/weight/vip/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    parser.add_argument(
        "--dinotxt_checkpoint",
        default=(
            "/root/data/weight/vip/"
            "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        ),
    )
    parser.add_argument(
        "--bpe_vocabulary",
        default="/root/data/weight/vip/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument(
        "--resize_policy",
        choices=("release_max_side", "paper_short_side"),
        default="release_max_side",
        help=(
            "release_max_side matches the public RS configs (448 max side); "
            "paper_short_side exposes the appendix's 336-short-side protocol"
        ),
    )
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--slide_crop", type=int, default=336)
    parser.add_argument("--slide_stride", type=int, default=112)
    parser.add_argument("--logit_scale", type=float, default=40.0)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--probability_threshold", type=float, default=0.0)
    parser.add_argument(
        "--low_confidence_action",
        choices=("auto", "background", "ignore"),
        default="auto",
        help=(
            "auto mirrors official VIP: explicit-background datasets map "
            "low confidence to background, while FLAIR emits ignore ID 255"
        ),
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--mask_id_base", choices=("auto", "zero", "one"), default="auto"
    )
    parser.add_argument(
        "--strict_protocol", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_unpinned_source", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    args = parser.parse_args()

    if not args.data_root:
        args.data_root = DEFAULT_DATA_ROOTS[args.dataset]
    if not args.class_file:
        args.class_file = str(
            Path(__file__).resolve().parent
            / "configs"
            / DEFAULT_CLASS_FILES[args.dataset]
        )
    if args.input_size <= 0 or args.slide_crop <= 0 or args.slide_stride <= 0:
        parser.error("input and sliding-window sizes must be positive")
    if args.slide_crop != 336:
        parser.error("the released VIP head requires --slide_crop 336")
    if args.max_samples < 0:
        parser.error("--max_samples cannot be negative")
    if not 0.0 <= args.probability_threshold <= 1.0:
        parser.error("--probability_threshold must lie in [0,1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must lie in [0,1]")
    if not torch.cuda.is_available():
        parser.error("VIP evaluation requires CUDA")
    return args


def main() -> None:
    args = _parse_args()
    spec = DATASET_SPECS[args.dataset]
    class_file = normalize_path(args.class_file)
    class_groups = read_vip_class_groups(class_file)
    _validate_class_order(args.dataset, class_groups)
    class_names = tuple(str(name) for name in spec["classes"])
    num_classes = len(class_names)
    palette = np.asarray(spec["palette"], dtype=np.uint8)
    ignore_index = spec["ignore_index"]
    try:
        (
            low_confidence_action,
            low_confidence_index,
            prediction_ignore_index,
        ) = resolve_low_confidence_policy(
            args.dataset,
            str(args.low_confidence_action),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    if (
        prediction_ignore_index is not None
        and float(args.probability_threshold) <= 0.0
    ):
        # Preserve the established square confusion-matrix format when the
        # threshold cannot reject any pixels.
        metric_prediction_ignore_index = None
    else:
        metric_prediction_ignore_index = prediction_ignore_index
    if (
        metric_prediction_ignore_index is not None
        and ignore_index is None
    ):
        raise ValueError(
            "Prediction abstention is unsupported for binary datasets without "
            "an ignore label; use low-confidence action 'background'"
        )

    source_root = normalize_path(args.vip_root)
    backbone_checkpoint = normalize_path(args.backbone_checkpoint)
    dinotxt_checkpoint = normalize_path(args.dinotxt_checkpoint)
    bpe_vocabulary = normalize_path(args.bpe_vocabulary)
    required = {
        "official VIP source": source_root / "dinosegmentor.py",
        "official VIP DINOv3 source": source_root / "dinov3" / "hub" / "dinotxt.py",
        "official VIP prompt templates": source_root / "prompts" / "imagenet_template.py",
        "VIP class aliases": class_file,
        "DINOv3 ViT-L/16 backbone": backbone_checkpoint,
        "dino.txt head and text encoder": dinotxt_checkpoint,
        "dino.txt BPE vocabulary": bpe_vocabulary,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing VIP source or managed weights:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/bootstrap_vip.sh and tools/check_vip_deps.py."
        )
    source_revision = _source_revision(source_root)
    if source_revision != VIP_SOURCE_REVISION and not args.allow_unpinned_source:
        raise RuntimeError(
            f"Paper evaluation requires VIP revision {VIP_SOURCE_REVISION}, "
            f"got {source_revision!r}. Run scripts/bootstrap_vip.sh."
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
        raise RuntimeError("No VIP evaluation samples were selected")
    local_records = indexed_records[rank::world_size]

    output_root = normalize_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = _make_output_dirs(output_root)
    protocol_identity = {
        "dataset_key": args.dataset,
        "method": "VIP",
        "source_revision": source_revision,
        "class_groups": [list(group) for group in class_groups],
        "resize_policy": args.resize_policy,
        "input_size": int(args.input_size),
        "slide_crop": int(args.slide_crop),
        "slide_stride": int(args.slide_stride),
        "logit_scale": float(args.logit_scale),
        "tau": float(args.tau),
        "temperature": float(args.temperature),
        "probability_threshold": float(args.probability_threshold),
        "low_confidence_action": low_confidence_action,
        "low_confidence_index": int(low_confidence_index),
        "prediction_ignore_index": metric_prediction_ignore_index,
        "backbone_checkpoint": str(backbone_checkpoint),
        "dinotxt_checkpoint": str(dinotxt_checkpoint),
        "bpe_vocabulary": str(bpe_vocabulary),
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
                "prediction_ignore_index": metric_prediction_ignore_index,
                "low_confidence_action": low_confidence_action,
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
                "model": "frozen DINOv3 ViT-L/16 + dino.txt",
                "vip_specific_checkpoint": False,
                "evaluation_adapter": "Vistar five-dataset protocol",
                "expected_source_revision": VIP_SOURCE_REVISION,
                "source_is_pinned": source_revision == VIP_SOURCE_REVISION,
                "allow_unpinned_source": bool(args.allow_unpinned_source),
                "data_root": str(data_root),
                "class_file": str(class_file),
                "strict_protocol": bool(args.strict_protocol),
                "num_discovered_samples": len(records),
                "num_selected_samples": len(indexed_records),
                "world_size": world_size,
                "save_images": bool(args.save_images),
                "primary_metric": spec["primary_metric"],
                "dataset_audit": dataset_audit,
                "custom_dataset_note": (
                    "VIP's public RS configs cover iSAID, Vaihingen, Potsdam, "
                    "and VDD. This adapter keeps the released model and "
                    "inference rule while applying the Vistar label protocols."
                ),
            },
        )

    model = VIPSegmenter(
        source_root=source_root,
        backbone_checkpoint=backbone_checkpoint,
        dinotxt_checkpoint=dinotxt_checkpoint,
        bpe_vocabulary=bpe_vocabulary,
        class_groups=class_groups,
        device=device,
        logit_scale=float(args.logit_scale),
        tau=float(args.tau),
        temperature=float(args.temperature),
        probability_threshold=float(args.probability_threshold),
        low_confidence_index=int(low_confidence_index),
        crop_size=int(args.slide_crop),
        stride=int(args.slide_stride),
    )
    confusion_columns = num_classes + int(
        metric_prediction_ignore_index is not None
    )
    local_confusion = np.zeros(
        (num_classes, confusion_columns),
        dtype=np.int64,
    )
    local_domain_confusions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        local_records, desc=f"VIP {spec['display_name']} rank {rank}", disable=False
    )
    for global_index, record in progress:
        image_rgb = load_rgb(record, args.dataset)
        height, width = image_rgb.shape[:2]
        target = load_target(record, args.dataset, height=height, width=width)
        resized = _resize_input(
            image_rgb, policy=args.resize_policy, input_size=int(args.input_size)
        )
        resized_height, resized_width = resized.shape[:2]
        windows = sliding_window_boxes(
            resized_height,
            resized_width,
            int(args.slide_crop),
            int(args.slide_stride),
        )
        pred_path = output_dirs["pred_mask"] / f"{record.name}.png"
        if pred_path.is_file() and not args.overwrite:
            prediction = _validate_cached_prediction(
                pred_path,
                expected_shape=(height, width),
                num_classes=num_classes,
                prediction_ignore_index=metric_prediction_ignore_index,
            )
            resumed = True
        else:
            prediction, observed_windows = model.predict(
                resized, output_size=(height, width)
            )
            if observed_windows != len(windows):
                raise RuntimeError("VIP sliding-window count changed unexpectedly")
            _save_id_mask(prediction, pred_path)
            resumed = False

        if args.save_images:
            _save_visualizations(
                output_dirs,
                record.name,
                image_rgb,
                prediction,
                target,
                palette,
                ignore_index=ignore_index,
                prediction_ignore_index=metric_prediction_ignore_index,
                overlay_alpha=args.overlay_alpha,
            )
        matrix = confusion_matrix(
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
            prediction_ignore_index=metric_prediction_ignore_index,
        )
        local_confusion += matrix
        if record.domain:
            local_domain_confusions.setdefault(
                record.domain, np.zeros_like(local_confusion)
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
                allow_prediction_ignore=(
                    metric_prediction_ignore_index is not None
                ),
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
                "model_input_height": int(resized_height),
                "model_input_width": int(resized_width),
                "num_slide_windows": len(windows),
                "num_rejected_pixels": int(
                    np.sum(prediction == int(prediction_ignore_index))
                )
                if prediction_ignore_index is not None
                else 0,
                "resumed_from_saved_prediction": resumed,
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
            payload = json.loads(
                (output_root / f"rank_{process_rank:05d}.json").read_text(
                    encoding="utf-8"
                )
            )
            merged_confusion += np.asarray(payload["confusion_matrix"], dtype=np.int64)
            for domain, values in payload["domain_confusions"].items():
                merged_domain_confusions.setdefault(
                    domain, np.zeros_like(merged_confusion)
                )
                merged_domain_confusions[domain] += np.asarray(values, dtype=np.int64)
            merged_rows.extend(payload["rows"])
        merged_rows.sort(key=lambda row: int(row["global_index"]))
        if len(merged_rows) != len(indexed_records):
            raise RuntimeError(
                f"Distributed result count mismatch: expected {len(indexed_records)}, "
                f"found {len(merged_rows)}"
            )
        expected_num_samples = int(dataset_audit["expected_num_samples"])
        result = {
            "dataset": spec["display_name"],
            "dataset_key": args.dataset,
            "split": spec["split"],
            "method": "VIP",
            "training_setting": "training-free",
            "model": "frozen DINOv3 ViT-L/16 + dino.txt",
            "vip_specific_checkpoint": False,
            "source_revision": source_revision,
            "expected_source_revision": VIP_SOURCE_REVISION,
            "source_is_pinned": source_revision == VIP_SOURCE_REVISION,
            "num_samples": len(merged_rows),
            "expected_num_samples": expected_num_samples,
            "complete_protocol_population": len(merged_rows) == expected_num_samples,
            "num_classes": num_classes,
            "classes": list(class_names),
            "class_groups": [list(group) for group in class_groups],
            "label_protocol": spec["label_protocol"],
            "inference_mode": "VIP overlapping sliding-window inference",
            "resize_policy": args.resize_policy,
            "input_size": int(args.input_size),
            "slide_crop": int(args.slide_crop),
            "slide_stride": int(args.slide_stride),
            "metric_size": "original",
            "num_slide_windows": sum(
                int(row["num_slide_windows"]) for row in merged_rows
            ),
            "logit_scale": float(args.logit_scale),
            "tau": float(args.tau),
            "temperature": float(args.temperature),
            "probability_threshold": float(args.probability_threshold),
            "low_confidence_action": low_confidence_action,
            "low_confidence_index": int(low_confidence_index),
            "prediction_ignore_index": metric_prediction_ignore_index,
            "test_time_augmentation": "none",
            "world_size": world_size,
            "save_pred_mask": True,
            "save_visualizations": bool(args.save_images),
            "primary_metric": spec["primary_metric"],
            "load_diagnostics": model.load_diagnostics.as_dict(),
            "confusion_matrix": merged_confusion.tolist(),
            "domain_metrics": {
                domain: {
                    "num_samples": sum(
                        1 for row in merged_rows if row["domain"] == domain
                    ),
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
        print(f"[eval_vip] saved outputs to: {output_root}")

    _barrier(world_size)
    del model
    torch.cuda.empty_cache()
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
