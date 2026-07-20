from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.levircd_common import (  # noqa: E402
    binary_counts,
    binary_metrics,
    discover_samples,
    load_binary,
    load_rgb,
    make_output_dirs,
    resolve_path,
    save_artifacts,
    write_json,
    write_jsonl,
)


def _distributed_context(device_arg: str) -> tuple[int, int, int, torch.device, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DynamicEarth M-C-I evaluation requires CUDA, but CUDA is unavailable."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device(device_arg)
        backend = "gloo"
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
        initialized_here = True
    return rank, world_size, local_rank, device, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def _configure_imports(dynamic_root: Path) -> None:
    candidates = (
        dynamic_root,
        dynamic_root / "third_party" / "segment_anything",
        dynamic_root / "third_party" / "SegEarth_OV",
        dynamic_root / "third_party" / "SimFeatUp",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _build_pipeline(
    dynamic_root: Path,
    sam_checkpoint: Path,
    segearth_weight: Path,
    device: torch.device,
    class_file: Path,
    *,
    feature_up: bool,
    change_threshold: float,
):
    _configure_imports(dynamic_root)
    try:
        from segment_anything import sam_model_registry
        from segment_anything.utils.amg import rle_to_mask
        from torchvision import transforms

        from dynamic_earth.comparator.bi_match import bitemporal_match
        from dynamic_earth.identifier.segearth_ov_ext import SegEarth_OV
        from dynamic_earth.identifier.utils import identify
        from dynamic_earth.sam_ext import MaskProposal
        from dynamic_earth.utils import get_model_and_processor
    except Exception as exc:
        raise RuntimeError(
            "DynamicEarth imports failed. Install requirements-dynamicearth.txt, "
            "run scripts/bootstrap_dynamicearth.sh, and verify with "
            "tools/check_dynamicearth_deps.py."
        ) from exc

    class_file.parent.mkdir(parents=True, exist_ok=True)
    class_file.write_text("background\nbuilding\n", encoding="utf-8")

    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint)).to(device)
    sam.eval()
    mask_proposal = MaskProposal()
    mask_proposal.make_mask_generator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.95,
        stability_score_offset=0.9,
        box_nms_thresh=0.7,
        min_mask_region_area=0,
    )
    mask_proposal.set_hyperparameters()

    comparator_model, comparator_processor = get_model_and_processor(
        "DINO", str(device)
    )
    comparator_model.eval()

    identifier = SegEarth_OV(
        clip_type="CLIP",
        vit_type="ViT-B/16",
        model_type="SegEarth",
        name_path=str(class_file),
        device=device,
        ignore_residual=True,
        feature_up=feature_up,
        feature_up_cfg={
            "model_name": "jbu_one",
            "model_path": str(segearth_weight),
        },
        cls_token_lambda=-0.3,
        prob_thd=0.0,
        logit_scale=50,
        slide_stride=112,
        slide_crop=224,
        bg_idx=0,
    )
    identifier.eval()
    identifier_processor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
            transforms.Resize((448, 448)),
        ]
    )
    return {
        "sam": sam,
        "mask_proposal": mask_proposal,
        "comparator_model": comparator_model,
        "comparator_processor": comparator_processor,
        "identifier": identifier,
        "identifier_processor": identifier_processor,
        "rle_to_mask": rle_to_mask,
        "bitemporal_match": bitemporal_match,
        "identify": identify,
        "change_threshold": float(change_threshold),
    }


@torch.inference_mode()
def _predict(
    pipeline: dict[str, Any],
    image_a: np.ndarray,
    image_b: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_a.shape != image_b.shape:
        raise ValueError(f"A/B shape mismatch: {image_a.shape} vs {image_b.shape}")
    mask_data, num_a = pipeline["mask_proposal"].forward(image_a, image_b)
    proposals = np.asarray(
        [pipeline["rle_to_mask"](rle).astype(bool) for rle in mask_data["rles"]]
    )
    num_proposals = len(proposals)

    if num_proposals:
        changed, changed_num_a = pipeline["bitemporal_match"](
            image_a,
            image_b,
            proposals,
            pipeline["comparator_model"],
            pipeline["comparator_processor"],
            num_a,
            change_confidence_threshold=pipeline["change_threshold"],
            device=str(device),
            model_config={
                "model_type": "DINO",
                "feature_dim": 768,
                "patch_size": 16,
            },
        )
    else:
        changed = proposals
        changed_num_a = 0
    num_after_comparator = len(changed)

    if num_after_comparator:
        identified, identified_num_a = pipeline["identify"](
            image_a,
            image_b,
            changed,
            changed_num_a,
            pipeline["identifier"],
            pipeline["identifier_processor"],
            model_type="SegEarth-OV",
            device=str(device),
            is_instance_class=True,
        )
    else:
        identified = changed
        identified_num_a = 0
    prediction = np.zeros(image_a.shape[:2], dtype=np.uint8)
    for mask in identified:
        prediction[np.asarray(mask, dtype=bool)] = 1
    return prediction, {
        "num_proposals": num_proposals,
        "num_proposals_from_A": int(num_a),
        "num_after_comparator": num_after_comparator,
        "num_after_comparator_from_A": int(changed_num_a),
        "num_after_identifier": int(len(identified)),
        "num_after_identifier_from_A": int(identified_num_a),
    }


def _gather_rows_and_counts(
    rows: list[dict[str, Any]],
    device: torch.device,
    world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    local_counts = [sum(int(row[key]) for row in rows) for key in ("tp", "tn", "fp", "fn")]
    tensor = torch.tensor(local_counts, dtype=torch.int64, device=device)
    if world_size > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, rows)
        merged = [row for rank_rows in gathered if rank_rows for row in rank_rows]
    else:
        merged = rows
    counts = dict(zip(("tp", "tn", "fp", "fn"), map(int, tensor.cpu().tolist())))
    return sorted(merged, key=lambda row: row["name"]), counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run official DynamicEarth M-C-I (SAM + DINO + SegEarth-OV) "
            "on official LEVIR-CD A/B/label data."
        )
    )
    parser.add_argument("--data_root", default="/root/data/LEVIR-CD")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dynamic_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segearth_weight", required=True)
    parser.add_argument("--change_threshold", type=float, default=145.0)
    parser.add_argument("--feature_up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.change_threshold <= 180.0:
        raise ValueError("change_threshold must be in [0, 180]")
    dynamic_root = resolve_path(args.dynamic_root)
    sam_checkpoint = resolve_path(args.sam_checkpoint)
    segearth_weight = resolve_path(args.segearth_weight)
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(f"Missing SAM checkpoint: {sam_checkpoint}")
    if args.feature_up and not segearth_weight.is_file():
        raise FileNotFoundError(f"Missing SegEarth-OV upsampler: {segearth_weight}")

    rank, world_size, _, device, initialized_here = _distributed_context(args.device)
    output_dirs = make_output_dirs(args.output_dir)
    split_root, samples = discover_samples(args.data_root, args.split, args.max_samples)
    pipeline = _build_pipeline(
        dynamic_root,
        sam_checkpoint,
        segearth_weight,
        device,
        output_dirs["runtime"] / f"rank_{rank}" / "class_names.txt",
        feature_up=args.feature_up,
        change_threshold=args.change_threshold,
    )

    local_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        samples[rank::world_size],
        desc=f"DynamicEarth LEVIR-CD rank {rank}",
        disable=rank != 0,
    )
    for sample in iterator:
        prediction_path = (
            output_dirs["pred_mask"] / f"{sample.stem}_pred_mask.png"
        )
        image_a = load_rgb(sample.image_a)
        image_b = load_rgb(sample.image_b)
        target = load_binary(sample.label)
        if prediction_path.is_file() and not args.overwrite:
            prediction = load_binary(prediction_path)
            prediction_meta = {"resumed_from_saved_prediction": True}
        else:
            prediction, prediction_meta = _predict(
                pipeline, image_a, image_b, device
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target mismatch for {sample.stem}: "
                f"{prediction.shape} vs {target.shape}"
            )
        counts = binary_counts(prediction, target)
        row = {
            "name": sample.stem,
            "image_a": str(sample.image_a),
            "image_b": str(sample.image_b),
            "label": str(sample.label),
            **counts,
            **binary_metrics(counts),
            **prediction_meta,
        }
        local_rows.append(row)
        save_artifacts(
            output_dirs,
            sample.stem,
            image_a,
            image_b,
            target,
            prediction,
            save_images=args.save_images,
        )

    rows, counts = _gather_rows_and_counts(local_rows, device, world_size)
    _barrier(world_size)
    if rank == 0:
        summary = {
            "method": "DynamicEarth",
            "variant": "M-C-I: SAM ViT-H + DINO ViT-B/16 + SegEarth-OV CLIP ViT-B/16",
            "dataset": "LEVIR-CD",
            "split": args.split,
            "split_root": str(split_root),
            "num_dataset_pairs": len(samples),
            "num_evaluated_pairs": len(rows),
            "protocol": "official DynamicEarth full-native-resolution M-C-I inference",
            "query": "building",
            "prediction_rule": "union of building-change proposal masks",
            "aggregation": "global pixel TP/TN/FP/FN",
            "metric_equations": "OpenDPR/DynamicEarth binary equations",
            "official_source": "https://github.com/likyoo/DynamicEarth",
            "official_source_revision": "c9ffd90cafbd791cd75a48a5717a902966c2436c",
            "sam_checkpoint": str(sam_checkpoint),
            "segearth_weight": str(segearth_weight),
            "settings": {
                "sam_points_per_side": 32,
                "sam_points_per_batch": 64,
                "sam_pred_iou_threshold": 0.5,
                "sam_stability_score_threshold": 0.95,
                "sam_stability_score_offset": 0.9,
                "sam_box_nms_threshold": 0.7,
                "change_angle_threshold_degrees": args.change_threshold,
                "identifier_resize": [448, 448],
                "feature_up": bool(args.feature_up),
                "cls_token_lambda": -0.3,
                "probability_threshold": 0.0,
            },
            "world_size": world_size,
            "save_images": bool(args.save_images),
            **counts,
            **binary_metrics(counts),
        }
        write_json(output_dirs["root"] / "metrics.json", summary)
        write_jsonl(output_dirs["root"] / "per_image_metrics.jsonl", rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[DynamicEarth] saved: {output_dirs['root']}")

    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
