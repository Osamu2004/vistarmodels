from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
    pad_tile,
    resolve_path,
    save_artifacts,
    tile_coordinates,
    write_json,
    write_jsonl,
)


SECOND_MEAN = np.asarray([0.439, 0.447, 0.459], dtype=np.float32)
SECOND_STD = np.asarray([0.193, 0.183, 0.189], dtype=np.float32)


def _distributed_context(device_arg: str) -> tuple[int, int, int, torch.device, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("RCDNet evaluation requires CUDA, but CUDA is unavailable.")
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


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.casefold() == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must decode to a dict: {path}")
    for key in ("model", "state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in ("module.", "model.", "network."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = value
    if not normalized:
        raise RuntimeError(f"No tensor weights found in checkpoint: {path}")
    return normalized


def _build_model(
    rcdnet_root: Path,
    checkpoint: Path,
    model_input_size: int,
    device: torch.device,
    allow_partial_checkpoint: bool,
):
    source_root = rcdnet_root / "RCDNet"
    if not (source_root / "models" / "builder.py").is_file():
        raise FileNotFoundError(f"Missing official RCDNet source under {source_root}")
    sys.path.insert(0, str(source_root))

    import torch.nn as nn
    from models.builder import EncoderDecoder

    config_module = importlib.import_module("configs.config_second")
    cfg = copy.deepcopy(config_module.config)
    cfg.image_height = int(model_input_size)
    cfg.image_width = int(model_input_size)
    cfg.use_imagenet_pretrain = False
    cfg.pretrained_model = None
    cfg.freeze_backbone = False

    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d)
    state = _load_checkpoint(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key for key in incompatible.missing_keys if "num_batches_tracked" not in key
    ]
    unexpected = list(incompatible.unexpected_keys)
    if (missing or unexpected) and not allow_partial_checkpoint:
        raise RuntimeError(
            "RCDNet checkpoint/model mismatch. "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    if missing or unexpected:
        print(
            "[RCDNet] warning: partial checkpoint load "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
    model.to(device).eval()
    return model, cfg, missing, unexpected


def _build_text_condition(model_id: str, prompt: str, device: torch.device):
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    inputs = tokenizer(
        [prompt.casefold()],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        embedding = text_encoder(**inputs).last_hidden_state
    return text_encoder, embedding


def _preprocess(image: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    if image.shape[:2] != (size, size):
        image = np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (size, size), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    array = image.astype(np.float32) / 255.0
    array = (array - SECOND_MEAN[None, None, :]) / SECOND_STD[None, None, :]
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def _predict(
    model,
    image_a: np.ndarray,
    image_b: np.ndarray,
    text_embedding: torch.Tensor,
    *,
    tile_size: int,
    model_input_size: int,
    threshold: float,
    device: torch.device,
    amp: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image_a.shape != image_b.shape:
        raise ValueError(f"A/B shape mismatch: {image_a.shape} vs {image_b.shape}")
    height, width = image_a.shape[:2]
    probabilities = np.zeros((height, width), dtype=np.float32)
    coordinates = tile_coordinates(height, width, tile_size)

    for top, left in coordinates:
        bottom = min(top + tile_size, height)
        right = min(left + tile_size, width)
        crop_h, crop_w = bottom - top, right - left
        tile_a = pad_tile(image_a[top:bottom, left:right], tile_size, tile_size)
        tile_b = pad_tile(image_b[top:bottom, left:right], tile_size, tile_size)
        tensor_a = _preprocess(tile_a, model_input_size, device)
        tensor_b = _preprocess(tile_b, model_input_size, device)
        with _autocast_context(device, amp):
            logits = model(tensor_a, tensor_b, captions=text_embedding)
            probability = logits.sigmoid()
        if probability.shape[-2:] != (tile_size, tile_size):
            probability = F.interpolate(
                probability.float(),
                size=(tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            )
        tile_probability = probability[0, 0, :crop_h, :crop_w].float().cpu().numpy()
        probabilities[top:bottom, left:right] = tile_probability

    prediction = (probabilities > float(threshold)).astype(np.uint8)
    return prediction, {
        "num_tiles": len(coordinates),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
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
        description="Run the official RCDNet architecture on official LEVIR-CD A/B/label data."
    )
    parser.add_argument("--data_root", default="/root/data/LEVIR-CD")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rcdnet_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt", default="building")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--model_input_size", type=int, default=512)
    parser.add_argument("--amp", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_partial_checkpoint", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {args.threshold}")
    if args.tile_size <= 0 or args.model_input_size <= 0:
        raise ValueError("tile_size and model_input_size must be positive")

    rank, world_size, _, device, initialized_here = _distributed_context(args.device)
    output_dirs = make_output_dirs(args.output_dir)
    split_root, samples = discover_samples(args.data_root, args.split, args.max_samples)
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing RCDNet checkpoint: {checkpoint}")

    model, cfg, missing, unexpected = _build_model(
        resolve_path(args.rcdnet_root),
        checkpoint,
        args.model_input_size,
        device,
        args.allow_partial_checkpoint,
    )
    text_encoder, text_embedding = _build_text_condition(
        args.clip_model, args.prompt, device
    )

    local_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        samples[rank::world_size],
        desc=f"RCDNet LEVIR-CD rank {rank}",
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
                model,
                image_a,
                image_b,
                text_embedding,
                tile_size=args.tile_size,
                model_input_size=args.model_input_size,
                threshold=args.threshold,
                device=device,
                amp=args.amp,
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
            "method": "RCDNet",
            "dataset": "LEVIR-CD",
            "split": args.split,
            "split_root": str(split_root),
            "num_dataset_pairs": len(samples),
            "num_evaluated_pairs": len(rows),
            "protocol": "native-extent non-overlapping tiled RCDNet inference",
            "query": args.prompt,
            "prediction_rule": f"sigmoid(logit) > {args.threshold}",
            "aggregation": "global pixel TP/TN/FP/FN",
            "metric_equations": "OpenDPR/DynamicEarth binary equations",
            "checkpoint": str(checkpoint),
            "checkpoint_release": (
                "official SECOND real-data checkpoint; the paper's synthetic-pretrained "
                "cross-domain checkpoint is not publicly released as of 2026-07-21"
            ),
            "official_source": "https://github.com/yilmazkorkmaz1/referring_change_detection",
            "official_source_revision": "0966e96ff7075476d77442bbf6623ed5086d52da",
            "architecture": {
                "config": "configs.config_second",
                "backbone": cfg.backbone,
                "decoder": cfg.decoder,
                "clip_model": args.clip_model,
                "normalization": {
                    "mean": SECOND_MEAN.tolist(),
                    "std": SECOND_STD.tolist(),
                    "source": "official RCDNet SECOND config",
                },
                "tile_size": args.tile_size,
                "model_input_size": args.model_input_size,
                "amp": args.amp,
            },
            "checkpoint_load": {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            },
            "world_size": world_size,
            "save_images": bool(args.save_images),
            **counts,
            **binary_metrics(counts),
        }
        write_json(output_dirs["root"] / "metrics.json", summary)
        write_jsonl(output_dirs["root"] / "per_image_metrics.jsonl", rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[RCDNet] saved: {output_dirs['root']}")

    del text_encoder, text_embedding, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if initialized_here:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
