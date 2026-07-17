from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from tqdm import tqdm

from common import binary_mask, colorize_mask, controlnet_prompt, load_jsonl, load_mask_ids, load_rgb


def init_runtime() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("ControlNet SECOND inference requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but the selected GPU does not support it")
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def sample_seed(base_seed: int, name: str) -> int:
    digest = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")
    return (base_seed + digest) % (2**63 - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained SD1.5 ControlNet on bidirectional SECOND source images and masks."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--controlnet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument(
        "--pipeline_mode",
        choices=("source_img2img", "mask_text2img"),
        default="source_img2img",
        help=(
            "source_img2img conditions on source image + change mask + text; "
            "mask_text2img preserves the legacy change-mask + text protocol"
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Source-image img2img strength; used only when pipeline_mode=source_img2img.",
    )
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.strength <= 1.0:
        raise ValueError(f"--strength must be in (0, 1], got {args.strength}")
    rank, local_rank, world_size, device = init_runtime()
    dtype = dtype_from_name(args.dtype)
    manifest = Path(args.manifest).expanduser().resolve()
    base_model = Path(args.base_model).expanduser().resolve()
    controlnet_path = Path(args.controlnet).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    rows = load_jsonl(manifest)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    rank_rows = rows[rank::world_size]
    for folder in (
        "source_rgb",
        "cond_mask",
        "cond_mask_official",
        "cond_mask_ids",
        "gt_rgb",
        "pred_rgb",
        "absdiff",
        "prompts",
    ):
        (output / folder).mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype, local_files_only=True)
    pipeline_cls = (
        StableDiffusionControlNetImg2ImgPipeline
        if args.pipeline_mode == "source_img2img"
        else StableDiffusionControlNetPipeline
    )
    pipe = pipeline_cls.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    pending = []
    for row in rank_rows:
        name = str(row["name"])
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        if args.overwrite or not pred_path.is_file():
            pending.append(row)
    iterator = range(0, len(pending), args.batch_size)
    for start in tqdm(iterator, desc=f"ControlNet SECOND rank {rank}", disable=rank != 0):
        batch_rows = pending[start : start + args.batch_size]
        names = [str(row["name"]) for row in batch_rows]
        mask_ids = [load_mask_ids(row, args.resolution) for row in batch_rows]
        conditions = [colorize_mask(ids) for ids in mask_ids]
        sources = [load_rgb(row, "source_image", args.resolution) for row in batch_rows]
        prompts = [controlnet_prompt(row, args.resolution) for row in batch_rows]
        generators = [torch.Generator(device=device).manual_seed(sample_seed(args.seed, name)) for name in names]
        negative_prompt = [args.negative_prompt] * len(batch_rows) if args.negative_prompt else None
        with torch.inference_mode():
            common_kwargs = {
                "prompt": prompts,
                "negative_prompt": negative_prompt,
                "generator": generators,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "controlnet_conditioning_scale": args.controlnet_scale,
                "height": args.resolution,
                "width": args.resolution,
            }
            if args.pipeline_mode == "source_img2img":
                predictions = pipe(
                    image=sources,
                    control_image=conditions,
                    strength=args.strength,
                    **common_kwargs,
                ).images
            else:
                predictions = pipe(
                    image=conditions,
                    **common_kwargs,
                ).images
        for row, name, ids, condition, source, prompt, prediction in zip(
            batch_rows, names, mask_ids, conditions, sources, prompts, predictions
        ):
            target = load_rgb(row, "target_image", args.resolution)
            pred = prediction.convert("RGB")
            if pred.size != (args.resolution, args.resolution):
                pred = pred.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
            source.save(output / "source_rgb" / f"{name}_source_rgb.png")
            target.save(output / "gt_rgb" / f"{name}_gt_rgb.png")
            Image.fromarray(ids, mode="L").save(output / "cond_mask_ids" / f"{name}_cond_mask_ids.png")
            condition.save(output / "cond_mask_official" / f"{name}_cond_mask_official.png")
            binary_mask(ids).save(output / "cond_mask" / f"{name}_cond_mask.png")
            pred.save(output / "pred_rgb" / f"{name}_pred_rgb.png")
            gt_array = np.asarray(target, dtype=np.int16)
            pred_array = np.asarray(pred, dtype=np.int16)
            Image.fromarray(np.abs(gt_array - pred_array).astype(np.uint8), mode="RGB").save(
                output / "absdiff" / f"{name}_absdiff.png"
            )
            (output / "prompts" / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        records = []
        missing = []
        for row in rows:
            name = str(row["name"])
            pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
            if not pred_path.is_file():
                missing.append(name)
                continue
            records.append(
                {
                    "name": name,
                    "direction": row.get("direction"),
                    "prompt": controlnet_prompt(row, args.resolution),
                    "seed": sample_seed(args.seed, name),
                    "pipeline_mode": args.pipeline_mode,
                    "source_image_conditioned": args.pipeline_mode == "source_img2img",
                    "strength": args.strength if args.pipeline_mode == "source_img2img" else None,
                    "prediction": str(pred_path),
                }
            )
        with (output / "generated_samples.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        config = {
            **vars(args),
            "manifest": str(manifest),
            "base_model": str(base_model),
            "controlnet": str(controlnet_path),
            "output_dir": str(output),
            "world_size": world_size,
            "requested_samples": len(rows),
            "completed_samples": len(records),
            "missing_samples": missing[:20],
            "source_image_conditioned": args.pipeline_mode == "source_img2img",
            "condition_contract": (
                "source-time SECOND image + target-side directional SECOND semantic change mask "
                "+ class-aware text"
                if args.pipeline_mode == "source_img2img"
                else "target-side directional SECOND semantic change mask + class-aware text"
            ),
        }
        (output / "inference_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if missing:
            raise RuntimeError(f"missing {len(missing)} predictions; first entries: {missing[:5]}")
        print(json.dumps(config, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
