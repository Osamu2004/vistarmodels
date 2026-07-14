from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from heterogeneous_batch import generate_fixed_mask_batch


LOVEDA_RAW_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
    ],
    dtype=np.int16,
)
BATCH_PROTOCOL = "syntheticgen_fixed_mask_heterogeneous_batch_v1"


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def source_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sample_name(path: Path) -> str:
    suffix = "_cond_mask.png"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected condition filename: {path.name}")
    return path.name[: -len(suffix)]


def valid_rgb(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == (size, size)
    except Exception:
        return False


def decode_loveda_raw(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path)
    if image.mode in {"L", "P"}:
        raw = np.asarray(image.convert("L"), dtype=np.int16)
        if raw.size and raw.min() >= 0 and raw.max() <= 7:
            ids = raw if raw.max() == 7 else raw + 1
            resized = Image.fromarray(ids.astype(np.uint8), mode="L").resize(
                (size, size), Image.Resampling.NEAREST
            )
            return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).long()
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    delta = rgb.astype(np.int32)[:, :, None, :] - LOVEDA_RAW_PALETTE.astype(np.int32)[None, None, :, :]
    ids = np.argmin((delta * delta).sum(axis=3), axis=2).astype(np.uint8) + 1
    resized = Image.fromarray(ids, mode="L").resize((size, size), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).long()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach().cpu() / 2.0 + 0.5).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def copy_resize(source: Path, target: Path, size: int, nearest: bool = False) -> None:
    image = Image.open(source).convert("RGB")
    if image.size != (size, size):
        resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        image = image.resize((size, size), resampling)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def seed_for(base_seed: int, name: str) -> int:
    digest = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")
    return int(base_seed) + digest


def discover_checkpoint(root: Path, kind: str) -> Path:
    if kind == "controlnet":
        candidates = [path.parent for path in root.rglob("ratio_projector.bin")]
        candidates += [
            path.parent
            for path in root.rglob("model_1.safetensors")
            if "layout" not in str(path).lower()
        ]
    else:
        candidates = [path.parent for path in root.rglob("d3pm_config.json")]
        candidates += [
            path.parent
            for path in root.rglob("model.safetensors")
            if "layout" in str(path).lower()
        ]
    unique = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))
    if not unique:
        raise FileNotFoundError(f"cannot discover {kind} checkpoint below {root}")
    return unique[0]


def build_items(eval_dir: Path, max_samples: int, prompt_template: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for mask_path in sorted((eval_dir / "cond_mask").glob("*_cond_mask.png")):
        name = sample_name(mask_path)
        gt_path = eval_dir / "gt_rgb" / f"{name}_gt_rgb.png"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        domain = "rural" if "rural" in name.lower() else "urban"
        items.append(
            {
                "name": name,
                "mask": mask_path,
                "target": gt_path,
                "domain": domain,
                "prompt": prompt_template.format(domain=domain),
            }
        )
        if max_samples > 0 and len(items) >= max_samples:
            break
    if not items:
        raise RuntimeError(f"no LoveDA conditions found under {eval_dir / 'cond_mask'}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SyntheticGen with different fixed LoveDA masks in each batch slot."
    )
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--official_dir", required=True)
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--layout_ckpt", default="")
    parser.add_argument("--controlnet_ckpt", default="")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--layout_size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prompt_template",
        default="A high-resolution satellite image of a {domain} area",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("SyntheticGen heterogeneous batching requires CUDA")

    official = resolve(args.official_dir)
    sample_pair_path = official / "src/scripts/sample_pair.py"
    if not sample_pair_path.is_file():
        raise FileNotFoundError(sample_pair_path)
    sys.path.insert(0, str(official))
    from src.scripts import sample_pair as upstream  # type: ignore
    upstream_commit = source_commit(official)

    weights = resolve(args.weight_dir)
    layout_ckpt = resolve(args.layout_ckpt) if args.layout_ckpt else discover_checkpoint(weights, "layout")
    controlnet_ckpt = (
        resolve(args.controlnet_ckpt)
        if args.controlnet_ckpt
        else discover_checkpoint(weights, "controlnet")
    )
    output = resolve(args.output_dir)
    eval_dir = resolve(args.eval_dir)
    dirs = {
        name: output / name
        for name in (
            "pred_rgb",
            "pred_rgb_native",
            "cond_mask",
            "cond_mask_ids",
            "gt_rgb",
            "absdiff",
            "prompts",
            "metadata",
        )
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    items = build_items(eval_dir, args.max_samples, args.prompt_template)
    pending = [
        item
        for item in items
        if args.overwrite
        or not valid_rgb(dirs["pred_rgb"] / f"{item['name']}_pred_rgb.png", args.eval_size)
    ]
    print(
        f"[syntheticgen_heterogeneous_batch] total={len(items)} pending={len(pending)} "
        f"batch_size={args.batch_size} layout={args.layout_size} resolution={args.resolution}"
    )
    print(f"[syntheticgen_heterogeneous_batch] layout_ckpt={layout_ckpt}")
    print(f"[syntheticgen_heterogeneous_batch] controlnet_ckpt={controlnet_ckpt}")
    print(f"[syntheticgen_heterogeneous_batch] upstream_commit={upstream_commit}")
    if not pending:
        print("[syntheticgen_heterogeneous_batch] all predictions already exist")
        return

    context_args = upstream.parse_args(
        [
            "--layout_ckpt", str(layout_ckpt),
            "--layout_size", str(args.layout_size),
            "--layout_diffusion_type", "d3pm",
            "--controlnet_ckpt", str(controlnet_ckpt),
            "--base_model", str(resolve(args.base_model)),
            "--save_dir", str(output / ".heterogeneous_batch_runtime"),
            "--image_size", str(args.resolution),
            "--batch_size", str(args.batch_size),
            "--num_inference_steps_layout", "1",
            "--num_inference_steps_image", str(args.steps),
            "--strength_layout", "0",
            "--guidance_scale", str(args.guidance_scale),
            "--guidance_rescale", str(args.guidance_rescale),
            "--control_scale", str(args.control_scale),
            "--dtype", args.dtype,
            "--device", args.device,
            "--sampler", "ddim",
        ]
    )
    ctx = upstream.build_context(context_args)
    ctx.layout_unet.to("cpu")
    ctx.layout_ratio_projector.to("cpu")
    if ctx.domain_embed is not None:
        ctx.domain_embed.to("cpu")
    torch.cuda.empty_cache()
    print(
        "[syntheticgen_heterogeneous_batch] offloaded unused Stage-A layout modules; "
        "fixed masks go directly to the released Stage-B renderer"
    )

    batch_starts = range(0, len(pending), args.batch_size)
    progress = tqdm(batch_starts, total=(len(pending) + args.batch_size - 1) // args.batch_size, desc="SyntheticGen batches")
    generated_count = 0
    for start in progress:
        real_items = pending[start : start + args.batch_size]
        padded_items = list(real_items)
        while len(padded_items) < args.batch_size:
            padded_items.append(real_items[-1])

        raw_masks = [decode_loveda_raw(item["mask"], args.layout_size) for item in padded_items]
        prompts = [str(item["prompt"]) for item in padded_items]
        seeds = [seed_for(args.seed, str(item["name"])) for item in padded_items]
        result = generate_fixed_mask_batch(
            ctx,
            upstream,
            raw_masks=raw_masks,
            prompts=prompts,
            seeds=seeds,
            image_size=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_rescale=args.guidance_rescale,
            control_scale=args.control_scale,
        )

        for batch_index, item in enumerate(real_items):
            name = str(item["name"])
            native = tensor_to_pil(result.images[batch_index])
            native_path = dirs["pred_rgb_native"] / f"{name}_pred_rgb.png"
            pred_path = dirs["pred_rgb"] / f"{name}_pred_rgb.png"
            native.save(native_path)
            native.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC).save(pred_path)
            copy_resize(item["mask"], dirs["cond_mask"] / f"{name}_cond_mask.png", args.eval_size, nearest=True)
            raw_eval = decode_loveda_raw(item["mask"], args.eval_size)
            ids_eval = torch.where(raw_eval > 0, raw_eval - 1, torch.full_like(raw_eval, 255))
            Image.fromarray(ids_eval.numpy().astype(np.uint8), mode="L").save(
                dirs["cond_mask_ids"] / f"{name}_cond_mask_ids.png"
            )
            copy_resize(item["target"], dirs["gt_rgb"] / f"{name}_gt_rgb.png", args.eval_size)
            gt = np.asarray(Image.open(dirs["gt_rgb"] / f"{name}_gt_rgb.png").convert("RGB"), dtype=np.int16)
            pred = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.int16)
            Image.fromarray(np.abs(gt - pred).astype(np.uint8), mode="RGB").save(
                dirs["absdiff"] / f"{name}_absdiff.png"
            )
            (dirs["prompts"] / f"{name}.txt").write_text(str(item["prompt"]) + "\n", encoding="utf-8")
            metadata = {
                "name": name,
                "domain": item["domain"],
                "prompt": item["prompt"],
                "seed": seeds[batch_index],
                "batch_size": args.batch_size,
                "batch_slot": batch_index,
                "heterogeneous_fixed_mask_batch": True,
                "batch_protocol": BATCH_PROTOCOL,
                "syntheticgen_upstream_commit": upstream_commit,
                "condition_image": str(resolve(str(item["mask"]))),
                "target_image": str(resolve(str(item["target"]))),
                "prediction": str(pred_path),
                "ratios_generated": result.ratios[batch_index].cpu().tolist(),
            }
            (dirs["metadata"] / f"{name}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            generated_count += 1
        progress.set_postfix(generated=generated_count, remaining=len(pending) - generated_count)

    print(f"[syntheticgen_heterogeneous_batch] generated={generated_count} output={output}")


if __name__ == "__main__":
    main()
