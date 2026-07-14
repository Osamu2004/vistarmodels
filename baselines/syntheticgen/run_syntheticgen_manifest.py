from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def sample_name(path: Path) -> str:
    suffix = "_cond_mask.png"
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[: -len(suffix)]


def palette_to_ids(image: Image.Image) -> np.ndarray:
    palette = np.asarray(
        [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
         [159, 129, 183], [0, 255, 0], [255, 195, 128]],
        dtype=np.int16,
    )
    rgb = np.asarray(image.convert("RGB"), dtype=np.int32)
    palette = palette.astype(np.int32)
    distance = ((rgb[..., None, :] - palette[None, None, ...]) ** 2).sum(axis=-1)
    return distance.argmin(axis=-1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run released SyntheticGen weights on saved LoveDA masks.")
    parser.add_argument("--syntheticgen_root", required=True)
    parser.add_argument("--eval_dir", default="")
    parser.add_argument("--manifest", default="", help="Common LoveDA JSONL with name/image/mask_rgb fields.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layout_ckpt", required=True)
    parser.add_argument("--controlnet_ckpt", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    upstream = resolve(args.syntheticgen_root)
    sys.path.insert(0, str(upstream))
    from src.scripts.sample_pair import build_context, generate_with_context, parse_args

    if not args.eval_dir and not args.manifest:
        parser.error("one of --eval_dir or --manifest is required")
    eval_dir = resolve(args.eval_dir) if args.eval_dir else None
    output = resolve(args.output_dir)
    for folder in ("cond_mask", "gt_rgb", "pred_rgb", "absdiff", "prompts", "mask_ids"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    model_args = parse_args([
        "--layout_ckpt", str(resolve(args.layout_ckpt)),
        "--layout_size", "256",
        "--layout_diffusion_type", "d3pm",
        "--controlnet_ckpt", str(resolve(args.controlnet_ckpt)),
        "--base_model", str(resolve(args.base_model)),
        "--save_dir", str(output / ".working"),
        "--image_size", "512",
        "--batch_size", "1",
        "--num_inference_steps_layout", "1",
        "--num_inference_steps_image", str(args.num_inference_steps),
        "--strength_layout", "0",
        "--guidance_scale", str(args.guidance_scale),
        "--control_scale", str(args.control_scale),
        "--dtype", args.dtype,
        "--device", "cuda",
        "--sampler", "ddim",
    ])
    context = build_context(model_args)
    if args.manifest:
        rows = [json.loads(line) for line in resolve(args.manifest).read_text().splitlines() if line.strip()]
        items = [(str(row["name"]), resolve(row["mask_rgb"]), resolve(row["image"])) for row in rows]
    else:
        masks = sorted((eval_dir / "cond_mask").glob("*_cond_mask.png"))
        items = [(sample_name(path), path, eval_dir / "gt_rgb" / f"{sample_name(path)}_gt_rgb.png") for path in masks]
    if args.max_samples > 0:
        items = items[: args.max_samples]
    metadata = []
    for name, mask_path, gt_path in tqdm(items, desc="SyntheticGen LoveDA"):
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        if pred_path.is_file() and not args.overwrite:
            continue
        ids = palette_to_ids(Image.open(mask_path))
        counts = np.bincount(ids.reshape(-1), minlength=7).astype(np.float64)
        ratios = counts / max(counts.sum(), 1.0)
        domain = "rural" if "rural" in name.lower() else "urban"
        prompt = f"A high-resolution satellite image of a {domain} area"
        digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        working = output / ".working" / name
        shutil.rmtree(working, ignore_errors=True)
        model_args.save_dir = str(working)
        model_args.init_mask = str(output / "mask_ids" / f"{name}.png")
        Image.fromarray(ids, mode="L").save(model_args.init_mask)
        model_args.mask_format = "indexed"
        model_args.strength_layout = 0.0
        model_args.ratios = ",".join(f"{value:.10f}" for value in ratios)
        model_args.domain = domain
        model_args.prompt = prompt
        model_args.seed = args.seed + digest
        generate_with_context(context, model_args)
        shutil.move(str(working / "image.png"), pred_path)
        shutil.copy2(mask_path, output / "cond_mask" / f"{name}_cond_mask.png")
        shutil.copy2(gt_path, output / "gt_rgb" / f"{name}_gt_rgb.png")
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.int16)
        pred = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.int16)
        Image.fromarray(np.abs(gt - pred).astype(np.uint8), mode="RGB").save(
            output / "absdiff" / f"{name}_absdiff.png"
        )
        (output / "prompts" / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata.append({"name": name, "domain": domain, "ratios": ratios.tolist(), "prompt": prompt})
        shutil.rmtree(working, ignore_errors=True)
    with (output / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")
    print(f"[run_syntheticgen_manifest] processed {len(metadata)} samples")


if __name__ == "__main__":
    main()
