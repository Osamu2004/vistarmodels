from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained SD ControlNet on bidirectional SECOND masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--controlnet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype, local_files_only=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None, local_files_only=True
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.max_samples:
        rows = rows[: args.max_samples]
    output = Path(args.output_dir).expanduser().resolve()
    for folder in ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows, desc="ControlNet SECOND"):
        name = row["name"]
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        if pred_path.is_file() and not args.overwrite:
            continue
        condition = Image.open(row["target_mask_rgb"]).convert("RGB")
        digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        generator = torch.Generator("cuda").manual_seed(args.seed + digest)
        pred = pipe(
            prompt=row["prompt"], image=condition, generator=generator,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        ).images[0].convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        pred.save(pred_path)
        shutil.copy2(row["source_image"], output / "source_rgb" / f"{name}_source_rgb.png")
        shutil.copy2(row["target_image"], output / "gt_rgb" / f"{name}_gt_rgb.png")
        shutil.copy2(row["target_mask_ids"], output / "cond_mask_ids" / f"{name}_cond_mask_ids.png")
        shutil.copy2(row["target_mask_rgb"], output / "cond_mask_official" / f"{name}_cond_mask_official.png")
        ids = np.asarray(Image.open(row["target_mask_ids"]).convert("L"))
        binary = np.repeat(((ids > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
        Image.fromarray(binary, mode="RGB").save(output / "cond_mask" / f"{name}_cond_mask.png")
        gt = np.asarray(Image.open(row["target_image"]).convert("RGB"), dtype=np.int16)
        pa = np.asarray(pred, dtype=np.int16)
        Image.fromarray(np.abs(gt - pa).astype(np.uint8), mode="RGB").save(output / "absdiff" / f"{name}_absdiff.png")
        (output / "prompts" / f"{name}.txt").write_text(row["prompt"] + "\n")


if __name__ == "__main__":
    main()
