from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an unconditional LoveDA DDPM on paired Table 3 slots.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pipe = DDPMPipeline.from_pretrained(args.model, local_files_only=True).to("cuda")
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for folder in ("cond_mask", "gt_rgb", "pred_rgb", "absdiff", "prompts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    masks = sorted((eval_dir / "cond_mask").glob("*_cond_mask.png"))
    if args.max_samples:
        masks = masks[: args.max_samples]
    for mask in tqdm(masks, desc="DDPM LoveDA"):
        name = mask.name[: -len("_cond_mask.png")]
        gt_path = eval_dir / "gt_rgb" / f"{name}_gt_rgb.png"
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        if pred_path.is_file() and not args.overwrite:
            continue
        digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        generator = torch.Generator("cuda").manual_seed(args.seed + digest)
        pred = pipe(batch_size=1, generator=generator, num_inference_steps=args.steps).images[0]
        pred = pred.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        pred.save(pred_path)
        shutil.copy2(mask, output / "cond_mask" / mask.name)
        shutil.copy2(gt_path, output / "gt_rgb" / gt_path.name)
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.int16)
        pa = np.asarray(pred, dtype=np.int16)
        Image.fromarray(np.abs(gt - pa).astype(np.uint8), mode="RGB").save(output / "absdiff" / f"{name}_absdiff.png")
        (output / "prompts" / f"{name}.txt").write_text("Unconditional LoveDA DDPM; mask not passed to model.\n")


if __name__ == "__main__":
    main()
