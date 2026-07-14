from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from common import ConditionedDiT, SecondManifestDataset, decode, encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditioned DiT-B/2 on SECOND.")
    parser.add_argument("--dit_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda")
    sys.path.insert(0, str(Path(args.dit_root).resolve()))
    from diffusion import create_diffusion
    diffusion = create_diffusion(str(args.steps))
    model = ConditionedDiT(args.dit_root).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["ema"])
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    dataset = SecondManifestDataset(args.manifest)
    rows = dataset.rows[: args.max_samples or None]
    output = Path(args.output_dir).expanduser().resolve()
    for folder in ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows, desc="DiT-B/2 SECOND"):
        name = row["name"]
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        source = SecondManifestDataset.load(row["source_image"]).unsqueeze(0).to(device, torch.bfloat16)
        mask = SecondManifestDataset.load(row["target_mask_rgb"]).unsqueeze(0).to(device, torch.bfloat16)
        with torch.no_grad():
            condition = torch.cat([encode(vae, source), encode(vae, mask)], dim=1).float()
            digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            generator = torch.Generator(device).manual_seed(args.seed + digest)
            noise = torch.randn((1, 4, 32, 32), generator=generator, device=device)
            y = torch.zeros(1, dtype=torch.long, device=device)
            latent = diffusion.p_sample_loop(model, noise.shape, noise, clip_denoised=False,
                                             model_kwargs={"y": y, "condition": condition}, progress=False, device=device)
            pred_tensor = decode(vae, latent.to(torch.bfloat16))[0].float().cpu()
        pred = TF.to_pil_image(pred_tensor).convert("RGB")
        pred.save(pred_path)
        shutil.copy2(row["source_image"], output / "source_rgb" / f"{name}_source_rgb.png")
        shutil.copy2(row["target_image"], output / "gt_rgb" / f"{name}_gt_rgb.png")
        shutil.copy2(row["target_mask_ids"], output / "cond_mask_ids" / f"{name}_cond_mask_ids.png")
        shutil.copy2(row["target_mask_rgb"], output / "cond_mask_official" / f"{name}_cond_mask_official.png")
        ids = np.asarray(Image.open(row["target_mask_ids"]).convert("L"))
        Image.fromarray(np.repeat(((ids > 0) * 255).astype(np.uint8)[..., None], 3, 2)).save(output / "cond_mask" / f"{name}_cond_mask.png")
        gt = np.asarray(Image.open(row["target_image"]).convert("RGB"), dtype=np.int16)
        Image.fromarray(np.abs(gt - np.asarray(pred, dtype=np.int16)).astype(np.uint8)).save(output / "absdiff" / f"{name}_absdiff.png")
        (output / "prompts" / f"{name}.txt").write_text(row["prompt"] + "\n")


if __name__ == "__main__":
    from torchvision.transforms import functional as TF
    main()
