from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ConditionedDiT, SecondManifestDataset, encode


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1 - decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source+target-mask conditioned DiT-B/2 on SECOND.")
    parser.add_argument("--dit_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True, help="Local SD Diffusers snapshot containing vae/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(Path(args.dit_root).resolve()))
    from diffusion import create_diffusion
    diffusion = create_diffusion(timestep_respacing="")
    model = ConditionedDiT(args.dit_root).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
    loader = DataLoader(SecondManifestDataset(args.manifest), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, initial=step, desc="DiT-B/2 SECOND")
    while step < args.max_steps:
        for target, source, mask, _ in loader:
            target, source, mask = target.to(device), source.to(device), mask.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                target_z = encode(vae, target.to(torch.bfloat16))
                condition = torch.cat([encode(vae, source.to(torch.bfloat16)), encode(vae, mask.to(torch.bfloat16))], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (target_z.shape[0],), device=device)
            y = torch.zeros(target_z.shape[0], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_losses(model, target_z.float(), t, {"y": y, "condition": condition.float()})["loss"].mean()
                loss = loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model, 0.9999)
            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")
            if step % args.save_every == 0 or step == args.max_steps:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                           output / f"checkpoint-{step:07d}.pt")
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
