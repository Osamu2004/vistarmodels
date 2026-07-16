from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from common import ConditionedDiT, SecondManifestDataset, count_parameters, encode, manifest_sha256, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source+directional-mask conditioned DiT-B/2 on SECOND.")
    parser.add_argument("--dit_root", required=True, help="Pinned checkout of facebookresearch/DiT")
    parser.add_argument("--manifest", required=True, help="Directional SECOND train JSONL")
    parser.add_argument("--vae", required=True, help="Local Stable Diffusion 1.5 Diffusers snapshot")
    parser.add_argument("--vae_subfolder", default="vae", help="Use an empty string if --vae points directly to the VAE")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=300000, help="Number of optimizer updates")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr_schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="0 disables clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=10000, help="Optimizer-step interval")
    parser.add_argument("--keep_last", type=int, default=3, help="Number of periodic checkpoints to retain; 0 keeps all")
    parser.add_argument("--log_every", type=int, default=20, help="Optimizer-step interval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dist_backend",
        choices=("gloo",),
        default="gloo",
        help="Distributed process-group backend; Gloo matches the VISTAR training stack.",
    )
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    parser.add_argument("--verify_files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def init_distributed(backend: str) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DiT-B/2 training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataloader_worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def autocast_context(dtype: torch.dtype | None):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name, target in ema_parameters.items():
        source = model_parameters[name]
        target.mul_(decay).add_(source, alpha=1.0 - decay)
    for name, target in ema.named_buffers():
        target.copy_(dict(model.named_buffers())[name])


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        if args.lr_schedule == "constant":
            return 1.0
        denominator = max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def latest_checkpoint(output: Path) -> Path | None:
    candidates = sorted(output.glob("checkpoint-*.pt"))
    return candidates[-1] if candidates else None


def resolve_resume(value: str, output: Path) -> Path | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "false", "0"}:
        return None
    if normalized == "auto":
        return latest_checkpoint(output)
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return path


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def gather_rng_states(world_size: int) -> list[dict[str, Any]]:
    local = rng_state()
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prune_checkpoints(output: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output.glob("checkpoint-*.pt"))
    for stale in checkpoints[:-keep_last]:
        stale.unlink()


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    optimizer_step: int,
    epoch: int,
    next_batch_in_epoch: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Path:
    all_rng_states = gather_rng_states(world_size)
    checkpoint_path = output / f"checkpoint-{optimizer_step:07d}.pt"
    if is_main(rank):
        payload = {
            "format_version": 2,
            "architecture": "DiT-B/2-source-mask-latent-condition",
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer_step": optimizer_step,
            "step": optimizer_step,
            "epoch": epoch,
            "next_batch_in_epoch": next_batch_in_epoch,
            "rng_states": all_rng_states,
            "args": vars(args),
        }
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        atomic_json(output / "latest.json", {"checkpoint": checkpoint_path.name, "optimizer_step": optimizer_step})
        prune_checkpoints(output, args.keep_last)
    barrier(world_size)
    return checkpoint_path


def append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("max_steps, batch_size, and grad_accum must be positive")
    if args.save_every <= 0 or args.log_every <= 0 or args.keep_last < 0:
        raise ValueError("save_every/log_every must be positive and keep_last must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")

    rank, world_size, local_rank, device = init_distributed(args.dist_backend)
    output = resolve_path(args.output_dir)
    if is_main(rank):
        output.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    try:
        precision = args.precision
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            if is_main(rank):
                print("[dit_second] WARNING: bf16 is unsupported on this GPU; falling back to fp32", flush=True)
            precision = "fp32"
        autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
        scaler = make_grad_scaler(enabled=precision == "fp16")

        # Rank offsets prevent identical VAE posterior samples while the sampler
        # itself remains deterministic through its shared base seed.
        seed_everything(args.seed + rank)
        dataset = SecondManifestDataset(
            args.manifest,
            image_size=args.image_size,
            hflip_prob=args.hflip_prob,
            vflip_prob=args.vflip_prob,
            verify_files=args.verify_files,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        loader_generator = torch.Generator().manual_seed(args.seed + rank)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=dataloader_worker_init,
            generator=loader_generator,
        )
        if len(loader) == 0:
            raise ValueError(
                f"no full batches: dataset={len(dataset)}, world_size={world_size}, per_gpu_batch={args.batch_size}"
            )

        sys.path.insert(0, str(resolve_path(args.dit_root)))
        from diffusion import create_diffusion

        diffusion = create_diffusion(timestep_respacing="")
        latent_size = args.image_size // 8
        model = ConditionedDiT(args.dit_root, latent_size=latent_size).to(device)
        ema = copy.deepcopy(model).to(device).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        vae_kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": autocast_dtype or torch.float32}
        if args.vae_subfolder:
            vae_kwargs["subfolder"] = args.vae_subfolder
        vae = AutoencoderKL.from_pretrained(args.vae, **vae_kwargs).to(device).eval()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(optimizer, args)
        optimizer_step = 0
        start_epoch = 0
        start_batch = 0
        resume_path = resolve_resume(args.resume, output)
        if resume_path is not None:
            payload = torch_load(resume_path)
            if int(payload.get("format_version", 0)) != 2:
                raise RuntimeError(
                    "legacy DiT checkpoint resume is disabled because the prototype counted micro-batches as steps. "
                    "Start a new run with --resume none; legacy model/EMA weights remain usable for inference."
                )
            if payload.get("architecture") != "DiT-B/2-source-mask-latent-condition":
                raise RuntimeError(f"unexpected checkpoint architecture: {payload.get('architecture')!r}")
            model.load_state_dict(payload["model"])
            ema.load_state_dict(payload.get("ema", payload["model"]))
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_to(optimizer, device)
            if "lr_scheduler" in payload:
                scheduler.load_state_dict(payload["lr_scheduler"])
            optimizer_step = int(payload.get("optimizer_step", payload.get("step", 0)))
            start_epoch = int(payload.get("epoch", 0))
            start_batch = int(payload.get("next_batch_in_epoch", 0))
            if "scaler" in payload:
                scaler.load_state_dict(payload["scaler"])
            saved_rng = payload.get("rng_states", [])
            if rank < len(saved_rng):
                restore_rng_state(saved_rng[rank])
            if is_main(rank):
                print(f"[dit_second] resumed {resume_path} at optimizer_step={optimizer_step}", flush=True)

        raw_model = model
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        global_batch = args.batch_size * world_size * args.grad_accum
        run_config = {
            **vars(args),
            "resolved_manifest": str(resolve_path(args.manifest)),
            "manifest_sha256": manifest_sha256(args.manifest),
            "dataset_samples": len(dataset),
            "world_size": world_size,
            "global_batch_size": global_batch,
            "effective_precision": precision,
            "trainable_parameters": count_parameters(raw_model),
            "torch_version": torch.__version__,
        }
        if is_main(rank):
            atomic_json(output / "train_config.json", run_config)
            print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            total=args.max_steps,
            initial=optimizer_step,
            disable=not is_main(rank),
            desc="DiT-B/2 SECOND optimizer steps",
        )
        log_path = output / "train_log.jsonl"
        epoch = start_epoch
        next_batch_in_epoch = start_batch
        accumulated_loss = 0.0
        micro_step = 0
        last_saved_step = optimizer_step if resume_path is not None else -1
        wall_start = time.time()

        while optimizer_step < args.max_steps:
            sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                target = batch["target"].to(device, non_blocking=True)
                source = batch["source"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.no_grad(), autocast_context(autocast_dtype):
                    target_z = encode(vae, target).float()
                    source_z = encode(vae, source).float()
                    mask_z = encode(vae, mask).float()
                condition = torch.cat([source_z, mask_z], dim=1)
                timesteps = torch.randint(0, diffusion.num_timesteps, (target_z.shape[0],), device=device)
                labels = torch.zeros(target_z.shape[0], dtype=torch.long, device=device)

                micro_step += 1
                should_update = micro_step == args.grad_accum
                sync_context = nullcontext() if should_update or world_size == 1 else model.no_sync()
                with sync_context:
                    with autocast_context(autocast_dtype):
                        loss = diffusion.training_losses(
                            model,
                            target_z,
                            timesteps,
                            {"y": labels, "condition": condition},
                        )["loss"].mean()
                        scaled_loss = loss / args.grad_accum
                    scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.detach())
                next_batch_in_epoch = batch_index + 1

                if not should_update:
                    continue
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_update = precision == "fp16" and scaler.get_scale() < previous_scale
                if skipped_update:
                    micro_step = 0
                    accumulated_loss = 0.0
                    if is_main(rank):
                        print("[dit_second] skipped an fp16 optimizer update after gradient overflow", flush=True)
                    continue
                scheduler.step()
                update_ema(ema, raw_model, args.ema_decay)
                optimizer_step += 1
                micro_step = 0

                mean_loss = torch.tensor(accumulated_loss / args.grad_accum, device=device)
                accumulated_loss = 0.0
                if world_size > 1:
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss /= world_size
                progress.update(1)
                progress.set_postfix(loss=f"{mean_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if optimizer_step % args.log_every == 0 or optimizer_step == 1:
                    if is_main(rank):
                        append_log(
                            log_path,
                            {
                                "optimizer_step": optimizer_step,
                                "epoch": epoch,
                                "next_batch_in_epoch": next_batch_in_epoch,
                                "loss": mean_loss.item(),
                                "lr": scheduler.get_last_lr()[0],
                                "elapsed_seconds": time.time() - wall_start,
                                "global_batch_size": global_batch,
                            },
                        )

                if optimizer_step % args.save_every == 0 or optimizer_step == args.max_steps:
                    checkpoint = save_checkpoint(
                        output,
                        raw_model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        optimizer_step,
                        epoch,
                        next_batch_in_epoch,
                        args,
                        rank,
                        world_size,
                    )
                    last_saved_step = optimizer_step
                    if is_main(rank):
                        print(f"[dit_second] saved {checkpoint}", flush=True)

                if optimizer_step >= args.max_steps:
                    break

            epoch += 1
            start_batch = 0
            next_batch_in_epoch = 0

        if last_saved_step != optimizer_step:
            save_checkpoint(
                output,
                raw_model,
                ema,
                optimizer,
                scheduler,
                scaler,
                optimizer_step,
                epoch,
                next_batch_in_epoch,
                args,
                rank,
                world_size,
            )
        progress.close()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
