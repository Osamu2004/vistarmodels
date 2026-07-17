from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModel

from common import colorize_mask, controlnet_prompt, load_jsonl, load_mask_ids, load_rgb


DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "MPI_LOCALRANKID",
    "PMI_RANK",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_SIZE",
)


def clear_single_process_distributed_environment() -> dict[str, str]:
    """Prevent Accelerate from treating a direct one-GPU run as torchrun.

    Interactive shells can retain rank variables from a previous distributed
    launch.  Accelerate uses those variables for auto-detection, so a direct
    Python process may otherwise enter its distributed branch before a default
    process group exists.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("--single_process cannot clear an initialized distributed process group")
    removed = {}
    for key in DISTRIBUTED_ENV_KEYS:
        value = os.environ.pop(key, None)
        if value is not None:
            removed[key] = value
    return removed


class SecondControlNetDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        tokenizer: AutoTokenizer,
        resolution: int,
        random_flip: bool,
        max_samples: int,
    ) -> None:
        self.rows = load_jsonl(manifest)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target = load_rgb(row, "target_image", self.resolution)
        mask_ids = load_mask_ids(row, self.resolution)
        condition = colorize_mask(mask_ids)
        if self.random_flip and random.random() < 0.5:
            target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            condition = condition.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target_array = np.asarray(target, dtype=np.float32) / 127.5 - 1.0
        condition_array = np.asarray(condition, dtype=np.float32) / 255.0
        prompt = controlnet_prompt(row, self.resolution)
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "pixel_values": torch.from_numpy(target_array).permute(2, 0, 1),
            "conditioning_pixel_values": torch.from_numpy(condition_array).permute(2, 0, 1),
            "input_ids": input_ids,
        }


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]).contiguous().float(),
        "conditioning_pixel_values": torch.stack(
            [example["conditioning_pixel_values"] for example in examples]
        ).contiguous().float(),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
    }


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(match.group(1))


def prune_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    for _, path in candidates[:-limit]:
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SD1.5 ControlNet on bidirectional SECOND masks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--controlnet_init", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_limit", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--dist_backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--single_process", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--no_random_flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removed_distributed_environment = (
        clear_single_process_distributed_environment() if args.single_process else {}
    )
    manifest = Path(args.manifest).expanduser().resolve()
    base_model = Path(args.base_model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not base_model.is_dir():
        raise NotADirectoryError(base_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_group = InitProcessGroupKwargs(backend=args.dist_backend, timeout=timedelta(minutes=60))
    project_config = ProjectConfiguration(project_dir=str(output_dir), total_limit=args.checkpoint_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        kwargs_handlers=[process_group],
    )
    set_seed(args.seed, device_specific=True)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, subfolder="tokenizer", use_fast=False, local_files_only=True
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", local_files_only=True)
    if args.controlnet_init:
        controlnet = ControlNetModel.from_pretrained(
            Path(args.controlnet_init).expanduser().resolve(), local_files_only=True
        )
    else:
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    if args.enable_xformers:
        try:
            controlnet.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            raise RuntimeError("--enable_xformers was requested but could not be enabled") from exc

    dataset = SecondControlNetDataset(
        str(manifest), tokenizer, args.resolution, not args.no_random_flip, args.max_train_samples
    )
    data_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        generator=data_generator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        raise ValueError(f"dataset has {len(dataset)} rows, smaller than batch size {args.batch_size}")

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    unet.eval()
    text_encoder.eval()

    global_step = 0
    resume_path: Path | None = None
    if args.resume == "auto":
        resume_path = latest_checkpoint(output_dir)
    elif args.resume not in {"", "none", "off", "false", "0"}:
        resume_path = Path(args.resume).expanduser().resolve()
    if resume_path is not None:
        if not resume_path.is_dir():
            raise NotADirectoryError(resume_path)
        accelerator.load_state(str(resume_path))
        global_step = checkpoint_step(resume_path)

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    first_epoch = global_step // updates_per_epoch
    resume_updates = global_step % updates_per_epoch
    resume_batches = resume_updates * args.gradient_accumulation_steps
    trainable_parameters = sum(parameter.numel() for parameter in controlnet.parameters() if parameter.requires_grad)
    config = {
        **vars(args),
        "manifest": str(manifest),
        "base_model": str(base_model),
        "output_dir": str(output_dir),
        "manifest_rows": len(dataset),
        "world_size": accelerator.num_processes,
        "global_batch_size": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
        "trainable_parameters": trainable_parameters,
        "resume_path": str(resume_path) if resume_path else None,
        "resume_global_step": global_step,
        "single_process": args.single_process,
        "cleared_distributed_environment": removed_distributed_environment,
        "versions": {
            "torch": torch.__version__,
            "accelerate": accelerate.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
        "condition_contract": "target-side directional SECOND semantic change mask + class-aware text",
    }
    if accelerator.is_main_process:
        (output_dir / "train_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config, indent=2), flush=True)
    accelerator.wait_for_everyone()

    log_path = output_dir / "train_log.jsonl"
    epoch = first_epoch
    while global_step < args.max_train_steps:
        epoch_dataloader = dataloader
        if epoch == first_epoch and resume_batches > 0:
            epoch_dataloader = accelerator.skip_first_batches(dataloader, resume_batches)
        for batch in epoch_dataloader:
            with accelerator.accumulate(controlnet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch["conditioning_pixel_values"].to(dtype=weight_dtype),
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_samples],
                    mid_block_additional_residual=mid_sample.to(dtype=weight_dtype),
                ).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"unsupported prediction type: {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                if accelerator.is_main_process and (global_step == 1 or global_step % args.log_every == 0):
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": reduced_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                if global_step % args.save_every == 0 or global_step >= args.max_train_steps:
                    checkpoint = output_dir / f"checkpoint-{global_step:07d}"
                    accelerator.save_state(str(checkpoint))
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        prune_checkpoints(output_dir, args.checkpoint_limit)
                if global_step >= args.max_train_steps:
                    break
        epoch += 1
        resume_batches = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(controlnet)
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        (output_dir / "completed.json").write_text(
            json.dumps({"global_step": global_step, "status": "complete"}, indent=2) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
