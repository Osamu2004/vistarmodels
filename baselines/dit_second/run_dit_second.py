from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from common import ConditionedDiT, SecondManifestDataset, decode, encode, resolve_path


OUTPUT_FOLDERS = (
    "source_rgb",
    "cond_mask",
    "cond_mask_official",
    "cond_mask_ids",
    "gt_rgb",
    "pred_rgb",
    "absdiff",
    "prompts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source+directional-mask conditioned DiT-B/2 on SECOND.")
    parser.add_argument("--dit_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--vae_subfolder", default="vae")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify_files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state(payload: Any, weights: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and weights in payload:
        return payload[weights]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    available = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    raise ValueError(f"cannot find {weights!r} weights in checkpoint; available={available}")


def sample_seed(base_seed: int, name: str) -> int:
    digest = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return (base_seed + digest) % (2**63 - 1)


def autocast_context(dtype: torch.dtype | None):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def sample_is_complete(output: Path, name: str) -> bool:
    expected = (
        output / "source_rgb" / f"{name}_source_rgb.png",
        output / "cond_mask" / f"{name}_cond_mask.png",
        output / "cond_mask_official" / f"{name}_cond_mask_official.png",
        output / "cond_mask_ids" / f"{name}_cond_mask_ids.png",
        output / "gt_rgb" / f"{name}_gt_rgb.png",
        output / "pred_rgb" / f"{name}_pred_rgb.png",
        output / "absdiff" / f"{name}_absdiff.png",
        output / "prompts" / f"{name}.txt",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def save_resized(
    path: str,
    destination: Path,
    size: tuple[int, int],
    *,
    mode: str = "RGB",
    nearest: bool = False,
) -> None:
    image = Image.open(path).convert(mode)
    if image.size != size:
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        image = image.resize(size, resample)
    image.save(destination)


def save_contract_row(output: Path, row: dict[str, Any], pred: Image.Image) -> None:
    name = str(row["name"])
    pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
    pred.save(pred_path)
    source_path = output / "source_rgb" / f"{name}_source_rgb.png"
    target_path = output / "gt_rgb" / f"{name}_gt_rgb.png"
    ids_path = output / "cond_mask_ids" / f"{name}_cond_mask_ids.png"
    official_path = output / "cond_mask_official" / f"{name}_cond_mask_official.png"
    save_resized(row["source_image"], source_path, pred.size)
    save_resized(row["target_image"], target_path, pred.size)
    save_resized(row["target_mask_ids"], ids_path, pred.size, mode="L", nearest=True)
    save_resized(
        row["target_mask_rgb"],
        official_path,
        pred.size,
        mode="RGB",
        nearest=True,
    )
    ids = np.asarray(Image.open(ids_path).convert("L"))
    binary = np.repeat(((ids > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(binary, mode="RGB").save(output / "cond_mask" / f"{name}_cond_mask.png")
    gt = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.int16)
    difference = np.abs(gt - np.asarray(pred, dtype=np.int16)).astype(np.uint8)
    Image.fromarray(difference, mode="RGB").save(output / "absdiff" / f"{name}_absdiff.png")
    prompt = row.get("prompt", "A realistic remote sensing target image matching the directional change mask.")
    (output / "prompts" / f"{name}.txt").write_text(str(prompt) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DiT-B/2 inference requires CUDA")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    device = torch.device("cuda")
    precision = args.precision
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("[dit_second] WARNING: bf16 is unsupported on this GPU; falling back to fp32", flush=True)
        precision = "fp32"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]

    dit_root = resolve_path(args.dit_root)
    sys.path.insert(0, str(dit_root))
    from diffusion import create_diffusion

    diffusion = create_diffusion(str(args.steps))
    model = ConditionedDiT(str(dit_root), latent_size=args.image_size // 8).to(device).eval()
    payload = torch_load(resolve_path(args.checkpoint))
    model.load_state_dict(checkpoint_state(payload, args.weights), strict=True)

    vae_kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": autocast_dtype or torch.float32}
    if args.vae_subfolder:
        vae_kwargs["subfolder"] = args.vae_subfolder
    vae = AutoencoderKL.from_pretrained(args.vae, **vae_kwargs).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    dataset = SecondManifestDataset(
        args.manifest,
        image_size=args.image_size,
        verify_files=args.verify_files,
    )
    if args.max_samples > 0:
        dataset.rows = dataset.rows[: args.max_samples]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    output = resolve_path(args.output_dir)
    for folder in OUTPUT_FOLDERS:
        (output / folder).mkdir(parents=True, exist_ok=True)
    inference_config = {
        **vars(args),
        "effective_precision": precision,
        "resolved_checkpoint": str(resolve_path(args.checkpoint)),
        "resolved_manifest": str(resolve_path(args.manifest)),
        "samples": len(dataset),
    }
    config_path = output / "inference_config.json"
    existing_predictions = any((output / "pred_rgb").glob("*_pred_rgb.png"))
    if args.skip_existing and existing_predictions and not config_path.is_file():
        raise RuntimeError(
            "existing predictions have no inference_config.json; choose a new output directory "
            "or pass --no-skip_existing to regenerate them safely"
        )
    if args.skip_existing and config_path.is_file() and existing_predictions:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "resolved_checkpoint",
            "resolved_manifest",
            "weights",
            "image_size",
            "steps",
            "seed",
        )
        mismatches = {
            key: (previous.get(key), inference_config.get(key))
            for key in compatibility_keys
            if previous.get(key) != inference_config.get(key)
        }
        if mismatches:
            raise RuntimeError(
                f"existing output was generated with incompatible settings: {mismatches}. "
                "Choose a new output directory or pass --no-skip_existing to regenerate."
            )
    config_path.write_text(json.dumps(inference_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generated_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"DiT-B/2 SECOND ({args.weights})"):
            names = [str(name) for name in batch["name"]]
            indices = [int(index) for index in batch["index"]]
            keep = [
                not (args.skip_existing and sample_is_complete(output, name))
                for name in names
            ]
            if not any(keep):
                continue
            selected = torch.tensor(keep, dtype=torch.bool)
            selected_names = [name for name, use in zip(names, keep) if use]
            selected_indices = [index for index, use in zip(indices, keep) if use]
            source = batch["source"][selected].to(device, non_blocking=True)
            mask = batch["mask"][selected].to(device, non_blocking=True)
            with autocast_context(autocast_dtype):
                condition = torch.cat(
                    [
                        encode(vae, source, sample_posterior=False),
                        encode(vae, mask, sample_posterior=False),
                    ],
                    dim=1,
                ).float()
            noise = torch.cat(
                [
                    torch.randn(
                        (1, 4, args.image_size // 8, args.image_size // 8),
                        generator=torch.Generator(device=device).manual_seed(sample_seed(args.seed, name)),
                        device=device,
                    )
                    for name in selected_names
                ],
                dim=0,
            )
            labels = torch.zeros(len(selected_names), dtype=torch.long, device=device)
            with autocast_context(autocast_dtype):
                latent = diffusion.p_sample_loop(
                    model,
                    noise.shape,
                    noise,
                    clip_denoised=False,
                    model_kwargs={"y": labels, "condition": condition},
                    progress=False,
                    device=device,
                )
                pred_tensors = decode(vae, latent.to(dtype=autocast_dtype or torch.float32)).float().cpu()
            for index, name, pred_tensor in zip(selected_indices, selected_names, pred_tensors):
                pred = TF.to_pil_image(pred_tensor).convert("RGB")
                row = dataset.rows[index]
                save_contract_row(output, row, pred)
                generated_rows.append(
                    {
                        "name": name,
                        "seed": sample_seed(args.seed, name),
                        "direction": row.get("direction", "unknown"),
                    }
                )

    complete_rows = []
    for row in dataset.rows:
        name = str(row["name"])
        if sample_is_complete(output, name):
            complete_rows.append(
                {
                    "name": name,
                    "seed": sample_seed(args.seed, name),
                    "direction": row.get("direction", "unknown"),
                }
            )
    with (output / "generated_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in complete_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        f"[dit_second] output={output} newly_generated={len(generated_rows)} "
        f"complete={len(complete_rows)} total={len(dataset)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
