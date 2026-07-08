from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        np = torch = Image = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "EarthSynth inference dependencies are missing. Install diffusers, "
            "accelerate, torch, pillow, numpy, and tqdm first."
        ) from exc


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _resolve_path(path: str) -> Path:
    return Path(_normalize_wsl_unc(path)).expanduser().resolve()


def _read_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["condition_image"] = _normalize_wsl_unc(row["condition_image"])
            if row.get("target_image"):
                row["target_image"] = _normalize_wsl_unc(row["target_image"])
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_control_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image


def _load_rgb_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def _save_rgb(image: Image.Image, path: Path, size: int | None = None, nearest: bool = False) -> None:
    out = image.convert("RGB")
    if size is not None and out.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        out = out.resize((size, size), resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    step = max(1, int(batch_size))
    return [items[start:start + step] for start in range(0, len(items), step)]


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"auto", "fp16", "float16"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; use auto, fp16, bf16, or fp32.")


def _make_generators(seed: int, sample_indices: list[int], device: torch.device) -> list[torch.Generator]:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return [
        torch.Generator(device=generator_device).manual_seed(int(seed) + int(sample_index))
        for sample_index in sample_indices
    ]


def _load_pipe(
    *,
    base_model: str,
    controlnet_model: str,
    controlnet_subfolder: str,
    dtype: torch.dtype,
    device: torch.device,
    scheduler: str,
    enable_xformers: bool,
    cpu_offload: bool,
    pipeline_progress: bool,
):
    try:
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            StableDiffusionControlNetPipeline,
        )
    except ImportError as exc:
        raise ImportError("Install EarthSynth dependencies with `pip install -r requirements.txt`.") from exc

    controlnet_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if controlnet_subfolder:
        controlnet_kwargs["subfolder"] = controlnet_subfolder
    controlnet = ControlNetModel.from_pretrained(controlnet_model, **controlnet_kwargs)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    scheduler_name = scheduler.lower()
    if scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name in {"dpm", "dpm_solver", "dpmsolver"}:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name not in {"default", ""}:
        raise ValueError("Unsupported scheduler. Choose default, ddim, euler, or dpm.")

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - depends on optional xformers build.
            print(f"[run_earthsynth_manifest] xformers enable failed: {exc}", file=sys.stderr)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=not pipeline_progress)
    return pipe


def _generate_batch(
    *,
    pipe: Any,
    prompts: list[str],
    control_images: list[Image.Image],
    negative_prompt: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
    sample_indices: list[int],
    device: torch.device,
) -> list[Image.Image]:
    generators = _make_generators(seed, sample_indices, device)
    negative_prompts = [negative_prompt] * len(prompts) if negative_prompt else None
    with torch.inference_mode():
        result = pipe(
            prompt=prompts,
            image=control_images,
            negative_prompt=negative_prompts,
            height=int(resolution),
            width=int(resolution),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generators,
        )
    return [image.convert("RGB") for image in result.images]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EarthSynth ControlNet inference from a JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", default="jaychempan/EarthSynth")
    parser.add_argument(
        "--controlnet_subfolder",
        default="controlnet",
        help="EarthSynth stores the Diffusers ControlNet under this subfolder; set to empty for a direct local ControlNet folder.",
    )
    parser.add_argument("--resolution", type=int, default=512, help="native generation/control resolution")
    parser.add_argument("--eval_size", type=int, default=512, help="saved pred_rgb size for metric comparison")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--scheduler", default="default", choices=["default", "ddim", "euler", "dpm", "dpm_solver", "dpmsolver"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"])
    parser.add_argument("--negative_prompt", default="Low resolution, cropped, worst quality, low quality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--pipeline_progress", action="store_true")
    args = parser.parse_args()

    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    rows = _read_manifest(manifest, args.max_samples)
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")

    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    pipe = _load_pipe(
        base_model=_normalize_wsl_unc(args.base_model),
        controlnet_model=_normalize_wsl_unc(args.controlnet_model),
        controlnet_subfolder=str(args.controlnet_subfolder),
        dtype=dtype,
        device=device,
        scheduler=args.scheduler,
        enable_xformers=bool(args.enable_xformers),
        cpu_offload=bool(args.cpu_offload),
        pipeline_progress=bool(args.pipeline_progress),
    )

    resolved_manifest = output_dir / "manifest_resolved.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as manifest_f:
        row_batches = _batched(rows, max(1, int(args.batch_size)))
        for batch_index, row_batch in enumerate(tqdm(row_batches, desc="EarthSynth inference batches")):
            batch_records: list[dict[str, Any]] = []
            generate_records: list[dict[str, Any]] = []
            for offset, row in enumerate(row_batch):
                index = batch_index * max(1, int(args.batch_size)) + offset
                name = str(row.get("name") or f"sample_{index:06d}")
                pred_path = output_dir / "pred_rgb" / f"{name}_pred_rgb.png"
                native_path = output_dir / "pred_rgb_native" / f"{name}_pred_rgb_{args.resolution}.png"
                cond_out = output_dir / "cond_mask" / f"{name}_cond_mask.png"
                gt_out = output_dir / "gt_rgb" / f"{name}_gt_rgb.png"

                condition_image = _load_control_image(Path(row["condition_image"]), int(args.resolution))
                _save_rgb(condition_image, cond_out, size=int(args.eval_size), nearest=True)
                if row.get("target_image"):
                    gt_rgb = _load_rgb_image(Path(row["target_image"]), int(args.eval_size))
                    _save_rgb(gt_rgb, gt_out, size=int(args.eval_size))

                record = {
                    "row": row,
                    "index": index,
                    "name": name,
                    "condition_image": condition_image,
                    "pred_path": pred_path,
                    "native_path": native_path,
                    "status": "skipped_existing" if pred_path.is_file() and not args.overwrite else "pending",
                }
                if record["status"] == "pending":
                    generate_records.append(record)
                batch_records.append(record)

            if generate_records:
                images = _generate_batch(
                    pipe=pipe,
                    prompts=[str(record["row"].get("prompt") or "") for record in generate_records],
                    control_images=[record["condition_image"] for record in generate_records],
                    negative_prompt=str(args.negative_prompt),
                    resolution=int(args.resolution),
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.guidance_scale),
                    controlnet_conditioning_scale=float(args.controlnet_conditioning_scale),
                    seed=int(args.seed),
                    sample_indices=[int(record["index"]) for record in generate_records],
                    device=device,
                )
                for record, image in zip(generate_records, images):
                    _save_rgb(image, record["native_path"])
                    _save_rgb(image, record["pred_path"], size=int(args.eval_size))
                    record["status"] = "generated"

            for record in batch_records:
                row = record["row"]
                manifest_f.write(
                    json.dumps(
                        {
                            **row,
                            "name": record["name"],
                            "pred_rgb": str(record["pred_path"]),
                            "pred_rgb_native": str(record["native_path"]),
                            "resolution": int(args.resolution),
                            "eval_size": int(args.eval_size),
                            "batch_size": int(args.batch_size),
                            "num_inference_steps": int(args.num_inference_steps),
                            "guidance_scale": float(args.guidance_scale),
                            "controlnet_conditioning_scale": float(args.controlnet_conditioning_scale),
                            "scheduler": str(args.scheduler),
                            "base_model": str(args.base_model),
                            "controlnet_model": str(args.controlnet_model),
                            "controlnet_subfolder": str(args.controlnet_subfolder),
                            "status": record["status"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[run_earthsynth_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
