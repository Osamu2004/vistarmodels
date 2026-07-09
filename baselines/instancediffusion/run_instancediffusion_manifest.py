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
            "InstanceDiffusion inference dependencies are missing. Install torch, "
            "transformers, accelerate, safetensors, pillow, numpy, tqdm, and the "
            "InstanceDiffusion diffusers fork first."
        ) from exc


LOVEDA_PALETTE_TEXT: dict[tuple[int, int, int], tuple[str, str]] = {
    (0, 0, 0): ("background", "background or unlabeled land-cover area in a remote sensing satellite image"),
    (255, 255, 255): ("building", "building roofs and built-up structures in a remote sensing satellite image"),
    (255, 0, 0): ("road", "roads and transportation surfaces in a remote sensing satellite image"),
    (0, 0, 255): ("water", "water bodies such as rivers, ponds, or lakes in a remote sensing satellite image"),
    (255, 255, 0): ("barren", "barren land or bare soil areas in a remote sensing satellite image"),
    (0, 255, 0): ("forest", "forest and tree-covered vegetation in a remote sensing satellite image"),
    (0, 255, 255): ("agriculture", "agricultural fields and cropland in a remote sensing satellite image"),
}


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


def _is_local_ref(text: str) -> bool:
    value = _normalize_wsl_unc(str(text))
    return value.startswith(("/", "./", "../", "~"))


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


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; use auto, bf16, fp16, or fp32.")


def _load_rgb(path: Path, size: int, *, nearest: bool) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        image = image.resize((size, size), resample)
    return image


def _save_rgb(image: Image.Image, path: Path, size: int | None = None, *, nearest: bool = False) -> None:
    out = image.convert("RGB")
    if size is not None and out.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        out = out.resize((size, size), resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _is_valid_rgb_file(path: Path, size: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode in {"RGB", "RGBA", "P"} and image.size == (int(size), int(size))
    except Exception:
        return False


def _box_from_mask(mask: np.ndarray, padding: int) -> list[float]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    x0 = max(0, int(xs.min()) - int(padding))
    y0 = max(0, int(ys.min()) - int(padding))
    x1 = min(width, int(xs.max()) + 1 + int(padding))
    y1 = min(height, int(ys.max()) + 1 + int(padding))
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _make_instance_conditions(
    mask_rgb: Image.Image,
    *,
    include_background: bool,
    min_box_area: int,
    max_boxes: int,
    box_padding: int,
) -> list[dict[str, Any]]:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.uint8)
    regions: list[dict[str, Any]] = []
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    for color_arr in colors:
        color = tuple(int(v) for v in color_arr.tolist())
        class_name, phrase = LOVEDA_PALETTE_TEXT.get(
            color,
            (
                f"rgb_{color[0]}_{color[1]}_{color[2]}",
                "a land-cover region in a remote sensing satellite image",
            ),
        )
        if class_name == "background" and not include_background:
            continue
        color_mask = (
            (arr[..., 0] == color[0])
            & (arr[..., 1] == color[1])
            & (arr[..., 2] == color[2])
        )
        area = int(color_mask.sum())
        if area < int(min_box_area):
            continue
        regions.append(
            {
                "class_name": class_name,
                "phrase": phrase,
                "color": [color[0], color[1], color[2]],
                "area": area,
                "box": _box_from_mask(color_mask, int(box_padding)),
            }
        )
    regions.sort(key=lambda item: item["area"], reverse=True)
    return regions[: max(0, int(max_boxes))]


def _write_instancediffusion_input(
    *,
    name: str,
    input_dir: Path,
    prompt: str,
    negative_prompt: str,
    regions: list[dict[str, Any]],
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    json_path = input_dir / f"{name}.json"
    payload = {
        "caption": prompt,
        "negative_prompt": negative_prompt,
        "annos": [
            {
                "caption": region["phrase"],
                "category_name": region["class_name"],
                "bbox_normalized_xyxy": region["box"],
                "color": region["color"],
                "area": region["area"],
            }
            for region in regions
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def _install_diffusers_source(diffusers_root: str) -> None:
    if not diffusers_root:
        return
    root = _resolve_path(diffusers_root)
    src = root / "src"
    if not (src / "diffusers").is_dir():
        raise NotADirectoryError(f"InstanceDiffusion diffusers source not found: {src / 'diffusers'}")
    sys.path.insert(0, str(src))


def _load_pipe(
    *,
    diffusers_root: str,
    model_name_or_path: str,
    dtype: torch.dtype,
    device: torch.device,
    scheduler: str,
    enable_xformers: bool,
    cpu_offload: bool,
    pipeline_progress: bool,
):
    _install_diffusers_source(diffusers_root)
    try:
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            StableDiffusionINSTDIFFPipeline,
        )
    except ImportError as exc:
        raise ImportError(
            "StableDiffusionINSTDIFFPipeline is not importable. Run "
            "`bash scripts/bootstrap_instancediffusion.sh` and keep "
            "INSTANCEDIFFUSION_DIFFUSERS_ROOT pointing to the cloned diffusers fork."
        ) from exc

    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        model_name_or_path,
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
            print(f"[run_instancediffusion_manifest] xformers enable failed: {exc}", file=sys.stderr)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=not pipeline_progress)
    return pipe


def _generate_one(
    *,
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    phrases: list[str],
    boxes: list[list[float]],
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    alpha: float,
    beta: float,
    seed: int,
    device: torch.device,
) -> Image.Image:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            instdiff_phrases=phrases,
            instdiff_boxes=boxes,
            instdiff_scheduled_sampling_alpha=float(alpha),
            instdiff_scheduled_sampling_beta=float(beta),
            height=int(resolution),
            width=int(resolution),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            num_images_per_prompt=1,
            output_type="pil",
            generator=generator,
        )
    return result.images[0].convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InstanceDiffusion inference from a Vistar-style JSONL manifest.")
    parser.add_argument("--diffusers_root", default="third_party/diffusers-instancediffusion")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="kyeongry/instancediffusion_sd15")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="diffusers InstanceDiffusion supports per-sample boxes; use 1")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--alpha", type=float, default=0.8, help="scheduled sampling alpha for gated self-attention")
    parser.add_argument("--beta", type=float, default=0.36, help="scheduled sampling beta for multi-instance sampler")
    parser.add_argument("--scheduler", default="default", choices=["default", "ddim", "euler", "dpm", "dpm_solver", "dpmsolver"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--negative_prompt", default="longbody, lowres, bad anatomy, cropped, worst quality, low quality")
    parser.add_argument("--min_box_area", type=int, default=64)
    parser.add_argument("--max_boxes", type=int, default=30)
    parser.add_argument("--box_padding", type=int, default=0)
    parser.add_argument("--include_background", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--pipeline_progress", action="store_true")
    parser.add_argument(
        "--global_caption",
        default="A high-resolution remote sensing satellite image with buildings, roads, water, barren land, forest, and agriculture.",
    )
    args = parser.parse_args()

    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    model_ref = _normalize_wsl_unc(args.model_name_or_path)
    if _is_local_ref(model_ref) and not Path(model_ref).expanduser().exists():
        raise FileNotFoundError(f"InstanceDiffusion model local path not found: {model_ref}")
    if int(args.batch_size) != 1:
        raise ValueError("InstanceDiffusion wrapper only supports batch_size=1 because boxes/phrases are per sample.")
    if int(args.resolution) % 8 != 0:
        raise ValueError(f"resolution={args.resolution} must be divisible by 8")

    rows = _read_manifest(manifest, int(args.max_samples))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")

    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb", "instancediffusion_inputs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    pipe = _load_pipe(
        diffusers_root=str(args.diffusers_root),
        model_name_or_path=model_ref,
        dtype=dtype,
        device=device,
        scheduler=str(args.scheduler),
        enable_xformers=bool(args.enable_xformers),
        cpu_offload=bool(args.cpu_offload),
        pipeline_progress=bool(args.pipeline_progress),
    )

    resolved_manifest = output_dir / "manifest_resolved.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as manifest_f:
        for index, row in enumerate(tqdm(rows, desc="InstanceDiffusion inference")):
            name = str(row.get("name") or f"sample_{index:06d}")
            pred_path = output_dir / "pred_rgb" / f"{name}_pred_rgb.png"
            native_path = output_dir / "pred_rgb_native" / f"{name}_pred_rgb_{args.resolution}.png"
            cond_out = output_dir / "cond_mask" / f"{name}_cond_mask.png"
            gt_out = output_dir / "gt_rgb" / f"{name}_gt_rgb.png"

            condition_image = _load_rgb(Path(row["condition_image"]), int(args.resolution), nearest=True)
            _save_rgb(condition_image, cond_out, size=int(args.eval_size), nearest=True)
            if row.get("target_image"):
                gt_rgb = _load_rgb(Path(row["target_image"]), int(args.eval_size), nearest=False)
                _save_rgb(gt_rgb, gt_out, size=int(args.eval_size), nearest=False)

            prompt = str(row.get("prompt") or args.global_caption)
            regions = _make_instance_conditions(
                condition_image,
                include_background=bool(args.include_background),
                min_box_area=int(args.min_box_area),
                max_boxes=int(args.max_boxes),
                box_padding=int(args.box_padding),
            )
            input_json = _write_instancediffusion_input(
                name=name,
                input_dir=output_dir / "instancediffusion_inputs",
                prompt=prompt,
                negative_prompt=str(args.negative_prompt),
                regions=regions,
            )

            has_valid_prediction = _is_valid_rgb_file(pred_path, int(args.eval_size))
            status = "skipped_existing" if has_valid_prediction and not args.overwrite else "pending"

            if status == "pending":
                image = _generate_one(
                    pipe=pipe,
                    prompt=prompt,
                    negative_prompt=str(args.negative_prompt),
                    phrases=[str(region["phrase"]) for region in regions],
                    boxes=[list(region["box"]) for region in regions],
                    resolution=int(args.resolution),
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.guidance_scale),
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    seed=int(args.seed) + index,
                    device=device,
                )
                _save_rgb(image, native_path)
                _save_rgb(image, pred_path, size=int(args.eval_size))
                status = "generated"

            manifest_f.write(
                json.dumps(
                    {
                        **row,
                        "name": name,
                        "pred_rgb": str(pred_path),
                        "pred_rgb_native": str(native_path),
                        "instancediffusion_input_json": str(input_json),
                        "resolution": int(args.resolution),
                        "eval_size": int(args.eval_size),
                        "batch_size": int(args.batch_size),
                        "num_inference_steps": int(args.num_inference_steps),
                        "guidance_scale": float(args.guidance_scale),
                        "alpha": float(args.alpha),
                        "beta": float(args.beta),
                        "scheduler": str(args.scheduler),
                        "model_name_or_path": str(args.model_name_or_path),
                        "diffusers_root": str(args.diffusers_root),
                        "min_box_area": int(args.min_box_area),
                        "max_boxes": int(args.max_boxes),
                        "box_padding": int(args.box_padding),
                        "include_background": bool(args.include_background),
                        "instance_regions": regions,
                        "status": status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[run_instancediffusion_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
