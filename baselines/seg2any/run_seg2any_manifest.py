from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        np = torch = Image = transforms = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "Seg2Any inference dependencies are missing. Install requirements-seg2any.txt "
            "and the official Seg2Any dependencies first."
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


def _is_local_ref(text: str) -> bool:
    value = _normalize_wsl_unc(str(text))
    return value.startswith(("/", "./", "../", "~"))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; use bf16, fp16, or fp32.")


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


def _segments_from_mask(mask_rgb: Image.Image) -> list[dict[str, Any]]:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.uint8)
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    segments: list[dict[str, Any]] = []
    for color_arr in colors:
        color = tuple(int(v) for v in color_arr.tolist())
        if color in LOVEDA_PALETTE_TEXT:
            class_name, text = LOVEDA_PALETTE_TEXT[color]
        else:
            class_name = f"rgb_{color[0]}_{color[1]}_{color[2]}"
            text = "a land-cover region in a remote sensing satellite image"
        segments.append(
            {
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "class_name": class_name,
                "text": text,
            }
        )
    return segments


def _write_seg2any_input(
    *,
    name: str,
    mask_rgb: Image.Image,
    input_dir: Path,
    global_caption: str,
    seed: int,
) -> Path:
    png_path = input_dir / f"{name}.png"
    json_path = input_dir / f"{name}.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_rgb.save(png_path)
    payload = {
        "caption": global_caption,
        "seed": int(seed),
        "segments_info": _segments_from_mask(mask_rgb),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def _draw_binary_contour(mask: np.ndarray, thickness: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool)
    boundary = np.zeros_like(mask_bool)
    boundary[:-1, :] |= mask_bool[:-1, :] != mask_bool[1:, :]
    boundary[1:, :] |= mask_bool[1:, :] != mask_bool[:-1, :]
    boundary[:, :-1] |= mask_bool[:, :-1] != mask_bool[:, 1:]
    boundary[:, 1:] |= mask_bool[:, 1:] != mask_bool[:, :-1]
    boundary &= mask_bool
    radius = max(0, int(thickness) - 1)
    if radius <= 0:
        return boundary
    padded = np.pad(boundary, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(boundary)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy : dy + boundary.shape[0], dx : dx + boundary.shape[1]]
    return dilated


def _draw_contours(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    thickness: int,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    out = image.copy()
    for idx, mask in enumerate(labels):
        color = colors[idx] if idx < len(colors) else (255, 255, 255)
        contour = _draw_binary_contour(mask, thickness)
        out[contour] = np.asarray(color, dtype=np.uint8)
    return out


def _make_seg2any_batch(
    *,
    seg_map_path: Path,
    seg_anno_path: Path,
    cond_scale_factor: int,
) -> dict[str, Any]:
    seg_map = Image.open(seg_map_path).convert("RGB")
    img_w, img_h = seg_map.size
    seg_array = np.asarray(seg_map, dtype=np.uint8)

    seg_anno = json.loads(seg_anno_path.read_text(encoding="utf-8"))
    scale = int(cond_scale_factor) * 16
    cond_resolution = [img_h // scale * 16, img_w // scale * 16]
    if cond_resolution[0] <= 0 or cond_resolution[1] <= 0:
        raise ValueError(
            f"Invalid Seg2Any condition resolution {cond_resolution} for image {(img_w, img_h)} "
            f"and cond_scale_factor={cond_scale_factor}"
        )

    cond_transforms = transforms.Compose(
        [
            transforms.Resize(cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    color_to_text = {
        tuple(int(v) for v in region["color"]): str(region["text"])
        for region in seg_anno["segments_info"]
    }
    labels = []
    regional_captions = []
    color_list = np.unique(seg_array.reshape(-1, 3), axis=0)
    for color_arr in color_list:
        color = tuple(int(v) for v in color_arr.tolist())
        if color not in color_to_text:
            continue
        mask = (
            (seg_array[..., 0] == color[0])
            & (seg_array[..., 1] == color[1])
            & (seg_array[..., 2] == color[2])
        )
        labels.append(mask)
        regional_captions.append(color_to_text[color])

    if not labels:
        raise ValueError(f"No annotated color regions found in {seg_map_path}")

    label = torch.from_numpy(np.stack(labels, axis=0)).long()
    cond_pixels = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
    cond_pixels = _draw_contours(
        cond_pixels,
        label.cpu().numpy(),
        thickness=1,
        colors=[(255, 255, 255)] * len(regional_captions),
    )
    cond_pixel_values = cond_transforms(Image.fromarray(cond_pixels))

    return {
        "label": label,
        "regional_captions": regional_captions,
        "global_caption": str(seg_anno["caption"]),
        "cond_pixel_values": cond_pixel_values,
        "image_width": cond_resolution[1] * int(cond_scale_factor),
        "image_height": cond_resolution[0] * int(cond_scale_factor),
        "seed": seg_anno.get("seed"),
    }


def _load_seg2any_pipeline(
    *,
    seg2any_root: Path,
    pretrained_model_name_or_path: str,
    lora_ckpt_path: Path,
    dtype: torch.dtype,
    device: torch.device,
    pipeline_progress: bool,
):
    sys.path.insert(0, str(seg2any_root))
    from src.models import FluxTransformer2DModel
    from src.pipelines import FluxRegionalPipeline

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipeline = FluxRegionalPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=dtype,
    )

    default_lora = lora_ckpt_path / "default"
    cond_lora = lora_ckpt_path / "cond"
    if not default_lora.is_dir():
        raise NotADirectoryError(f"Seg2Any LoRA checkpoint must contain default/: {lora_ckpt_path}")
    pipeline.load_lora_weights(str(default_lora), adapter_name="default")
    pipeline.set_adapters("default")
    if cond_lora.is_dir():
        pipeline.load_lora_weights(str(cond_lora), adapter_name="cond")
        pipeline.set_adapters(["cond", "default"])
    pipeline.set_progress_bar_config(disable=not pipeline_progress)
    pipeline = pipeline.to(device)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Seg2Any inference from a Vistar-style JSONL manifest.")
    parser.add_argument("--seg2any_root", default="third_party/Seg2Any")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model_name_or_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_ckpt_path", default="/root/data/weight/seg2any/sacap_1m/seg2any/checkpoint-20000")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--cond_scale_factor", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--cond2image_attention_weight", type=float, default=1.0)
    parser.add_argument("--regional_max_sequence_length", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--attention_mask_method", default="hard", choices=["hard", "base", "place"])
    parser.add_argument("--hard_attn_block_start", type=int, default=19)
    parser.add_argument("--hard_attn_block_end", type=int, default=37)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pipeline_progress", action="store_true")
    parser.add_argument(
        "--global_caption",
        default="A high-resolution remote sensing satellite image with buildings, roads, water, barren land, forest, and agriculture.",
    )
    args = parser.parse_args()

    seg2any_root = _resolve_path(args.seg2any_root)
    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    lora_ckpt_path = _resolve_path(args.lora_ckpt_path)
    pretrained_ref = _normalize_wsl_unc(args.pretrained_model_name_or_path)

    if not seg2any_root.is_dir():
        raise NotADirectoryError(f"Seg2Any root not found: {seg2any_root}")
    if not lora_ckpt_path.is_dir():
        raise NotADirectoryError(f"Seg2Any LoRA checkpoint not found: {lora_ckpt_path}")
    if _is_local_ref(pretrained_ref) and not Path(pretrained_ref).expanduser().exists():
        raise FileNotFoundError(f"FLUX.1-dev local path not found: {pretrained_ref}")
    if int(args.resolution) % (16 * int(args.cond_scale_factor)) != 0:
        raise ValueError(
            f"resolution={args.resolution} must be divisible by 16 * cond_scale_factor={16 * int(args.cond_scale_factor)}"
        )

    rows = _read_manifest(manifest, int(args.max_samples))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")

    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb", "seg2any_inputs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    pipeline = _load_seg2any_pipeline(
        seg2any_root=seg2any_root,
        pretrained_model_name_or_path=pretrained_ref,
        lora_ckpt_path=lora_ckpt_path,
        dtype=dtype,
        device=device,
        pipeline_progress=bool(args.pipeline_progress),
    )

    resolved_manifest = output_dir / "manifest_resolved.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as manifest_f:
        for index, row in enumerate(tqdm(rows, desc="Seg2Any inference")):
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

            has_valid_prediction = _is_valid_rgb_file(pred_path, int(args.eval_size))
            status = "skipped_existing" if has_valid_prediction and not args.overwrite else "pending"
            input_json = _write_seg2any_input(
                name=name,
                mask_rgb=condition_image,
                input_dir=output_dir / "seg2any_inputs",
                global_caption=str(row.get("prompt") or args.global_caption),
                seed=int(args.seed) + index,
            )
            input_png = input_json.with_suffix(".png")

            if status == "pending":
                batch = _make_seg2any_batch(
                    seg_map_path=input_png,
                    seg_anno_path=input_json,
                    cond_scale_factor=int(args.cond_scale_factor),
                )
                generator = torch.Generator(device=device).manual_seed(int(batch["seed"]))
                with torch.inference_mode():
                    image = pipeline(
                        global_prompt=batch["global_caption"],
                        regional_prompts=batch["regional_captions"],
                        regional_labels=batch["label"],
                        cond=(batch["cond_pixel_values"] + 1) / 2.0,
                        attention_mask_method=str(args.attention_mask_method),
                        is_filter_cond_token=True,
                        cond2image_attention_weight=float(args.cond2image_attention_weight),
                        hard_attn_block_range=[int(args.hard_attn_block_start), int(args.hard_attn_block_end)],
                        height=int(batch["image_height"]),
                        width=int(batch["image_width"]),
                        cond_scale_factor=int(args.cond_scale_factor),
                        num_images_per_prompt=1,
                        guidance_scale=float(args.guidance_scale),
                        num_inference_steps=int(args.num_inference_steps),
                        generator=generator,
                        max_sequence_length=int(args.max_sequence_length),
                        regional_max_sequence_length=int(args.regional_max_sequence_length),
                    ).images[0]
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
                        "seg2any_input_png": str(input_png),
                        "seg2any_input_json": str(input_json),
                        "resolution": int(args.resolution),
                        "eval_size": int(args.eval_size),
                        "cond_scale_factor": int(args.cond_scale_factor),
                        "num_inference_steps": int(args.num_inference_steps),
                        "guidance_scale": float(args.guidance_scale),
                        "attention_mask_method": str(args.attention_mask_method),
                        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
                        "lora_ckpt_path": str(args.lora_ckpt_path),
                        "status": status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[run_seg2any_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
