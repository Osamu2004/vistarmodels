from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
            "AnySD dependencies are missing. Install requirements-anysd.txt and a CUDA-matched "
            "PyTorch build first."
        ) from exc


SECOND_CLASSES = {
    0: "unchanged",
    1: "water",
    2: "bare land",
    3: "low vegetation",
    4: "tree",
    5: "buildings",
    6: "playgrounds",
}
SECOND_PROMPT_CLASSES = {
    0: "unchanged",
    1: "changed inland water",
    2: "changed bare land",
    3: "changed grass",
    4: "changed forest",
    5: "changed building",
    6: "changed playground",
}
SECOND_PROMPT_COLORS = {
    "changed inland water": "blue",
    "changed bare land": "gray",
    "changed grass": "dark green",
    "changed forest": "bright green",
    "changed building": "brown",
    "changed playground": "yellow",
}
SECOND_PALETTE = np.asarray(
    [
        [255, 255, 255],
        [0, 0, 255],
        [128, 128, 128],
        [0, 128, 0],
        [0, 255, 0],
        [128, 0, 0],
        [255, 255, 0],
    ],
    dtype=np.uint8,
) if np is not None else None
OUTPUT_DIRS = (
    "source_rgb",
    "cond_mask",
    "cond_mask_official",
    "cond_mask_ids",
    "model_input_source_rgb",
    "model_input_visual_segment",
    "gt_rgb",
    "pred_rgb",
    "absdiff",
    "prompts",
)


def resolve(value: str) -> Path:
    text = str(value)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [part for part in text.strip("\\").split("\\") if part]
            if len(parts) >= 3:
                text = "/" + "/".join(parts[2:])
            break
    return Path(text).expanduser().resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC) if image.size != (size, size) else image


def load_ids(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    values = np.asarray(image, dtype=np.uint8)
    if np.any(values > max(SECOND_CLASSES)):
        raise ValueError(f"Semantic condition contains class IDs outside 0..{max(SECOND_CLASSES)}: {path}")
    return values


def official_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(SECOND_PALETTE[mask_ids], "RGB")


def binary_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.where(mask_ids[..., None] > 0, [255, 255, 255], [0, 0, 0]).astype(np.uint8),
        "RGB",
    )


def format_color_map() -> str:
    return "; ".join(
        f"{SECOND_PROMPT_CLASSES[class_id]}=RGB({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        for class_id, rgb in enumerate(SECOND_PALETTE)
    )


def format_prompt_color_map(class_names: list[str]) -> str:
    # Include only classes present in this mask. This keeps the instruction
    # within CLIP's 77-token limit while preserving exact text/mask agreement.
    entries = ["white unchanged"]
    entries.extend(
        f"{SECOND_PROMPT_COLORS[class_name]} {class_name}"
        for class_name in class_names
    )
    return ", ".join(entries)


def valid(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == (size, size)
    except Exception:
        return False


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def prompt_for(
    *,
    class_names: list[str],
    direction: str,
    mode: str,
    mask_mode: str,
) -> str:
    if direction not in {"t1_to_t2", "t2_to_t1"}:
        raise ValueError(f"Unsupported SECOND direction: {direction!r}")
    if mask_mode == "oneclass":
        if len(class_names) != 1:
            raise ValueError(
                f"One-class AnySD prompt expects one category, got {class_names}"
            )
        class_name = class_names[0]
        if mode == "official_visual_segment":
            return (
                f"Follow the given segmentation image [V*] to add {class_name} in the indicated regions "
                "of this overhead satellite image while preserving all other geographic content"
            )
        if mode == "short_visual_segment":
            return f"Follow the given segmentation image [V*] to add {class_name}"
        if mode == "text_only":
            return (
                f"Change the indicated regions into {class_name} while preserving "
                "the rest of the satellite image"
            )
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    if mask_mode != "full_multiclass":
        raise ValueError(f"Unsupported mask mode: {mask_mode!r}")
    target_time = "post-change" if direction == "t1_to_t2" else "pre-change"
    source_time = "pre-change" if direction == "t1_to_t2" else "post-change"
    present = ", ".join(class_names) if class_names else "no changes"
    if mode == "official_visual_segment":
        return (
            "Use segmentation [V*] to edit the "
            f"{source_time} overhead image into the {target_time} image. "
            f"Color map: {format_prompt_color_map(class_names)}."
        )
    if mode == "short_visual_segment":
        return (
            "Follow the full semantic segmentation image [V*] to generate the "
            f"corresponding {target_time} overhead image with {present}"
        )
    if mode == "text_only":
        return (
            f"Generate the corresponding {target_time} overhead satellite image. "
            f"Preserve unchanged regions and change the indicated regions into: {present}."
        )
    raise ValueError(f"Unsupported prompt mode: {mode!r}")


def stable_generation_seed(
    base_seed: int, sample_name: str, direction: str
) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}\0{sample_name}\0{direction}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (
        2**63 - 1
    )


def prompt_token_count(pipe: Any, prompt: str) -> int:
    tokenized = pipe.tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = tokenized["input_ids"]
    count = len(input_ids)
    maximum = int(pipe.tokenizer.model_max_length)
    if count > maximum:
        raise ValueError(
            f"AnySD prompt has {count} CLIP tokens, exceeding {maximum}: {prompt}"
        )
    return count


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_visual_segment_pipeline(args: argparse.Namespace, device: torch.device):
    anysd_root = resolve(args.anysd_root)
    if not (anysd_root / "anysd/src/pipe.py").is_file():
        raise FileNotFoundError(f"Official AnySD source is missing: {anysd_root}")
    sys.path.insert(0, str(anysd_root))

    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPVisionModelWithProjection
    from anysd.src.pipe import AnySDInstructPix2PixPipeline
    from anysd.src.unet import UNet2DConditionAnySD

    model_dir = resolve(args.model)
    base_dir = resolve(args.base_model)
    dtype = torch.float16
    unet = UNet2DConditionAnySD.from_pretrained(
        model_dir, subfolder="unet", torch_dtype=dtype, local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_dir, subfolder="image_encoder", torch_dtype=dtype, local_files_only=True
    )
    pipe = AnySDInstructPix2PixPipeline.from_pretrained(
        base_dir,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        image_encoder=image_encoder,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
        local_files_only=True,
    )
    task_embs = torch_load(model_dir / "experts/task_embs.bin")
    pipe.load_ip_adapter(str(model_dir), subfolder="experts", weight_name="visual_seg.bin")
    pipe.set_ip_adapter_scale(args.reference_image_guidance_scale)
    pipe = pipe.to(device)
    # DiffusionPipeline.to() moves registered modules, but task_embs is an
    # arbitrary pipeline attribute rather than a registered module. Assign it
    # only after the pipeline move so every torchrun rank uses its local GPU.
    pipe.task_embs = torch.nn.Parameter(
        task_embs.to(device=device, dtype=dtype),
        requires_grad=False,
    )
    if pipe.task_embs.device != device:
        raise RuntimeError(
            f"AnySD task embeddings are on {pipe.task_embs.device}, expected {device}"
        )
    print(
        f"[run_anysd_manifest] rank={args.rank} "
        f"task_embs_device={pipe.task_embs.device}"
    )
    if args.xformers:
        pipe.enable_xformers_memory_efficient_attention()
    if args.vae_tiling:
        pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=args.rank != 0)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official AnySD visual-segment editing on SECOND.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anysd_root", default="third_party/AnySD")
    parser.add_argument("--model", default="/root/data/weight/anysd/AnySD")
    parser.add_argument("--base_model", default="/root/data/weight/anysd/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--image_guidance_scale", type=float, default=2.0)
    parser.add_argument("--reference_image_guidance_scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--only_changed", action="store_true")
    parser.add_argument(
        "--mask_mode",
        choices=["full_multiclass", "oneclass"],
        default="full_multiclass",
        help=(
            "Use every category in the target-side directional mask by default. "
            "oneclass preserves the former shared random-class protocol."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["official_visual_segment", "short_visual_segment", "text_only"],
        default="official_visual_segment",
    )
    parser.add_argument("--xformers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vae_tiling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    args = parser.parse_args()

    if args.resolution <= 0 or args.eval_size <= 0 or args.resolution % 8 != 0:
        raise ValueError("AnySD resolution must be positive and divisible by 8; eval_size must be positive")
    if args.guidance_scale <= 1.0 or args.image_guidance_scale < 1.0:
        raise ValueError("AnySD visual editing expects guidance_scale > 1 and image_guidance_scale >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("AnySD inference requires CUDA")
    if not 0 <= args.local_rank < torch.cuda.device_count():
        raise RuntimeError(f"local_rank={args.local_rank} has no visible CUDA device")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    all_rows = load_rows(resolve(args.manifest), 0)
    if args.only_changed:
        if args.mask_mode == "full_multiclass":
            all_rows = [
                row for row in all_rows if row.get("present_class_ids", [])
            ]
        else:
            all_rows = [
                row
                for row in all_rows
                if int(row.get("selected_class_id", -1)) > 0
            ]
    if args.max_samples > 0:
        all_rows = all_rows[: args.max_samples]
    rows = all_rows[args.rank::args.world_size]
    if not all_rows:
        raise ValueError("Manifest has no rows to generate")
    requires_model = (
        bool(rows)
        if args.mask_mode == "full_multiclass"
        else any(
            int(row.get("selected_class_id", -1)) > 0 for row in rows
        )
    )
    pipe = load_visual_segment_pipeline(args, device) if requires_model else None

    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    print(
        f"[run_anysd_manifest] rank={args.rank}/{args.world_size} device={device} "
        f"records={len(rows)}/{len(all_rows)} expert=visual_segment "
        f"mask_mode={args.mask_mode}"
    )

    for row_index, row in enumerate(tqdm(rows, desc=f"AnySD SECOND rank {args.rank}", disable=args.rank != 0)):
        if str(row.get("consumer", "")) != "anysd":
            raise ValueError(
                "Manifest was not built for AnySD. Rebuild with "
                "tools/build_anysd_second_manifest.py for full_multiclass or "
                "tools/build_rcdgen_second_manifest.py --consumer anysd for oneclass."
            )
        source_path = resolve(str(row["source_image"]))
        target_path = resolve(str(row["target_image"]))
        direction = str(row["direction"])
        base = safe_name(str(row.get("name", f"sample_{row_index:06d}")))
        if args.mask_mode == "full_multiclass":
            if row.get("protocol") != "second_full_multiclass_targetmask_v1":
                raise ValueError(
                    "Manifest does not use the full multi-class AnySD protocol. "
                    "Rebuild it with tools/build_anysd_second_manifest.py."
                )
            mask_path = resolve(str(row["semantic_change_mask"]))
            present_class_ids = [
                int(value) for value in row.get("present_class_ids", [])
            ]
            if any(
                value <= 0 or value not in SECOND_CLASSES
                for value in present_class_ids
            ):
                raise ValueError(
                    f"Invalid full multi-class IDs in manifest row: {row}"
                )
            expected_prompt_names = [
                SECOND_PROMPT_CLASSES[value]
                for value in present_class_ids
            ]
            if list(row.get("present_prompt_names", [])) != expected_prompt_names:
                raise ValueError(
                    "Manifest prompt vocabulary does not match the AnySD runner "
                    f"for {base}: manifest={row.get('present_prompt_names')}, "
                    f"runner={expected_prompt_names}"
                )
            class_id = None
        else:
            mask_path = resolve(
                str(row["selected_semantic_change_mask"])
            )
            class_id = int(row["selected_class_id"])
            selected_class_name = str(row["selected_class_name"])
            if (
                class_id not in SECOND_CLASSES
                or selected_class_name != SECOND_CLASSES[class_id]
            ):
                raise ValueError(
                    f"Invalid shared class selection in manifest row: {row}"
                )
            present_class_ids = [class_id] if class_id > 0 else []
        source = load_rgb(source_path, args.resolution)
        target = load_rgb(target_path, args.eval_size)
        semantic_ids = load_ids(mask_path, args.resolution)
        actual_present_ids = sorted(
            int(value)
            for value in np.unique(semantic_ids)
            if int(value) > 0
        )
        if args.mask_mode == "full_multiclass" and not set(
            actual_present_ids
        ).issubset(present_class_ids):
            raise ValueError(
                "Manifest class list does not match the semantic mask for "
                f"{base}: manifest={present_class_ids}, mask={actual_present_ids}"
            )
        if args.mask_mode == "full_multiclass":
            # Nearest-neighbor downsampling may remove an extremely thin
            # class. Keep the text synchronized with the mask actually seen
            # by AnySD at the requested inference resolution.
            present_class_ids = actual_present_ids
            prompt_class_names = [
                SECOND_PROMPT_CLASSES[value]
                for value in present_class_ids
            ]
        else:
            prompt_class_names = [SECOND_PROMPT_CLASSES[class_id]]
        class_name = ", ".join(prompt_class_names) or "unchanged"
        prompt = prompt_for(
            class_names=prompt_class_names,
            direction=direction,
            mode=args.prompt_mode,
            mask_mode=args.mask_mode,
        )
        visual_segment = official_visual(semantic_ids)
        pred_path = dirs["pred_rgb"] / f"{base}_pred_rgb.png"
        token_count: int | None = None

        if valid(pred_path, args.eval_size) and not args.overwrite:
            status = "skipped_existing"
        elif args.mask_mode == "oneclass" and class_id == 0:
            save(source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "source_copy_no_change"
        else:
            assert pipe is not None
            token_count = prompt_token_count(pipe, prompt)
            generation_seed = stable_generation_seed(
                args.seed, str(row.get("sample_name", base)), direction
            )
            generator = torch.Generator(device=device).manual_seed(generation_seed)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                result = pipe(
                    prompt=prompt,
                    image=source,
                    ip_adapter_image=visual_segment,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    generator=generator,
                    e_code=12,
                ).images[0]
            result = result.convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
            save(result, pred_path)
            status = "generated_black" if np.asarray(result).max() == 0 else "generated"

        eval_source = source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
        eval_ids = np.asarray(
            Image.fromarray(semantic_ids, "L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        eval_visual = official_visual(eval_ids)
        save(eval_source, dirs["source_rgb"] / f"{base}_source_rgb.png")
        save(target, dirs["gt_rgb"] / f"{base}_gt_rgb.png")
        save(Image.fromarray(eval_ids, "L"), dirs["cond_mask_ids"] / f"{base}_cond_mask_ids.png")
        save(
            eval_visual
            if args.mask_mode == "full_multiclass"
            else binary_visual(eval_ids),
            dirs["cond_mask"] / f"{base}_cond_mask.png",
        )
        save(eval_visual, dirs["cond_mask_official"] / f"{base}_cond_mask_official.png")
        save(source, dirs["model_input_source_rgb"] / f"{base}_model_input_source_rgb.png")
        save(visual_segment, dirs["model_input_visual_segment"] / f"{base}_model_input_visual_segment.png")
        pred = Image.open(pred_path).convert("RGB")
        diff = np.abs(np.asarray(pred, dtype=np.int16) - np.asarray(target, dtype=np.int16)).astype(np.uint8)
        save(Image.fromarray(diff, "RGB"), dirs["absdiff"] / f"{base}_absdiff.png")
        (dirs["prompts"] / f"{base}.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata_item = {
            "name": base,
            "status": status,
            "dataset": "SECOND",
            "direction": direction,
            "category": class_name,
            "present_class_ids": present_class_ids,
            "present_class_names": prompt_class_names,
            "prompt": prompt,
            "prompt_token_count": token_count,
            "prompt_mode": args.prompt_mode,
            "expert": "visual_segment",
            "mask_mode": args.mask_mode,
            "mask_protocol": row.get("protocol")
            if args.mask_mode == "full_multiclass"
            else "second_oneclass_targetmask_v1",
            "category_selection_policy": (
                "all_target_side_changed_categories"
                if args.mask_mode == "full_multiclass"
                else "shared_second_oneclass_targetmask_v1"
            ),
            "condition_passed_to_model": [
                "source_rgb",
                (
                    "full_multiclass_semantic_change_mask"
                    if args.mask_mode == "full_multiclass"
                    else "selected_semantic_change_mask"
                ),
                "text_prompt",
            ],
            "ground_truth_change_mask_passed_to_model": True,
            "model_outputs": ["pred_rgb"],
            "source_image": str(source_path),
            "target_image": str(target_path),
            "semantic_change_mask": str(mask_path),
            "pred_rgb": str(pred_path),
        }
        if args.mask_mode == "oneclass":
            metadata_item.update(
                {
                    "class_id": class_id,
                    "selected_class_id": class_id,
                    "selected_class_name": row.get(
                        "selected_class_name"
                    ),
                    "selected_semantic_change_mask": str(mask_path),
                    "class_selection_file": row.get(
                        "class_selection_file"
                    ),
                    "class_selection_record": row.get(
                        "class_selection_record"
                    ),
                }
            )
        metadata.append(metadata_item)

    metadata_path = output / f"prompts_rank{args.rank:02d}.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.rank == 0:
        class_map = {
            "dataset": "SECOND",
            "condition_mask_mode": "semantic",
            "mask_mode": args.mask_mode,
            "classes": [
                {
                    "id": class_id,
                    "label_name": SECOND_CLASSES[class_id],
                    "prompt_name": SECOND_PROMPT_CLASSES[class_id],
                    "rgb": SECOND_PALETTE[class_id].tolist(),
                }
                for class_id in SECOND_CLASSES
            ],
            "color_map": format_color_map(),
        }
        (output / "class_map.json").write_text(
            json.dumps(class_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[run_anysd_manifest] rank={args.rank} wrote {len(metadata)} outputs")


if __name__ == "__main__":
    main()
