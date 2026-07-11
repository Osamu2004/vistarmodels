from __future__ import annotations

import argparse
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
            "RCDGen inference dependencies are missing. Install requirements-rcdgen.txt "
            "and a CUDA-matched PyTorch build first."
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
# Canonical names above must match Vistar's shared JSONL.  RCDGen was trained
# with these category strings, so retain its native prompt vocabulary while
# keeping the class-id contract unchanged.
RCDGEN_PROMPT_CLASSES = {
    0: "unchanged",
    1: "water bodies",
    2: "non-vegetated ground surface",
    3: "low vegetation",
    4: "tree",
    5: "building",
    6: "playground",
}
SECOND_PALETTE = np.asarray(
    [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
     [0, 255, 0], [128, 0, 0], [255, 255, 0]],
    dtype=np.uint8,
) if np is not None else None
OUTPUT_DIRS = (
    "source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids",
    "gt_rgb", "pred_rgb", "pred_change_mask", "absdiff", "prompts",
)


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC) if image.size != (size, size) else image


def ids(path: Path, size: int) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"L", "P", "I", "I;16"}:
        array = np.asarray(image)
    else:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int32)
        palette = SECOND_PALETTE.astype(np.int32)
        distance = ((pixels[..., None, :] - palette[None, None, :, :]) ** 2).sum(axis=-1)
        array = distance.argmin(axis=-1).astype(np.uint8)
    image = Image.fromarray(array.astype(np.uint8), "L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def valid(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == (size, size)
    except Exception:
        return False


def binary_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.where(mask_ids[..., None] > 0, [255, 255, 255], [0, 0, 0]).astype(np.uint8),
        "RGB",
    )


def official_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(SECOND_PALETTE[mask_ids.clip(0, len(SECOND_CLASSES) - 1)], "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official RCDGen on a SECOND JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="/root/data/weight/rcdgen/RCDGen")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("RCDGen inference requires CUDA")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid distributed shard: rank={args.rank}, world_size={args.world_size}")
    if not 0 <= args.local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"local_rank={args.local_rank} has no visible CUDA device; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
            f"visible_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.local_rank)
    device = f"cuda:{args.local_rank}"

    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.RCDGenSDPipeline import StableDiffusionInstructPix2PixPipeline

    model = args.model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet_ema", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(resolve(args.manifest), args.max_samples)
    rows = all_rows[args.rank::args.world_size]
    metadata_path = output / f"prompts_rank{args.rank:02d}.jsonl"
    metadata: list[dict[str, Any]] = []
    print(
        f"[run_rcdgen_manifest] rank={args.rank}/{args.world_size} device={device} "
        f"records={len(rows)}/{len(all_rows)}"
    )

    for row_index, row in enumerate(
        tqdm(rows, desc=f"RCDGen SECOND rank {args.rank}", disable=args.rank != 0)
    ):
        source_path = resolve(str(row["source_image"]))
        target_path = resolve(str(row["target_image"]))
        selected_mask_path = resolve(str(row["selected_semantic_change_mask"]))
        class_selection_file = resolve(str(row["class_selection_file"]))
        selected_class_id = int(row["selected_class_id"])
        selected_class_name = str(row["selected_class_name"])
        if selected_class_id not in SECOND_CLASSES or selected_class_name != SECOND_CLASSES[selected_class_id]:
            raise ValueError(f"Invalid shared class selection in manifest row: {row}")
        source_image = rgb(source_path, args.resolution)
        target_image = rgb(target_path, args.eval_size)
        selected_ids = ids(selected_mask_path, args.resolution)
        if selected_class_id > 0 and not bool((selected_ids == selected_class_id).any()):
            raise ValueError(
                f"Selected class {selected_class_id} disappeared after resize for manifest row {row['name']!r}. "
                "Rebuild the manifest from the unchanged shared protocol."
            )
        row_name = str(row.get("name", f"sample_{row_index:06d}"))
        base = safe_name(row_name)
        prompt = f"change in {RCDGEN_PROMPT_CLASSES[selected_class_id]}"
        pred_path = dirs["pred_rgb"] / f"{base}_pred_rgb.png"
        pred_mask_path = dirs["pred_change_mask"] / f"{base}_pred_change_mask.png"
        if valid(pred_path, args.eval_size) and valid(pred_mask_path, args.eval_size) and not args.overwrite:
            status = "skipped_existing"
        elif selected_class_id == 0:
            # The protocol represents a no-change pair with class 0.  Asking a
            # change generator to invent a class here would invalidate its GT.
            save(source_image.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            save(Image.new("L", (args.eval_size, args.eval_size), color=0), pred_mask_path)
            status = "source_copy_no_change"
        else:
            draw_seed = int(dict(row.get("class_selection_record", {})).get("draw_seed", 0))
            generator = torch.Generator(device).manual_seed((int(args.seed) + draw_seed) % (2**63 - 1))
            result = pipe(
                prompt,
                image=source_image,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images
            pred = result[0][0].convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
            pred_mask = result[1][0].convert("L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
            save(pred, pred_path)
            save(pred_mask, pred_mask_path)
            status = "generated"

        source_out = dirs["source_rgb"] / f"{base}_source_rgb.png"
        gt_out = dirs["gt_rgb"] / f"{base}_gt_rgb.png"
        cond_l = Image.fromarray(selected_ids.astype(np.uint8), "L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
        cond_ids = np.asarray(cond_l, dtype=np.uint8)
        save(source_image.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), source_out)
        save(target_image, gt_out)
        save(cond_l, dirs["cond_mask_ids"] / f"{base}_cond_mask_ids.png")
        save(binary_visual(cond_ids), dirs["cond_mask"] / f"{base}_cond_mask.png")
        save(official_visual(cond_ids), dirs["cond_mask_official"] / f"{base}_cond_mask_official.png")
        pred = Image.open(pred_path).convert("RGB")
        diff = np.abs(np.asarray(pred, dtype=np.int16) - np.asarray(target_image, dtype=np.int16)).astype(np.uint8)
        save(Image.fromarray(diff, "RGB"), dirs["absdiff"] / f"{base}_absdiff.png")
        (dirs["prompts"] / f"{base}.txt").write_text(prompt + "\n", encoding="utf-8")
        selection_record = dict(row.get("class_selection_record", {}))
        metadata.append({
            "name": base, "status": status, "dataset": "SECOND", "direction": row.get("direction"),
            "category": selected_class_name, "prompt_category": RCDGEN_PROMPT_CLASSES[selected_class_id],
            "class_id": selected_class_id, "prompt": prompt,
            "category_selection_policy": "shared_second_oneclass_targetmask_v1",
            "class_selection_file": str(class_selection_file),
            "class_selection_record": selection_record,
            "available_changed_target_classes": selection_record.get("available_target_classes", []),
            "condition_passed_to_model": ["source_rgb", "text_prompt"],
            "ground_truth_change_mask_passed_to_model": False,
            "source_image": str(source_path), "target_image": str(target_path),
            "target_change_label": str(row["target_change_label"]),
            "selected_semantic_change_mask": str(selected_mask_path),
            "pred_rgb": str(pred_path), "pred_change_mask": str(pred_mask_path),
        })

    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.rank == 0:
        (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    print(
        f"[run_rcdgen_manifest] rank={args.rank} wrote {len(metadata)} shared-protocol outputs to {output}"
    )


if __name__ == "__main__":
    main()
