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
            "InstructPix2Pix dependencies are missing. Install requirements-instructpix2pix.txt "
            "and a CUDA-matched PyTorch build first."
        ) from exc


SECOND_CLASSES = {
    0: "unchanged", 1: "water", 2: "bare land", 3: "low vegetation",
    4: "tree", 5: "buildings", 6: "playgrounds",
}
SECOND_PALETTE = np.asarray(
    [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
     [0, 255, 0], [128, 0, 0], [255, 255, 0]], dtype=np.uint8,
) if np is not None else None
OUTPUT_DIRS = (
    "source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids",
    "gt_rgb", "pred_rgb", "absdiff", "prompts",
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
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    image = Image.fromarray(array, "L")
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
    return Image.fromarray(np.where(mask_ids[..., None] > 0, 255, 0).astype(np.uint8).repeat(3, axis=2), "RGB")


def official_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(SECOND_PALETTE[mask_ids.clip(0, len(SECOND_CLASSES) - 1)], "RGB")


def edit_prompt(class_id: int, class_name: str) -> str:
    if class_id == 0:
        return "Keep this remote sensing image unchanged."
    singular = {"buildings": "building", "playgrounds": "playground"}.get(class_name, class_name)
    return (
        f"Create a realistic {singular} change in this remote sensing image while preserving "
        "all geographically unchanged regions, viewing angle, scale, and image style."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InstructPix2Pix on the shared SECOND manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="/root/data/weight/instructpix2pix/instruct-pix2pix")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("InstructPix2Pix inference requires CUDA")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"Invalid distributed shard: rank={args.rank}, world_size={args.world_size}")
    if not 0 <= args.local_rank < torch.cuda.device_count():
        raise RuntimeError(f"local_rank={args.local_rank} has no visible CUDA device")
    torch.cuda.set_device(args.local_rank)
    device = f"cuda:{args.local_rank}"

    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=args.rank != 0)

    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(resolve(args.manifest), args.max_samples)
    rows = all_rows[args.rank::args.world_size]
    metadata: list[dict[str, Any]] = []
    print(f"[run_instructpix2pix_manifest] rank={args.rank}/{args.world_size} device={device} records={len(rows)}/{len(all_rows)}")

    for row_index, row in enumerate(tqdm(rows, desc=f"InstructPix2Pix SECOND rank {args.rank}", disable=args.rank != 0)):
        source_path = resolve(str(row["source_image"]))
        target_path = resolve(str(row["target_image"]))
        selected_mask_path = resolve(str(row["selected_semantic_change_mask"]))
        class_id = int(row["selected_class_id"])
        class_name = str(row["selected_class_name"])
        if class_id not in SECOND_CLASSES or class_name != SECOND_CLASSES[class_id]:
            raise ValueError(f"Invalid shared class selection in manifest row: {row}")
        base = safe_name(str(row.get("name", f"sample_{row_index:06d}")))
        prompt = edit_prompt(class_id, class_name)
        source_image = rgb(source_path, args.resolution)
        target_image = rgb(target_path, args.eval_size)
        selected_ids = ids(selected_mask_path, args.resolution)
        pred_path = dirs["pred_rgb"] / f"{base}_pred_rgb.png"

        if valid(pred_path, args.eval_size) and not args.overwrite:
            status = "skipped_existing"
        elif class_id == 0:
            save(source_image.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "source_copy_no_change"
        else:
            draw_seed = int(dict(row.get("class_selection_record", {})).get("draw_seed", 0))
            generator = torch.Generator(device).manual_seed((args.seed + draw_seed) % (2**63 - 1))
            result = pipe(
                prompt=prompt, image=source_image, num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale, guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            save(result.convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "generated"

        source_out = dirs["source_rgb"] / f"{base}_source_rgb.png"
        gt_out = dirs["gt_rgb"] / f"{base}_gt_rgb.png"
        cond_l = Image.fromarray(selected_ids, "L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
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
        metadata.append({
            "name": base, "status": status, "dataset": "SECOND", "direction": row.get("direction"),
            "category": class_name, "class_id": class_id, "prompt": prompt,
            "category_selection_policy": "shared_second_oneclass_targetmask_v1",
            "class_selection_file": row.get("class_selection_file"),
            "class_selection_record": row.get("class_selection_record", {}),
            "condition_passed_to_model": ["source_rgb", "text_prompt"],
            "ground_truth_change_mask_passed_to_model": False,
            "model_outputs": ["pred_rgb"],
            "source_image": str(source_path), "target_image": str(target_path), "pred_rgb": str(pred_path),
        })

    metadata_path = output / f"prompts_rank{args.rank:02d}.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.rank == 0:
        (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    print(f"[run_instructpix2pix_manifest] rank={args.rank} wrote {len(metadata)} outputs")


if __name__ == "__main__":
    main()
