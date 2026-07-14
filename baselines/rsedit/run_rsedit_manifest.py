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
        raise ImportError("Install requirements-rsedit.txt before running RSEdit.") from exc


SECOND_CLASSES = {
    0: "unchanged", 1: "water", 2: "bare land", 3: "low vegetation",
    4: "tree", 5: "buildings", 6: "playgrounds",
}
SECOND_PALETTE = np.asarray([
    [255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
    [0, 255, 0], [128, 0, 0], [255, 255, 0],
], dtype=np.uint8) if np is not None else None
OUTPUT_DIRS = ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts")


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def prompt_for(names: list[str]) -> str:
    if not names:
        return "Keep this remote sensing image unchanged."
    joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return (
        f"Edit this remote sensing image to create realistic changes involving {joined}, "
        "while preserving geographically unchanged regions, viewing angle, scale, and image style."
    )


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC) if image.size != (size, size) else image


def load_ids(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public RSEdit UNet checkpoint on a SECOND manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="BiliSakura/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("RSEdit inference requires CUDA")
    torch.cuda.set_device(args.local_rank)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(f"cuda:{args.local_rank}")
    pipe.set_progress_bar_config(disable=args.rank != 0)

    rows = [json.loads(line) for line in resolve(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    rows = rows[args.rank::args.world_size]
    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []

    for row in tqdm(rows, desc=f"RSEdit SECOND rank {args.rank}", disable=args.rank != 0):
        name = safe_name(str(row["name"]))
        source = load_rgb(resolve(row["source_image"]), args.resolution)
        target = load_rgb(resolve(row["target_image"]), args.eval_size)
        mask_ids = load_ids(resolve(row["condition_mask_ids"]), args.eval_size)
        class_ids = sorted(int(value) for value in np.unique(mask_ids) if int(value) > 0)
        class_names = [SECOND_CLASSES[value] for value in class_ids]
        prompt = prompt_for(class_names)
        pred_path = dirs["pred_rgb"] / f"{name}_pred_rgb.png"
        if pred_path.is_file() and not args.overwrite:
            status = "skipped_existing"
        elif not class_ids:
            save(source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "source_copy_no_change"
        else:
            digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            generator = torch.Generator(device=f"cuda:{args.local_rank}").manual_seed(args.seed + digest)
            prediction = pipe(
                prompt=prompt, image=source, generator=generator,
                num_inference_steps=args.num_inference_steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
            ).images[0].convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
            save(prediction, pred_path)
            status = "generated"

        source_eval = source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
        save(source_eval, dirs["source_rgb"] / f"{name}_source_rgb.png")
        save(target, dirs["gt_rgb"] / f"{name}_gt_rgb.png")
        save(Image.fromarray(mask_ids, "L"), dirs["cond_mask_ids"] / f"{name}_cond_mask_ids.png")
        save(Image.fromarray(np.repeat(np.where(mask_ids[..., None] > 0, 255, 0).astype(np.uint8), 3, axis=2), "RGB"), dirs["cond_mask"] / f"{name}_cond_mask.png")
        save(Image.fromarray(SECOND_PALETTE[mask_ids.clip(0, 6)], "RGB"), dirs["cond_mask_official"] / f"{name}_cond_mask_official.png")
        pred = Image.open(pred_path).convert("RGB")
        diff = np.abs(np.asarray(pred, np.int16) - np.asarray(target, np.int16)).astype(np.uint8)
        save(Image.fromarray(diff, "RGB"), dirs["absdiff"] / f"{name}_absdiff.png")
        (dirs["prompts"] / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata.append({
            "name": name, "status": status, "dataset": "SECOND", "direction": row.get("direction"),
            "changed_class_ids": class_ids, "changed_class_names": class_names, "prompt": prompt,
            "model": args.model, "condition_passed_to_model": ["source_rgb", "text_prompt"],
            "ground_truth_change_mask_passed_to_model": False, "model_outputs": ["pred_rgb"],
        })

    with (output / f"prompts_rank{args.rank:02d}.jsonl").open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.rank == 0:
        (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    print(f"[run_rsedit_manifest] rank={args.rank} wrote {len(metadata)} records")


if __name__ == "__main__":
    main()
