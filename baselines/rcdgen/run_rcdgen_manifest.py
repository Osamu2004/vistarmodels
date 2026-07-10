from __future__ import annotations

import argparse
import json
import random
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
    0: "non-change",
    1: "low vegetation",
    2: "non-vegetated ground surface",
    3: "tree",
    4: "water bodies",
    5: "building",
    6: "playground",
}
SECOND_PALETTE = np.asarray(
    [[255, 255, 255], [0, 128, 0], [128, 128, 128], [0, 255, 0],
     [0, 0, 255], [128, 0, 0], [255, 0, 0]],
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
    parser.add_argument(
        "--category_policy",
        choices=["random", "all"],
        default="random",
        help="random selects one reproducible changed target class per record; all expands every class",
    )
    parser.add_argument(
        "--category",
        default="auto",
        help="auto uses category_policy; otherwise force one official SECOND class name",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("RCDGen inference requires CUDA")

    from diffusers import UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion.RCDGenSDPipeline import StableDiffusionInstructPix2PixPipeline

    model = args.model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet_ema", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)

    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    rows = load_rows(resolve(args.manifest), args.max_samples)
    metadata_path = output / "prompts.jsonl"
    metadata: list[dict[str, Any]] = []

    for row_index, row in enumerate(tqdm(rows, desc="RCDGen SECOND records")):
        source_path = resolve(str(row["source_image"]))
        target_path = resolve(str(row["target_image"]))
        target_mask_path = resolve(str(row["target_mask"]))
        source_image = rgb(source_path, args.resolution)
        target_image = rgb(target_path, args.eval_size)
        target_ids = ids(target_mask_path, args.resolution)
        source_ids = ids(resolve(str(row["source_mask"])), args.resolution)
        changed = source_ids != target_ids
        present = sorted(int(v) for v in np.unique(target_ids[changed]) if int(v) in SECOND_CLASSES and int(v) != 0)
        if args.category != "auto":
            matching = [idx for idx, name in SECOND_CLASSES.items() if name == args.category]
            if not matching:
                raise ValueError(
                    f"unsupported --category={args.category!r}; expected auto or one of "
                    f"{sorted(SECOND_CLASSES.values())}"
                )
            selected = matching
            selection_policy = "fixed"
        elif args.category_policy == "all":
            selected = present or [0]
            selection_policy = "all_present_target_classes"
        else:
            # Use a record-local RNG so selection is reproducible and does not
            # depend on resume state or the number of preceding model calls.
            selector = random.Random(args.seed + row_index * 1009)
            selected = [selector.choice(present)] if present else [0]
            selection_policy = "seeded_random_present_target_class"

        for class_id in selected:
            category = SECOND_CLASSES[class_id]
            row_name = str(row.get("name", f"sample_{row_index:06d}"))
            base = safe_name(f"{row_name}_{category}" if args.category_policy == "all" else row_name)
            prompt = f"change in {category}"
            pred_path = dirs["pred_rgb"] / f"{base}_pred_rgb.png"
            pred_mask_path = dirs["pred_change_mask"] / f"{base}_pred_change_mask.png"
            if valid(pred_path, args.eval_size) and valid(pred_mask_path, args.eval_size) and not args.overwrite:
                status = "skipped_existing"
            else:
                generator = torch.Generator("cuda").manual_seed(args.seed + row_index * 100 + class_id)
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
            cond_ids = np.where(changed & (target_ids == class_id), class_id, 0).astype(np.uint8)
            cond_l = Image.fromarray(cond_ids, "L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
            cond_vis = Image.fromarray(np.where(np.asarray(cond_l)[..., None] > 0, [255, 0, 0], [255, 255, 255]).astype(np.uint8), "RGB")
            save(source_image.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), source_out)
            save(target_image, gt_out)
            save(cond_l, dirs["cond_mask_ids"] / f"{base}_cond_mask_ids.png")
            save(cond_vis, dirs["cond_mask"] / f"{base}_cond_mask.png")
            save(cond_vis, dirs["cond_mask_official"] / f"{base}_cond_mask_official.png")
            pred = Image.open(pred_path).convert("RGB")
            diff = np.abs(np.asarray(pred, dtype=np.int16) - np.asarray(target_image, dtype=np.int16)).astype(np.uint8)
            save(Image.fromarray(diff, "RGB"), dirs["absdiff"] / f"{base}_absdiff.png")
            (dirs["prompts"] / f"{base}.txt").write_text(prompt + "\n", encoding="utf-8")
            metadata.append({
                "name": base, "status": status, "dataset": "SECOND", "direction": row.get("direction"),
                "category": category, "class_id": class_id, "prompt": prompt,
                "category_selection_policy": selection_policy,
                "available_changed_target_classes": [SECOND_CLASSES[idx] for idx in present],
                "selection_seed": args.seed + row_index * 1009,
                "condition_passed_to_model": ["source_rgb", "text_prompt"],
                "ground_truth_change_mask_passed_to_model": False,
                "source_image": str(source_path), "target_image": str(target_path),
                "pred_rgb": str(pred_path), "pred_change_mask": str(pred_mask_path),
            })

    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    print(
        f"[run_rcdgen_manifest] wrote {len(metadata)} category-conditioned outputs to {output}; "
        f"category_policy={args.category_policy}"
    )


if __name__ == "__main__":
    main()
