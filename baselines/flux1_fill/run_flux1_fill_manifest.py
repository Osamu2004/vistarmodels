from __future__ import annotations

import argparse
import json
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
            "FLUX.1 Fill-dev inference dependencies are missing. Install requirements-flux1-fill.txt "
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
    "model_input_mask",
    "gt_rgb",
    "pred_rgb",
    "absdiff",
    "prompts",
)


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC) if image.size != (size, size) else image


def load_binary_mask(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return Image.fromarray(np.where(np.asarray(image) > 0, 255, 0).astype(np.uint8), "L")


def load_semantic_mask(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def valid(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == (size, size)
    except Exception:
        return False


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def binary_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.where(mask_ids[..., None] > 0, [255, 255, 255], [0, 0, 0]).astype(np.uint8),
        "RGB",
    )


def official_visual(mask_ids: np.ndarray) -> Image.Image:
    return Image.fromarray(SECOND_PALETTE[mask_ids.clip(0, len(SECOND_CLASSES) - 1)], "RGB")


def prompt_for(direction: str, class_name: str, prompt_mode: str) -> str:
    if direction not in {"t1_to_t2", "t2_to_t1"}:
        raise ValueError(f"Unsupported SECOND direction: {direction!r}")
    if prompt_mode == "fill_target":
        # Flux Fill is trained to fill the white region with the *described
        # visual content*.  The binary mask already provides the spatial
        # instruction, so keep this a short image caption rather than adding
        # an unrelated editing instruction.
        return f"an overhead satellite image showing {class_name}"
    if prompt_mode == "legacy_change":
        # Kept solely to reproduce outputs made by the first wrapper.
        return f"change in {class_name}"
    raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")


def load_pipeline(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.1 Fill-dev requires a CUDA GPU for this baseline.")
    from diffusers import FluxFillPipeline

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    pipe = FluxFillPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    if args.vae_tiling:
        pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FLUX.1 Fill-dev using Vistar's shared SECOND one-class manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="/root/data/weight/flux1_fill/FLUX.1-Fill-dev")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--only_changed", action="store_true")
    parser.add_argument(
        "--prompt_mode",
        choices=["fill_target", "legacy_change"],
        default="fill_target",
        help="fill_target is the recommended FLUX Fill caption; legacy_change reproduces the old run.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument(
        "--save_model_inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the exact resolution source and black/white mask passed to Flux Fill for inspection.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resolution <= 0 or args.eval_size <= 0:
        raise ValueError("--resolution and --eval_size must be positive")
    if args.resolution % 16 != 0:
        raise ValueError(f"FLUX.1 Fill-dev resolution must be divisible by 16, got {args.resolution}")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError(f"--strength must be in [0, 1], got {args.strength}")

    output = resolve(args.output_dir)
    dirs = {name: output / name for name in OUTPUT_DIRS}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    rows = load_rows(resolve(args.manifest), 0)
    if args.only_changed:
        rows = [row for row in rows if int(row.get("selected_class_id", -1)) > 0]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("Manifest has no rows to generate")

    # Avoid allocating the 12B model when a smoke manifest contains only
    # no-change examples, whose correct generation is a source-image copy.
    requires_model = any(int(row.get("selected_class_id", -1)) > 0 for row in rows)
    pipe = load_pipeline(args) if requires_model else None
    metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(tqdm(rows, desc="FLUX.1 Fill-dev SECOND records")):
        consumer = str(row.get("consumer", ""))
        if consumer != "flux1_fill":
            raise ValueError(
                "Manifest was not built for FLUX.1 Fill-dev. Rebuild it with "
                "tools/build_rcdgen_second_manifest.py --consumer flux1_fill."
            )
        source_path = resolve(str(row["source_image"]))
        target_path = resolve(str(row["target_image"]))
        binary_mask_path = resolve(str(row["selected_binary_change_mask"]))
        semantic_mask_path = resolve(str(row["selected_semantic_change_mask"]))
        selection_file = resolve(str(row["class_selection_file"]))
        class_id = int(row["selected_class_id"])
        class_name = str(row["selected_class_name"])
        if class_id not in SECOND_CLASSES or class_name != SECOND_CLASSES[class_id]:
            raise ValueError(f"Invalid shared selected class in manifest row: {row}")
        source = load_rgb(source_path, args.resolution)
        target = load_rgb(target_path, args.eval_size)
        binary_mask = load_binary_mask(binary_mask_path, args.resolution)
        semantic_ids = load_semantic_mask(semantic_mask_path, args.resolution)
        binary_mask_ids = np.asarray(binary_mask, dtype=np.uint8)
        binary_mask_active = binary_mask_ids > 0
        semantic_mask_active = semantic_ids > 0
        if not np.array_equal(binary_mask_active, semantic_mask_active):
            raise ValueError(
                "The binary mask sent to Flux Fill does not match the selected semantic mask for "
                f"manifest row {row.get('name')!r}."
            )
        if class_id > 0:
            if not bool((semantic_ids == class_id).any()) or not bool(binary_mask_active.any()):
                raise ValueError(f"Selected mask is empty for manifest row {row.get('name')!r}")
            if np.any((semantic_ids != 0) & (semantic_ids != class_id)):
                raise ValueError(f"Selected semantic mask contains another class for row {row.get('name')!r}")
        elif bool(binary_mask_active.any()) or bool(semantic_mask_active.any()):
            raise ValueError(f"No-change selection has a non-empty mask for row {row.get('name')!r}")

        name = str(row.get("name", f"sample_{row_index:06d}"))
        pred_path = dirs["pred_rgb"] / f"{name}_pred_rgb.png"
        prompt = prompt_for(str(row["direction"]), class_name, args.prompt_mode)
        if args.save_model_inputs:
            save(source, dirs["model_input_source_rgb"] / f"{name}_source_rgb.png")
            save(binary_mask, dirs["model_input_mask"] / f"{name}_white_repaint_black_preserve.png")
        if valid(pred_path, args.eval_size) and not args.overwrite:
            status = "skipped_existing"
        elif class_id == 0:
            save(source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "source_copy_no_change"
        else:
            assert pipe is not None
            draw_seed = int(dict(row.get("class_selection_record", {})).get("draw_seed", 0))
            generator = torch.Generator("cpu").manual_seed((int(args.seed) + draw_seed) % (2**63 - 1))
            generated = pipe(
                prompt=prompt,
                image=source,
                mask_image=binary_mask,
                height=args.resolution,
                width=args.resolution,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
            ).images[0].convert("RGB")
            save(generated.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), pred_path)
            status = "generated"

        semantic_eval = np.asarray(
            Image.fromarray(semantic_ids, "L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        save(source.resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC), dirs["source_rgb"] / f"{name}_source_rgb.png")
        save(target, dirs["gt_rgb"] / f"{name}_gt_rgb.png")
        save(Image.fromarray(semantic_eval, "L"), dirs["cond_mask_ids"] / f"{name}_cond_mask_ids.png")
        save(binary_visual(semantic_eval), dirs["cond_mask"] / f"{name}_cond_mask.png")
        save(official_visual(semantic_eval), dirs["cond_mask_official"] / f"{name}_cond_mask_official.png")
        pred = Image.open(pred_path).convert("RGB")
        diff = np.abs(np.asarray(pred, dtype=np.int16) - np.asarray(target, dtype=np.int16)).astype(np.uint8)
        save(Image.fromarray(diff, "RGB"), dirs["absdiff"] / f"{name}_absdiff.png")
        (dirs["prompts"] / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata.append({
            "name": name,
            "status": status,
            "dataset": "SECOND",
            "direction": row["direction"],
            "model": "black-forest-labs/FLUX.1-Fill-dev",
            "model_path": str(resolve(args.model)),
            "class_id": class_id,
            "class_name": class_name,
            "prompt": prompt,
            "prompt_mode": args.prompt_mode,
            "strength": args.strength,
            "class_selection_file": str(selection_file),
            "class_selection_record": row.get("class_selection_record"),
            "condition_passed_to_model": ["source_rgb", "selected_binary_change_mask", "text_prompt"],
            "ground_truth_change_mask_passed_to_model": True,
            "inpaint_mask_semantics": "white=repaint, black=preserve",
            "mask_white_pixels_at_model_resolution": int(binary_mask_active.sum()),
            "mask_area_ratio_at_model_resolution": float(binary_mask_active.mean()),
            "source_image": str(source_path),
            "target_image": str(target_path),
            "selected_binary_change_mask": str(binary_mask_path),
            "selected_semantic_change_mask": str(semantic_mask_path),
            "model_input_source_rgb": (
                str(dirs["model_input_source_rgb"] / f"{name}_source_rgb.png") if args.save_model_inputs else None
            ),
            "model_input_mask": (
                str(dirs["model_input_mask"] / f"{name}_white_repaint_black_preserve.png")
                if args.save_model_inputs
                else None
            ),
            "pred_rgb": str(pred_path),
        })

    with (output / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "class_map.json").write_text(json.dumps(SECOND_CLASSES, indent=2), encoding="utf-8")
    print(f"[run_flux1_fill_manifest] wrote {len(metadata)} shared-protocol outputs to {output}")


if __name__ == "__main__":
    main()
