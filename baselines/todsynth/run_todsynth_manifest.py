from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-dataset TODSynth/CRFM inference for Vistar LoveDA.")
    parser.add_argument("--todsynth_root", required=True)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--checkpoint", required=True, help="training output folder containing model.safetensors")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--vectors_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--num_cls", type=int, default=7)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crfm", action="store_true")
    parser.add_argument("--mmseg_config", default="")
    parser.add_argument("--mmseg_ckpt", default="")
    parser.add_argument("--rectified_step", type=int, default=4)
    parser.add_argument("--max_step_size", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Install requirements-todsynth.txt and the official MMSeg stack when using --crfm.") from exc

    root = Path(args.todsynth_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from src.datasets.infer_dataset import SegmentationDataset
    from src.models.sd3_mmdit import MaskDit_sd3_5
    import src.utils.inference as todsynth_inference
    from src.utils.utils import encode_images
    from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

    # The public inference.py uses this symbol but does not import it.
    todsynth_inference.StableDiffusion3PipelineOutput = StableDiffusion3PipelineOutput
    batch_imgage_generation = todsynth_inference.batch_imgage_generation

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    dtype = torch.float32 if accelerator.mixed_precision == "no" else (torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16)
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model, subfolder="transformer")
    transformer = MaskDit_sd3_5(sd3_transformer=transformer).to(accelerator.device, dtype=dtype)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    transformer.load_state_dict(load_file(str(state_path)), strict=False)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(accelerator.device, dtype=dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    transformer.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()

    control_model = None
    inference_fn = batch_imgage_generation
    if args.crfm:
        if not args.mmseg_config or not args.mmseg_ckpt:
            raise ValueError("--crfm requires --mmseg_config and --mmseg_ckpt")
        from mmseg.apis import init_model
        import src.utils.crfm as todsynth_crfm
        # Same missing import exists in the public crfm.py.
        todsynth_crfm.StableDiffusion3PipelineOutput = StableDiffusion3PipelineOutput
        inference_with_crfm = todsynth_crfm.inference_with_crfm
        control_model = init_model(args.mmseg_config, checkpoint=args.mmseg_ckpt).to(accelerator.device, dtype=dtype).eval()
        control_model.requires_grad_(False)
        inference_fn = inference_with_crfm

    dataset = SegmentationDataset(
        data_root=args.data_root, txt_file=args.json_file, vae_scale_factor=vae_scale_factor,
        vectors_path=args.vectors_path, size=args.resolution, num_cls=args.num_cls, debug=False,
    )

    def collate(examples):
        conditions = examples[0]["condition_types"]
        latents = [torch.stack([example["condition_latents"][index] for example in examples]).float() for index in range(len(conditions))]
        return {
            "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"].squeeze(0) for example in examples]),
            "prompt_embeds": torch.stack([example["prompt_embeds"].squeeze(0) for example in examples]),
            "cond_types": conditions, "cond_latents": latents,
            "img_names": [example["img_name"] for example in examples],
            "control_conditions": torch.stack([example["control_condtion"] for example in examples]),
        }

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    transformer, loader = accelerator.prepare(transformer, loader)
    data_root, output = Path(args.data_root).resolve(), Path(args.output_dir).expanduser().resolve()
    for folder in ("cond_mask", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    index_rows = [json.loads(line) for line in Path(args.json_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    row_by_target = {Path(row["target"]).name: row for row in index_rows}
    metadata = []

    for batch in loader:
        names = batch["img_names"]
        pending = [not (output / "pred_rgb" / f"{Path(name).stem}_pred_rgb.png").is_file() or args.overwrite for name in names]
        if not any(pending):
            continue
        condition_dict = {"cond_types": [], "cond_latents": []}
        with torch.no_grad():
            for kind, images in zip(batch["cond_types"], batch["cond_latents"]):
                condition_dict["cond_types"].append(kind)
                condition_dict["cond_latents"].append(encode_images(vae, images.to(accelerator.device, dtype=dtype), dtype))
            generators = []
            for name in names:
                digest = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
                generators.append(torch.Generator(device=accelerator.device).manual_seed(args.seed + digest))
            kwargs = dict(
                transformer=transformer, vae=vae, scheduler=scheduler, image_processor=processor,
                prompt_embeds=batch["prompt_embeds"].to(accelerator.device),
                pooled_prompt_embeds=batch["pooled_prompt_embeds"].to(accelerator.device),
                num_inference_steps=args.num_inference_steps, condition_dict=condition_dict,
                width=args.resolution, height=args.resolution, generator=generators,
            )
            if args.crfm:
                kwargs.update(conditional_model=control_model, condition_targets=batch["control_conditions"].to(accelerator.device), rectified_step=args.rectified_step, max_step_size=args.max_step_size, ignore_index=255)
            images = inference_fn(**kwargs).images

        for name_file, prediction in zip(names, images):
            row = row_by_target[name_file]
            name = Path(name_file).stem
            label_path, target_path = data_root / row["source"], data_root / row["target"]
            ids = Image.open(label_path).convert("L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
            ids_np = np.asarray(ids, dtype=np.uint8)
            mask_rgb = Image.fromarray(np.asarray([[0,0,0],[255,255,255],[255,0,0],[0,0,255],[255,255,0],[0,255,0],[0,255,255]], dtype=np.uint8)[ids_np.clip(0, 6)], "RGB")
            target = Image.open(target_path).convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
            pred = prediction.convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
            ids.save(output / "cond_mask_ids" / f"{name}_cond_mask_ids.png")
            mask_rgb.save(output / "cond_mask" / f"{name}_cond_mask.png")
            target.save(output / "gt_rgb" / f"{name}_gt_rgb.png")
            pred.save(output / "pred_rgb" / f"{name}_pred_rgb.png")
            Image.fromarray(np.abs(np.asarray(pred, np.int16) - np.asarray(target, np.int16)).astype(np.uint8), "RGB").save(output / "absdiff" / f"{name}_absdiff.png")
            (output / "prompts" / f"{name}.txt").write_text(str(row.get("prompts", row.get("prompt", ""))) + "\n", encoding="utf-8")
            metadata.append({"name": name, "method": "TODSynth+CRFM" if args.crfm else "TODSynth", "condition_passed_to_model": ["semantic_mask", "text_prompt"], "ground_truth_rgb_passed_to_model": False})

    rank = accelerator.process_index
    with (output / f"prompts_rank{rank:02d}.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()
    print(f"[run_todsynth_manifest] rank={rank} wrote {len(metadata)} samples")


if __name__ == "__main__":
    main()
