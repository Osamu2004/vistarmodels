from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def one_of(folder: Path, names: tuple[str, ...]) -> bool:
    return any((folder / name).is_file() for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local SD1.5 and SECOND ControlNet prerequisites.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--controlnet", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--check_rows", type=int, default=8)
    args = parser.parse_args()

    import accelerate
    import diffusers
    import numpy
    import torch
    import transformers
    import PIL

    base = Path(args.base_model).expanduser().resolve()
    errors: list[str] = []
    required_configs = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "unet/config.json",
    )
    if not base.is_dir():
        errors.append(f"base model directory does not exist: {base}")
    else:
        for relative in required_configs:
            if not (base / relative).is_file():
                errors.append(f"missing base-model file: {base / relative}")
        for component in ("text_encoder", "vae", "unet"):
            folder = base / component
            if not one_of(folder, ("model.safetensors", "diffusion_pytorch_model.safetensors", "pytorch_model.bin", "diffusion_pytorch_model.bin")):
                errors.append(f"missing weights under: {folder}")

    controlnet_path = None
    if args.controlnet:
        controlnet_path = Path(args.controlnet).expanduser().resolve()
        if not controlnet_path.is_dir():
            errors.append(f"ControlNet directory does not exist: {controlnet_path}")
        else:
            if not (controlnet_path / "config.json").is_file():
                errors.append(f"missing ControlNet config: {controlnet_path / 'config.json'}")
            if not one_of(controlnet_path, ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")):
                errors.append(f"missing ControlNet weights under: {controlnet_path}")

    manifest_rows = None
    prompt_examples: list[str] = []
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        if not manifest.is_file():
            errors.append(f"manifest does not exist: {manifest}")
        else:
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest_rows = len(rows)
            if not rows:
                errors.append(f"manifest is empty: {manifest}")
            names: set[str] = set()
            for row in rows[: max(1, args.check_rows)]:
                name = str(row.get("name", ""))
                if not name:
                    errors.append("manifest row has no name")
                elif name in names:
                    errors.append(f"duplicate manifest name in checked prefix: {name}")
                names.add(name)
                for key in ("source_image", "target_image"):
                    value = row.get(key)
                    if not value or not Path(str(value)).expanduser().is_file():
                        errors.append(f"{name}: missing {key}: {value}")
                mask_value = row.get("target_mask_ids") or row.get("target_mask_source")
                if not mask_value or not Path(str(mask_value)).expanduser().is_file():
                    errors.append(f"{name}: missing target mask: {mask_value}")
                prompt = row.get("controlnet_prompt")
                if isinstance(prompt, str) and prompt.strip():
                    if len(prompt_examples) < 2:
                        prompt_examples.append(prompt.strip())
                else:
                    class_ids = row.get("changed_class_ids")
                    if not isinstance(class_ids, list):
                        errors.append(
                            f"{name}: needs controlnet_prompt or changed_class_ids for class-aware text"
                        )

    if args.require_cuda and not torch.cuda.is_available():
        errors.append("CUDA is required but torch.cuda.is_available() is false")
    report = {
        "status": "ok" if not errors else "error",
        "python": platform.python_version(),
        "base_model": str(base),
        "controlnet": str(controlnet_path) if controlnet_path else None,
        "manifest_rows": manifest_rows,
        "prompt_examples": prompt_examples,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "versions": {
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "accelerate": accelerate.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "PIL": PIL.__version__,
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
