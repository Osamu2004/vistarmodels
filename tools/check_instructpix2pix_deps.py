from __future__ import annotations

import importlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path


DEPENDENCIES = {
    "torch": "torch", "diffusers": "diffusers", "transformers": "transformers",
    "accelerate": "accelerate", "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors", "pillow": "PIL", "numpy": "numpy", "tqdm": "tqdm",
}
REQUIRED_PATHS = (
    "model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
    "text_encoder/config.json", "vae/config.json", "unet/config.json",
)
WEIGHT_GROUPS = (
    ("text_encoder/model.safetensors", "text_encoder/pytorch_model.bin"),
    ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.bin"),
    ("unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.bin"),
)


def main() -> int:
    failures: list[str] = []
    modules: dict[str, object] = {}
    for pip_name, import_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(import_name)
            modules[import_name] = module
            try:
                version = metadata.version(pip_name)
            except Exception:
                version = str(getattr(module, "__version__", "unknown"))
            print(f"ok               {import_name:18s} version={version}")
        except Exception as exc:
            failures.append(f"missing {pip_name}")
            print(f"MISSING required {import_name:18s} pip={pip_name} reason={exc}")

    try:
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        _ = EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        print("ok               stock Diffusers InstructPix2Pix pipeline")
    except Exception as exc:
        failures.append("InstructPix2Pix pipeline unavailable")
        print(f"MISSING required StableDiffusionInstructPix2PixPipeline reason={exc}")

    model_dir = Path(os.environ.get(
        "INSTRUCTPIX2PIX_MODEL_DIR", "/root/data/weight/instructpix2pix/instruct-pix2pix"
    )).expanduser()
    missing = [path for path in REQUIRED_PATHS if not (model_dir / path).is_file()]
    for alternatives in WEIGHT_GROUPS:
        if not any((model_dir / path).is_file() and (model_dir / path).stat().st_size > 0 for path in alternatives):
            missing.append(" or ".join(alternatives))
    if missing:
        failures.append("incomplete model snapshot")
        print(f"MISSING required model snapshot={model_dir}")
        for path in missing:
            print(f"  missing/empty: {path}")
    else:
        try:
            index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
            if index.get("_class_name") != "StableDiffusionInstructPix2PixPipeline":
                raise ValueError(f"unexpected _class_name={index.get('_class_name')!r}")
            size = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
            print(f"ok               model snapshot={model_dir} size={size / (1024 ** 3):.2f} GiB")
        except Exception as exc:
            failures.append("invalid model index")
            print(f"INCOMPATIBLE     model_index.json reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("INSTRUCTPIX2PIX_REQUIRE_CUDA", "1").lower() not in {"0", "false", "no"}
    if torch is not None:
        try:
            if require_cuda and not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("CUDA is unavailable")
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                tensor = torch.ones((16, 16), device="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
                _ = (tensor @ tensor).sum().item()
                print(f"ok               CUDA device={torch.cuda.get_device_name(0)}")  # type: ignore[attr-defined]
            else:
                print("skipped          CUDA runtime check")
        except Exception as exc:
            failures.append("CUDA check failed")
            print(f"MISSING required working CUDA reason={exc}")

    if failures:
        print("\nInstructPix2Pix dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print(f"\nRepair with:\n  {sys.executable} -m pip install -r requirements-instructpix2pix.txt")
        print("  bash scripts/bootstrap_instructpix2pix.sh")
        return 1
    print("\nInstructPix2Pix dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
