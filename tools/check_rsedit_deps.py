from __future__ import annotations

import importlib
import os
from pathlib import Path


def main() -> int:
    failures = []
    for module in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors", "PIL", "numpy", "tqdm"):
        try:
            imported = importlib.import_module(module)
            print(f"ok               {module} {getattr(imported, '__version__', '')}")
        except Exception as exc:
            failures.append(module)
            print(f"MISSING required {module}: {exc}")
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        _ = StableDiffusionInstructPix2PixPipeline
        print("ok               stock InstructPix2Pix pipeline")
    except Exception as exc:
        failures.append("pipeline")
        print(f"MISSING required pipeline: {exc}")
    model = Path(os.environ.get(
        "RSEDIT_MODEL_DIR",
        "/root/data/weight/rsedit/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14",
    ))
    for required in ("model_index.json", "unet/config.json", "vae/config.json"):
        if not (model / required).is_file():
            failures.append(required)
            print(f"MISSING required {model / required}")
    print("\nRSEdit dependency check " + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
