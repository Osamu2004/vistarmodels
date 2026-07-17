from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path


DEPENDENCIES = {
    "torch": "torch",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors",
    "peft": "peft",
    "pillow": "PIL",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "termcolor": "termcolor",
}
SOURCE_FILES = (
    "anysd/src/model.py",
    "anysd/src/pipe.py",
    "anysd/src/unet.py",
    "anysd/src/utils.py",
    "anysd/src/adapter.py",
    "anysd/ip_adapter/attention_processor.py",
)
ANYSD_FILES = (
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "experts/task_embs.bin",
    "experts/visual_seg.bin",
)
BASE_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "vae/config.json",
)
BASE_WEIGHT_GROUPS = (
    ("text_encoder/model.safetensors", "text_encoder/pytorch_model.bin"),
    ("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.bin"),
)


def _check_files(root: Path, required: tuple[str, ...], label: str, failures: list[str]) -> None:
    missing = [item for item in required if not (root / item).is_file() or (root / item).stat().st_size == 0]
    if missing:
        failures.append(f"incomplete {label}")
        print(f"MISSING required {label}={root}")
        for item in missing:
            print(f"  missing/empty: {item}")
    else:
        size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        print(f"ok               {label}={root} size={size / (1024 ** 3):.2f} GiB")


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

    root = Path(os.environ.get("ANYSD_ROOT", "third_party/AnySD")).expanduser().resolve()
    _check_files(root, SOURCE_FILES, "official AnySD source", failures)
    model_dir = Path(os.environ.get("ANYSD_MODEL_DIR", "/root/data/weight/anysd/AnySD")).expanduser().resolve()
    _check_files(model_dir, ANYSD_FILES, "AnySD snapshot", failures)
    base_dir = Path(os.environ.get(
        "ANYSD_BASE_MODEL_DIR", "/root/data/weight/anysd/stable-diffusion-v1-5"
    )).expanduser().resolve()
    _check_files(base_dir, BASE_FILES, "SD1.5 base snapshot", failures)
    for alternatives in BASE_WEIGHT_GROUPS:
        if not any((base_dir / item).is_file() and (base_dir / item).stat().st_size > 0 for item in alternatives):
            failures.append("incomplete SD1.5 base weights")
            print("MISSING required SD1.5 weight: " + " or ".join(alternatives))

    if root.is_dir() and "diffusers" in modules and "transformers" in modules:
        sys.path.insert(0, str(root))
        try:
            from anysd.src.pipe import AnySDInstructPix2PixPipeline
            from anysd.src.unet import UNet2DConditionAnySD

            _ = AnySDInstructPix2PixPipeline, UNet2DConditionAnySD
            print("ok               official AnySD custom pipeline imports")
        except Exception as exc:
            failures.append("official AnySD pipeline import failed")
            print(f"INCOMPATIBLE     official AnySD pipeline reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("ANYSD_REQUIRE_CUDA", "1").lower() not in {"0", "false", "no"}
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
        print("\nAnySD dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print(f"\nRepair with:\n  {sys.executable} -m pip install -r requirements-anysd.txt")
        print("  bash scripts/bootstrap_anysd.sh")
        return 1
    print("\nAnySD dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
