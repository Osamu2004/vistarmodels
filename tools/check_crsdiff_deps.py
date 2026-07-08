from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    required: bool
    note: str


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("torch", "torch", True, "provided by the base CUDA environment"),
    Dependency("torchvision", "torchvision", True, "used by Stable Diffusion modules"),
    Dependency("numpy", "numpy", True, "image/tensor preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("opencv-python", "cv2", True, "condition resizing"),
    Dependency("einops", "einops", True, "tensor layout transforms"),
    Dependency("omegaconf", "omegaconf", True, "CRS-Diff YAML config loading"),
    Dependency("pytorch-lightning", "pytorch_lightning", True, "CRS-Diff LightningModule base classes"),
    Dependency("open-clip-torch", "open_clip", True, "CRS-Diff text encoder module import"),
    Dependency("transformers", "transformers", True, "CLIP/T5 tokenizer and model imports"),
    Dependency("timm", "timm", True, "official CRS-Diff annotator/model imports"),
    Dependency("scipy", "scipy", True, "latent diffusion utility imports"),
    Dependency("safetensors", "safetensors", True, "optional safetensors checkpoint loader"),
    Dependency("pyyaml", "yaml", True, "YAML support"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
    Dependency("albumentations", "albumentations", False, "official annotator/degradation utilities"),
    Dependency("basicsr", "basicsr", False, "official annotator checkpoint download utilities"),
    Dependency("ttach", "ttach", False, "official segmentation annotator"),
    Dependency("xformers", "xformers", False, "optional memory-efficient attention; CRS-Diff falls back without it"),
    Dependency("gradio", "gradio", False, "only needed for official CRS-Diff web UI"),
    Dependency("datasets", "datasets", False, "only needed for official sample inference script"),
)


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    missing_required: list[Dependency] = []
    missing_optional: list[Dependency] = []

    for dep in DEPENDENCIES:
        try:
            module = importlib.import_module(dep.import_name)
        except Exception as exc:
            status = "MISSING required" if dep.required else "missing optional"
            print(f"{status:17s} {dep.import_name:22s} pip={dep.pip_name:22s} reason={exc}")
            if dep.required:
                missing_required.append(dep)
            else:
                missing_optional.append(dep)
            continue
        print(f"ok               {dep.import_name:22s} pip={dep.pip_name:22s} version={_version(module)}")

    if missing_required:
        torch_missing = [dep for dep in missing_required if dep.pip_name in {"torch", "torchvision"}]
        pip_missing = [dep for dep in missing_required if dep.pip_name not in {"torch", "torchvision"}]
        print("\nMissing required packages. Install with:")
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if torch_missing:
            print("Install torch/torchvision from your CUDA-matched PyTorch channel, or clone an existing CUDA env.")
    if missing_optional:
        regular_optional = [
            dep
            for dep in missing_optional
            if dep.pip_name not in {"xformers", "basicsr"}
        ]
        print("\nOptional packages missing. For official CRS-Diff annotators/UI, install:")
        if regular_optional:
            install = " ".join(dep.pip_name for dep in regular_optional)
            print(f"pip install {install}")
        if any(dep.pip_name == "basicsr" for dep in missing_optional):
            print("pip install --no-build-isolation basicsr")
        print("Install xformers separately only if it matches your torch/CUDA version.")

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
