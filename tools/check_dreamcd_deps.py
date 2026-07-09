from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    required: bool
    note: str


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("torch", "torch", True, "CUDA-matched PyTorch runtime"),
    Dependency("torchvision", "torchvision", True, "DreamCD preview/grid utilities"),
    Dependency("numpy", "numpy", True, "image/tensor preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("huggingface-hub", "huggingface_hub", True, "automatic DreamCD checkpoint download"),
    Dependency("opencv-python", "cv2", True, "Albumentations interpolation constants"),
    Dependency("albumentations", "albumentations", True, "official DreamCD dataset resizing/cropping"),
    Dependency("omegaconf", "omegaconf", True, "DreamCD YAML config loading"),
    Dependency("pytorch-lightning", "pytorch_lightning", True, "LatentDiffusion LightningModule base class"),
    Dependency("einops", "einops", True, "tensor layout transforms"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
    Dependency("termcolor", "termcolor", True, "official logger formatting"),
    Dependency("kornia", "kornia", True, "latent diffusion imports"),
    Dependency("torchmetrics", "torchmetrics", True, "pytorch-lightning compatibility in official env"),
    Dependency("test-tube", "test_tube", True, "official latent-diffusion utilities"),
    Dependency("taming-transformers", "taming", True, "VQ/VQGAN modules imported by DreamCD"),
    Dependency("transformers", "transformers", False, "official requirement; not needed by the wrapper path in normal inference"),
    Dependency("clip", "clip", False, "official requirement; not needed by the wrapper path in normal inference"),
    Dependency("imageio", "imageio", False, "official requirement"),
    Dependency("streamlit", "streamlit", False, "only needed for old official demos"),
    Dependency("pudb", "pudb", False, "debugger only"),
)


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _version(dep: Dependency, module: object) -> str:
    try:
        return metadata.version(dep.pip_name)
    except Exception:
        return str(getattr(module, "__version__", "unknown"))


def _dreamcd_root() -> Path:
    default_root = Path("third_party/DreamCD")
    return Path(_normalize_wsl_unc(os.environ.get("DREAMCD_ROOT", str(default_root)))).expanduser().resolve()


def _install_lightning_compat() -> None:
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except Exception:
        return
    module_name = "pytorch_lightning.utilities.distributed"
    if module_name in sys.modules:
        return
    import types

    compat_module = types.ModuleType(module_name)
    compat_module.rank_zero_only = rank_zero_only
    sys.modules[module_name] = compat_module


def main() -> int:
    missing_required: list[Dependency] = []
    missing_optional: list[Dependency] = []

    root = _dreamcd_root()
    if (root / "changeanywhere2_synthesis.py").is_file():
        sys.path.insert(0, str(root))
        print(f"ok               DreamCD source      path={root}")
    else:
        print(f"MISSING required DreamCD source      path={root}")
        print("Run: bash scripts/bootstrap_dreamcd.sh")
        missing_required.append(Dependency("DreamCD", "DreamCD source", True, "official source clone"))

    for dep in DEPENDENCIES:
        try:
            module = importlib.import_module(dep.import_name)
        except Exception as exc:
            status = "MISSING required" if dep.required else "missing optional"
            print(f"{status:17s} {dep.import_name:22s} pip={dep.pip_name:24s} reason={exc}")
            if dep.required:
                missing_required.append(dep)
            else:
                missing_optional.append(dep)
            continue
        print(f"ok               {dep.import_name:22s} pip={dep.pip_name:24s} version={_version(dep, module)}")

    _install_lightning_compat()
    if (root / "changeanywhere2_synthesis.py").is_file():
        try:
            import ldm.data.changeanywhere2  # noqa: F401
            import scripts.sample_diffusion  # noqa: F401

            print("ok               DreamCD modules     official dataset/model imports work")
        except Exception as exc:
            print(f"MISSING required DreamCD modules     reason={exc}")
            missing_required.append(Dependency("DreamCD modules", "ldm/scripts", True, "official module imports"))

    ldm_ckpt = Path(
        _normalize_wsl_unc(
            os.environ.get("DREAMCD_CKPT", str(root / "checkpoints/second/ldm.ckpt"))
        )
    ).expanduser()
    vqvae_ckpt = Path(
        _normalize_wsl_unc(
            os.environ.get("DREAMCD_VQVAE_CKPT", str(root / "checkpoints/second/vqvae.ckpt"))
        )
    ).expanduser()
    for label, path in (("LDM checkpoint", ldm_ckpt), ("VQ-VAE checkpoint", vqvae_ckpt)):
        if path.is_file():
            print(f"ok               {label:22s} path={path.resolve()}")
        else:
            print(f"missing weight   {label:22s} path={path.resolve()}")

    try:
        import torch

        print(f"info             torch cuda available={torch.cuda.is_available()}")
    except Exception:
        pass

    if missing_required:
        print("\nMissing required packages or source.")
        pip_missing = [
            dep
            for dep in missing_required
            if dep.pip_name not in {"torch", "torchvision", "DreamCD", "DreamCD modules"}
        ]
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if any(dep.pip_name in {"torch", "torchvision"} for dep in missing_required):
            print("Install torch/torchvision from your CUDA-matched PyTorch channel.")
        if any(dep.pip_name == "DreamCD" for dep in missing_required):
            print("Run: bash scripts/bootstrap_dreamcd.sh")
        print("DreamCD's official requirement.txt pins an old CUDA 11.1 stack; prefer a separate conda env.")

    if missing_optional:
        print("\nOptional packages missing.")
        install = " ".join(dep.pip_name for dep in missing_optional)
        print(f"pip install {install}")

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
