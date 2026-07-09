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
    min_version: str = ""


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("torch", "torch", True, "CUDA-matched PyTorch runtime"),
    Dependency("numpy", "numpy", True, "mask preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("einops", "einops", True, "PLACE tensor rearranges"),
    Dependency("omegaconf", "omegaconf", True, "PLACE config loading"),
    Dependency("pytorch-lightning", "pytorch_lightning", True, "latent-diffusion compatibility"),
    Dependency("transformers", "transformers", True, "CLIP text encoder/tokenizer"),
    Dependency("scipy", "scipy", True, "PLACE loads color150.mat"),
    Dependency("open-clip-torch", "open_clip", False, "official dataset.py imports open_clip"),
    Dependency("opencv-python", "cv2", False, "official scripts import cv2"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
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


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in text.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_too_old(found: str, minimum: str) -> bool:
    if not minimum:
        return False
    found_tuple = _version_tuple(found)
    min_tuple = _version_tuple(minimum)
    if not found_tuple or not min_tuple:
        return False
    pad = max(len(found_tuple), len(min_tuple))
    return found_tuple + (0,) * (pad - len(found_tuple)) < min_tuple + (0,) * (pad - len(min_tuple))


def _place_root() -> Path:
    default = Path("third_party/PLACE")
    return Path(_normalize_wsl_unc(os.environ.get("PLACE_ROOT", str(default)))).expanduser()


def main() -> int:
    missing_required: list[Dependency] = []
    too_old_required: list[tuple[Dependency, str]] = []
    missing_optional: list[Dependency] = []

    place_root = _place_root()
    if place_root.is_dir():
        print(f"ok               PLACE source       path={place_root.resolve()}")
        sys.path.insert(0, str(place_root.resolve()))
    else:
        print(f"MISSING required PLACE source       path={place_root.resolve()}")
        print("Run: bash scripts/bootstrap_place.sh")

    config_path = Path(
        _normalize_wsl_unc(os.environ.get("PLACE_CONFIG", str(place_root / "configs/stable-diffusion/PLACE.yaml")))
    ).expanduser()
    if config_path.is_file():
        print(f"ok               PLACE config       path={config_path.resolve()}")
    else:
        print(f"MISSING required PLACE config       path={config_path.resolve()}")

    ckpt_path = Path(_normalize_wsl_unc(os.environ.get("PLACE_CKPT", "/root/data/weight/place/coco_best.ckpt"))).expanduser()
    if ckpt_path.is_file():
        print(f"ok               PLACE checkpoint   path={ckpt_path.resolve()}")
    else:
        print(f"missing optional PLACE checkpoint   path={ckpt_path.resolve()}")
        print("Download coco_best.ckpt or ade20k_best.ckpt from the official Google Drive linked in baselines/place/README.md.")

    for dep in DEPENDENCIES:
        try:
            module = importlib.import_module(dep.import_name)
        except Exception as exc:
            status = "MISSING required" if dep.required else "missing optional"
            print(f"{status:17s} {dep.import_name:18s} pip={dep.pip_name:18s} reason={exc}")
            if dep.required:
                missing_required.append(dep)
            else:
                missing_optional.append(dep)
            continue

        version = _version(dep, module)
        version_note = f" version={version}"
        if dep.min_version:
            version_note += f" min={dep.min_version}"
        if dep.required and _version_too_old(version, dep.min_version):
            print(f"TOO OLD required {dep.import_name:18s} pip={dep.pip_name:18s}{version_note}")
            too_old_required.append((dep, version))
        else:
            print(f"ok               {dep.import_name:18s} pip={dep.pip_name:18s}{version_note}")

    try:
        from ldm.models.diffusion.plms import PLMSSampler
        from ldm.util import instantiate_from_config

        _ = PLMSSampler
        _ = instantiate_from_config
        print("ok               PLACE ldm modules are importable")
    except Exception as exc:
        print(f"MISSING required PLACE ldm modules reason={exc}")
        missing_required.append(Dependency("PLACE", "ldm", True, "official PLACE source checkout"))

    if missing_required or too_old_required:
        print("\nMissing or incompatible required packages.")
        pip_missing = [
            dep
            for dep in missing_required
            if dep.pip_name not in {"torch", "PLACE"}
        ]
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if any(dep.pip_name == "torch" for dep in missing_required):
            print("Install torch from the CUDA-matched PyTorch channel, or clone an existing CUDA env.")
        if any(dep.pip_name == "PLACE" for dep in missing_required):
            print("Run: bash scripts/bootstrap_place.sh")
        if too_old_required:
            install = " ".join(f"{dep.pip_name}>={dep.min_version}" for dep, _ in too_old_required)
            print(f"pip install -U {install}")

    if missing_optional:
        print("\nOptional packages missing.")
        print("These are used by the official PLACE scripts/datasets, but the Vistar wrapper may still run without them.")

    return 1 if missing_required or too_old_required or not place_root.is_dir() or not config_path.is_file() else 0


if __name__ == "__main__":
    raise SystemExit(main())
