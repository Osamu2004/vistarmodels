from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    required: bool
    note: str
    min_version: str = ""


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("torch", "torch", True, "CUDA-matched PyTorch runtime"),
    Dependency("diffusers", "diffusers", True, "StableDiffusionControlNetPipeline and ControlNetModel", "0.30.3"),
    Dependency("accelerate", "accelerate", True, "Diffusers device/offload utilities"),
    Dependency("transformers", "transformers", True, "CLIP text encoder/tokenizer used by Stable Diffusion"),
    Dependency("safetensors", "safetensors", True, "HuggingFace model weight loading"),
    Dependency("huggingface-hub", "huggingface_hub", True, "model download/cache access"),
    Dependency("numpy", "numpy", True, "image/tensor preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
    Dependency("xformers", "xformers", False, "optional memory-efficient attention"),
)


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


def main() -> int:
    missing_required: list[Dependency] = []
    too_old_required: list[tuple[Dependency, str]] = []
    missing_optional: list[Dependency] = []

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
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        _ = ControlNetModel
        _ = StableDiffusionControlNetPipeline
        print("ok               diffusers pipeline classes are importable")
    except Exception as exc:
        print(f"MISSING required diffusers pipeline classes reason={exc}")
        if not any(dep.import_name == "diffusers" for dep in missing_required):
            missing_required.append(
                Dependency(
                    "diffusers",
                    "diffusers",
                    True,
                    "StableDiffusionControlNetPipeline and ControlNetModel",
                    "0.30.3",
                )
            )

    if missing_required or too_old_required:
        print("\nMissing or incompatible required packages.")
        pip_missing = [
            dep
            for dep in missing_required
            if dep.pip_name != "torch"
        ]
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if any(dep.pip_name == "torch" for dep in missing_required):
            print("Install torch from the CUDA-matched PyTorch channel, or clone an existing CUDA env.")
        if too_old_required:
            install = " ".join(f"{dep.pip_name}>={dep.min_version}" for dep, _ in too_old_required)
            print(f"pip install -U {install}")

    if missing_optional:
        print("\nOptional packages missing.")
        if any(dep.pip_name == "xformers" for dep in missing_optional):
            print("Install xformers only if a build matches your torch/CUDA version.")

    return 1 if missing_required or too_old_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
