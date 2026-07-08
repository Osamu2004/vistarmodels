from __future__ import annotations

import importlib
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
    Dependency("torch", "torch", True, "CUDA PyTorch runtime"),
    Dependency("torchvision", "torchvision", True, "Seg2Any mask transform pipeline"),
    Dependency("diffusers", "diffusers", True, "FLUX pipeline components", "0.32.2"),
    Dependency("transformers", "transformers", True, "CLIP/T5 tokenizers and encoders", "4.55.2"),
    Dependency("accelerate", "accelerate", True, "official Seg2Any environment dependency", "1.3.0"),
    Dependency("peft", "peft", True, "LoRA loading support", "0.14.0"),
    Dependency("safetensors", "safetensors", True, "weight loading"),
    Dependency("sentencepiece", "sentencepiece", True, "T5 tokenizer backend"),
    Dependency("protobuf", "google.protobuf", True, "tokenizer/model config support"),
    Dependency("huggingface-hub", "huggingface_hub", True, "model download/cache access"),
    Dependency("numpy", "numpy", True, "image preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("opencv-python", "cv2", True, "official Seg2Any utilities import cv2"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
    Dependency("sam2", "sam2", False, "only needed for Seg2Any official evaluation metrics, not LoveDA generation inference"),
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
            print(f"{status:17s} {dep.import_name:20s} pip={dep.pip_name:20s} reason={exc}")
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
            print(f"TOO OLD required {dep.import_name:20s} pip={dep.pip_name:20s}{version_note}")
            too_old_required.append((dep, version))
        else:
            print(f"ok               {dep.import_name:20s} pip={dep.pip_name:20s}{version_note}")

    seg2any_root = Path("third_party/Seg2Any")
    if seg2any_root.is_dir():
        print(f"ok               seg2any source       path={seg2any_root.resolve()}")
    else:
        print(f"MISSING required seg2any source       path={seg2any_root.resolve()}")
        print("Run: bash scripts/bootstrap_seg2any.sh")
        missing_required.append(Dependency("Seg2Any", "third_party/Seg2Any", True, "official Seg2Any source checkout"))

    if missing_required or too_old_required:
        print("\nMissing or incompatible required packages.")
        pip_missing = [
            dep
            for dep in missing_required
            if dep.pip_name not in {"torch", "torchvision", "Seg2Any"}
        ]
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if any(dep.pip_name == "torch" for dep in missing_required):
            print("Install torch/torchvision from the CUDA-matched PyTorch channel, or clone an existing CUDA env.")
        if too_old_required:
            install = " ".join(f"{dep.pip_name}>={dep.min_version}" for dep, _ in too_old_required)
            print(f"pip install -U {install}")
        print("For the pinned Seg2Any environment, use: pip install -r requirements-seg2any.txt")

    if missing_optional:
        print("\nOptional packages missing.")
        print("sam2 is only needed for Seg2Any's official SAM-based evaluation, not this LoveDA generation wrapper.")

    return 1 if missing_required or too_old_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
