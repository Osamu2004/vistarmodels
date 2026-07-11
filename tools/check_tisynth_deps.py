from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    required: bool = True


DEPENDENCIES = (
    Dependency("torch", "torch"),
    Dependency("torchvision", "torchvision"),
    Dependency("numpy", "numpy"),
    Dependency("pillow", "PIL"),
    Dependency("einops", "einops"),
    Dependency("omegaconf", "omegaconf"),
    Dependency("pytorch-lightning", "pytorch_lightning"),
    Dependency("open-clip-torch", "open_clip"),
    Dependency("transformers", "transformers"),
    Dependency("scipy", "scipy"),
    Dependency("safetensors", "safetensors"),
    Dependency("tqdm", "tqdm"),
    Dependency("xformers", "xformers", False),
)


def main() -> int:
    missing: list[Dependency] = []
    for dependency in DEPENDENCIES:
        try:
            module = importlib.import_module(dependency.import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"ok               {dependency.import_name:22s} version={version}")
        except Exception as exc:
            status = "MISSING required" if dependency.required else "missing optional"
            print(f"{status:17s} {dependency.import_name:22s} reason={exc}")
            if dependency.required:
                missing.append(dependency)
    if missing:
        packages = " ".join(dependency.pip_name for dependency in missing)
        print(f"\nMissing required dependencies. Install the official tisynth.yml or: pip install {packages}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
