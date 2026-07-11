from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path


def version_tuple(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split("+")[0].replace("-", ".").split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        result.append(int(digits))
    return tuple(result)


def at_least(found: str, required: str) -> bool:
    left, right = version_tuple(found), version_tuple(required)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def has_weight(directory: Path) -> bool:
    return any(
        path.is_file() and path.stat().st_size > 0 and path.suffix in {".safetensors", ".bin"}
        for path in directory.glob("*")
    )


def main() -> int:
    failures: list[str] = []
    packages = (
        ("torch", "torch", "2.2.0"),
        ("diffusers", "diffusers", "0.32.0"),
        ("transformers", "transformers", "4.44.0"),
        ("accelerate", "accelerate", "0.33.0"),
        ("huggingface-hub", "huggingface_hub", "0.24.0"),
        ("safetensors", "safetensors", "0.4.5"),
        ("numpy", "numpy", ""),
        ("pillow", "PIL", ""),
        ("tqdm", "tqdm", "4.66.0"),
    )
    modules: dict[str, object] = {}
    for pip_name, import_name, minimum in packages:
        try:
            module = importlib.import_module(import_name)
            modules[import_name] = module
            try:
                found = metadata.version(pip_name)
            except Exception:
                found = str(getattr(module, "__version__", "unknown"))
            if minimum and not at_least(found, minimum):
                failures.append(f"{pip_name}>={minimum}")
                print(f"INCOMPATIBLE {import_name:18s} version={found} minimum={minimum}")
            else:
                print(f"ok           {import_name:18s} version={found}")
        except Exception as exc:
            failures.append(f"missing {pip_name}")
            print(f"MISSING      {import_name:18s} reason={exc}")

    try:
        from diffusers import FluxFillPipeline

        _ = FluxFillPipeline
        print("ok           diffusers.FluxFillPipeline")
    except Exception as exc:
        failures.append("FluxFillPipeline unavailable")
        print(f"MISSING      diffusers.FluxFillPipeline reason={exc}")

    model_dir = Path(os.environ.get("FLUX1_FILL_MODEL_DIR", "/root/data/weight/flux1_fill/FLUX.1-Fill-dev")).expanduser()
    required_configs = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "transformer/config.json",
        "vae/config.json",
    )
    missing = [name for name in required_configs if not (model_dir / name).is_file()]
    missing_weights = [component for component in ("text_encoder", "text_encoder_2", "transformer", "vae") if not has_weight(model_dir / component)]
    if missing or missing_weights:
        failures.append("incomplete FLUX.1 Fill-dev snapshot")
        print(f"MISSING      model snapshot={model_dir}")
        for name in missing:
            print(f"  missing config: {name}")
        for component in missing_weights:
            print(f"  missing weights: {component}/*.safetensors or *.bin")
    else:
        total_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
        print(f"ok           FLUX.1 Fill-dev snapshot={model_dir} size={total_bytes / (1024 ** 3):.2f} GiB")

    torch = modules.get("torch")
    if torch is not None:
        require_cuda = truthy(os.environ.get("FLUX1_FILL_REQUIRE_CUDA", "1"))
        try:
            available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
            if require_cuda and not available:
                failures.append("CUDA unavailable")
                print("MISSING      CUDA; set FLUX1_FILL_REQUIRE_CUDA=0 only for static checks")
            elif available:
                tensor = torch.ones((8, 8), dtype=torch.bfloat16, device="cuda")  # type: ignore[attr-defined]
                if tensor.sum().item() <= 0:
                    raise RuntimeError("unexpected CUDA tensor result")
                print(f"ok           CUDA device={torch.cuda.get_device_name(0)}")  # type: ignore[attr-defined]
            else:
                print("skipped      CUDA runtime check")
        except Exception as exc:
            failures.append("CUDA check failed")
            print(f"MISSING      CUDA reason={exc}")

    if failures:
        print("\nFLUX.1 Fill-dev dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nRepair with:")
        print(f"  {sys.executable} -m pip install -r requirements-flux1-fill.txt")
        print("  accept the FLUX.1 Fill-dev license on Hugging Face, then run bash scripts/bootstrap_flux1_fill.sh")
        return 1
    print("\nFLUX.1 Fill-dev dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
