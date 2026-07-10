from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    expected: str = ""
    minimum: str = ""


DEPENDENCIES = (
    Dependency("torch", "torch", minimum="2.2.0"),
    Dependency("diffusers", "diffusers", expected="0.31.0"),
    Dependency("transformers", "transformers", expected="4.39.1"),
    Dependency("accelerate", "accelerate", expected="0.34.2"),
    Dependency("huggingface-hub", "huggingface_hub", minimum="0.25.1"),
    Dependency("safetensors", "safetensors", minimum="0.4.5"),
    Dependency("numpy", "numpy"),
    Dependency("pillow", "PIL"),
    Dependency("tqdm", "tqdm", minimum="4.66.0"),
)


MODEL_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet_ema/config.json",
    "unet_ema/diffusion_pytorch_model.safetensors",
)


def _version_tuple(value: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in value.split("+")[0].replace("-", ".").split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _at_least(found: str, required: str) -> bool:
    left, right = _version_tuple(found), _version_tuple(required)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    failures: list[str] = []
    modules: dict[str, object] = {}

    for dep in DEPENDENCIES:
        try:
            module = importlib.import_module(dep.import_name)
            modules[dep.import_name] = module
            try:
                version = metadata.version(dep.pip_name)
            except Exception:
                version = str(getattr(module, "__version__", "unknown"))
            compatible = True
            requirement = ""
            if dep.expected and version != dep.expected:
                compatible = False
                requirement = f"expected={dep.expected}"
            elif dep.minimum and not _at_least(version, dep.minimum):
                compatible = False
                requirement = f"minimum={dep.minimum}"
            if compatible:
                print(f"ok               {dep.import_name:18s} version={version}")
            else:
                failures.append(f"incompatible {dep.pip_name}")
                print(f"INCOMPATIBLE     {dep.import_name:18s} version={version} {requirement}")
        except Exception as exc:
            failures.append(f"missing {dep.pip_name}")
            print(f"MISSING required {dep.import_name:18s} pip={dep.pip_name} reason={exc}")

    repo_root = Path(__file__).resolve().parents[1]
    source_root = Path(os.environ.get("RCDGEN_ROOT", repo_root / "third_party/referring_change_detection")).expanduser()
    source_pipeline = source_root / "RCDGen/RCDGenSDPipeline.py"
    if source_pipeline.is_file() and source_pipeline.stat().st_size > 0:
        print(f"ok               official source pipeline={source_pipeline}")
    else:
        failures.append("missing official RCDGen source")
        print(f"MISSING required official source pipeline={source_pipeline}")

    installed_pipeline: Path | None = None
    try:
        pipeline_module = importlib.import_module("diffusers.pipelines.stable_diffusion.RCDGenSDPipeline")
        pipeline_class = getattr(pipeline_module, "StableDiffusionInstructPix2PixPipeline")
        _ = pipeline_class
        installed_pipeline = Path(str(pipeline_module.__file__)).resolve()
        print(f"ok               custom pipeline import={installed_pipeline}")
    except Exception as exc:
        failures.append("custom RCDGen pipeline is not importable")
        print(f"MISSING required RCDGenSDPipeline reason={exc}")

    if source_pipeline.is_file() and installed_pipeline and installed_pipeline.is_file():
        if _sha256(source_pipeline) == _sha256(installed_pipeline):
            print("ok               installed pipeline matches official source")
        else:
            failures.append("installed pipeline differs from official source")
            print("INCOMPATIBLE     installed pipeline differs from official RCDGen source")

    model_dir = Path(os.environ.get("RCDGEN_MODEL_DIR", "/root/data/weight/rcdgen/RCDGen")).expanduser()
    missing_files = [name for name in MODEL_FILES if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0]
    if missing_files:
        failures.append("incomplete Hugging Face model snapshot")
        print(f"MISSING required model snapshot={model_dir}")
        for name in missing_files:
            print(f"  missing/empty: {name}")
    else:
        try:
            model_index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
            class_name = model_index.get("_class_name")
            if class_name != "StableDiffusionInstructPix2PixPipeline":
                failures.append("unexpected model pipeline class")
                print(f"INCOMPATIBLE     model _class_name={class_name!r}")
            else:
                total_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
                print(f"ok               HF snapshot={model_dir} size={total_bytes / (1024 ** 3):.2f} GiB")
                print("ok               EMA UNet weights present")
        except Exception as exc:
            failures.append("invalid model_index.json")
            print(f"INCOMPATIBLE     model_index.json reason={exc}")

    torch = modules.get("torch")
    if torch is not None:
        require_cuda = _truthy("RCDGEN_REQUIRE_CUDA")
        try:
            available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
            if require_cuda and not available:
                failures.append("CUDA unavailable")
                print("MISSING required CUDA; use RCDGEN_REQUIRE_CUDA=0 only for a static check")
            elif available:
                tensor = torch.ones((16, 16), device="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
                result = (tensor @ tensor).sum().item()
                if result <= 0:
                    raise RuntimeError("unexpected CUDA tensor result")
                print(f"ok               CUDA allocation device={torch.cuda.get_device_name(0)}")  # type: ignore[attr-defined]
            else:
                print("skipped          CUDA runtime check (RCDGEN_REQUIRE_CUDA=0)")
        except Exception as exc:
            failures.append("CUDA tensor allocation failed")
            print(f"MISSING required working CUDA reason={exc}")

    if failures:
        print("\nRCDGen dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nRepair with:")
        print(f"  {sys.executable} -m pip install -r requirements-rcdgen.txt")
        print("  bash scripts/bootstrap_rcdgen.sh")
        return 1

    print("\nRCDGen dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
