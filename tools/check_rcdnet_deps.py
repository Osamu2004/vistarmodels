from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "timm": "timm",
    "numpy": "numpy",
    "Pillow": "PIL",
    "einops": "einops",
    "easydict": "easydict",
    "tqdm": "tqdm",
}


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _required_file(label: str, path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing {label}")
        print(f"MISSING required {label}={path}")
    else:
        print(f"ok               {label}={path} size={path.stat().st_size}")


def main() -> int:
    failures: list[str] = []
    modules: dict[str, object] = {}
    for pip_name, import_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(import_name)
            modules[import_name] = module
            print(
                f"ok               {import_name:18s} "
                f"version={getattr(module, '__version__', 'unknown')}"
            )
        except Exception as exc:
            failures.append(f"missing {pip_name}")
            print(f"MISSING required {import_name:18s} reason={exc}")

    root = _path("RCDNET_ROOT", "third_party/referring_change_detection")
    source = root / "RCDNet"
    for path in (
        source / "models" / "builder.py",
        source / "models" / "encoders" / "dual_vmamba.py",
        source / "models" / "decoders" / "MambaDecoder.py",
        source / "configs" / "config_second.py",
    ):
        _required_file("official RCDNet source", path, failures)

    checkpoint = _path(
        "RCDNET_CHECKPOINT",
        "/root/data/weight/rcdnet/SECOND-model.safetensors",
    )
    _required_file("official SECOND real-data checkpoint", checkpoint, failures)
    if checkpoint.is_file() and "safetensors" in modules:
        try:
            from safetensors import safe_open

            with safe_open(checkpoint, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
            if "decode_head.output.weight" not in keys:
                raise RuntimeError("missing decode_head.output.weight")
            print(f"ok               RCDNet checkpoint tensors={len(keys)}")
        except Exception as exc:
            failures.append("invalid RCDNet checkpoint")
            print(f"INCOMPATIBLE     RCDNet checkpoint reason={exc}")

    for module_name in ("selective_scan", "selective_scan_cuda_core"):
        try:
            importlib.import_module(module_name)
            print(f"ok               compiled module={module_name}")
        except Exception as exc:
            failures.append(f"missing {module_name}")
            print(f"MISSING required compiled module={module_name} reason={exc}")

    if (root / ".git").is_dir():
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"ok               official revision={revision}")
        except Exception as exc:
            print(f"warning          cannot read source revision reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("RCDNET_REQUIRE_CUDA", "1").casefold() not in {
        "0",
        "false",
        "no",
    }
    if torch is not None:
        try:
            if require_cuda and not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("CUDA is unavailable")
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                print(
                    "ok               CUDA device="
                    + torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
                )
        except Exception as exc:
            failures.append("CUDA check failed")
            print(f"MISSING required working CUDA reason={exc}")

    if failures:
        print("\nRCDNet dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-rcdnet.txt")
        print("  bash scripts/bootstrap_rcdnet.sh")
        return 1
    print("\nRCDNet dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
