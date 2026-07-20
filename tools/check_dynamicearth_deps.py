from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "numpy": "numpy",
    "Pillow": "PIL",
    "opencv-python-headless": "cv2",
    "scikit-image": "skimage",
    "tqdm": "tqdm",
    "mmcv": "mmcv",
    "mmengine": "mmengine",
    "mmsegmentation": "mmseg",
    "einops": "einops",
    "ftfy": "ftfy",
    "regex": "regex",
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

    root = _path("DYNAMIC_EARTH_ROOT", "third_party/DynamicEarth")
    for path in (
        root / "dynamic_earth" / "sam_ext" / "mask_proposal.py",
        root / "dynamic_earth" / "comparator" / "bi_match.py",
        root / "dynamic_earth" / "identifier" / "segearth_ov_ext.py",
        root / "third_party" / "segment_anything" / "segment_anything" / "build_sam.py",
        root / "third_party" / "SimFeatUp" / "setup.py",
    ):
        _required_file("official DynamicEarth source", path, failures)

    sam_checkpoint = _path(
        "DYNAMIC_EARTH_SAM_CHECKPOINT",
        "/root/data/weight/dynamicearth/sam_vit_h_4b8939.pth",
    )
    segearth_weight = _path(
        "DYNAMIC_EARTH_SEGEARTH_WEIGHT",
        "/root/data/weight/dynamicearth/xclip_jbu_one_million_aid.ckpt",
    )
    _required_file("SAM ViT-H checkpoint", sam_checkpoint, failures)
    _required_file("SegEarth-OV SimFeatUp checkpoint", segearth_weight, failures)
    if sam_checkpoint.is_file() and sam_checkpoint.stat().st_size < 1_000_000_000:
        failures.append("SAM checkpoint is too small")
        print("INCOMPATIBLE     SAM checkpoint is unexpectedly small")

    extra_paths = (
        root,
        root / "third_party" / "segment_anything",
        root / "third_party" / "SegEarth_OV",
        root / "third_party" / "SimFeatUp",
    )
    import sys

    for path in extra_paths:
        if path.is_dir():
            sys.path.insert(0, str(path))
    for module_name in (
        "segment_anything",
        "adaptive_conv_cuda_impl",
        "adaptive_conv_cpp_impl",
        "mmcv._ext",
    ):
        try:
            importlib.import_module(module_name)
            print(f"ok               compiled/runtime module={module_name}")
        except Exception as exc:
            failures.append(f"missing {module_name}")
            print(f"MISSING required module={module_name} reason={exc}")

    if (root / ".git").is_dir():
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"ok               official revision={revision}")
        except Exception as exc:
            print(f"warning          cannot read source revision reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get(
        "DYNAMIC_EARTH_REQUIRE_CUDA", "1"
    ).casefold() not in {"0", "false", "no"}
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
        print("\nDynamicEarth dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-dynamicearth.txt")
        print("  bash scripts/bootstrap_dynamicearth.sh")
        return 1
    print("\nDynamicEarth dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
