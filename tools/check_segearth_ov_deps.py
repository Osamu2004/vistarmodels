from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEGEARTH_COMMIT = "3e22a969b32c6d751bdbba64a88a0b670e630f55"
EXPECTED_SIMFEATUP_COMMIT = "78a0ba70b1d6ea7283684a88c98ce338af4593ca"
EXPECTED_CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
EXPECTED_SIMFEATUP_SHA256 = "cabc594d0042535f3413ac89d5f0b8b3173aecf18e2e469fb91b015ea4de49d8"

DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "scipy": "scipy",
    "tifffile": "tifffile",
    "tqdm": "tqdm",
    "einops": "einops",
    "fairscale": "fairscale",
    "fsspec": "fsspec",
    "ftfy": "ftfy",
    "regex": "regex",
    "timm": "timm",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "mmcv": "mmcv",
    "mmengine": "mmengine",
    "mmsegmentation": "mmseg",
    "featup": "featup",
}


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(label: str, path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing {label}")
        print(f"MISSING required {label}={path}")
        return
    print(f"ok               {label}={path} size={path.stat().st_size}")


def _check_revision(
    label: str,
    root: Path,
    expected: str,
    failures: list[str],
) -> None:
    if not (root / ".git").is_dir():
        failures.append(f"missing {label} Git metadata")
        print(f"MISSING required {label} Git metadata={root}")
        return
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        failures.append(f"cannot read {label} revision")
        print(f"INCOMPATIBLE     {label} revision reason={exc}")
        return
    if revision != expected:
        failures.append(f"unexpected {label} revision")
        print(f"INCOMPATIBLE     {label} revision={revision} expected={expected}")
    else:
        print(f"ok               {label} revision={revision}")


def main() -> int:
    failures: list[str] = []
    modules: dict[str, object] = {}
    for pip_name, import_name in DEPENDENCIES.items():
        try:
            module = importlib.import_module(import_name)
            modules[import_name] = module
            try:
                version = metadata.version(pip_name)
            except Exception:
                version = str(getattr(module, "__version__", "unknown"))
            print(f"ok               {import_name:18s} version={version}")
        except Exception as exc:
            failures.append(f"missing {pip_name}")
            print(f"MISSING required {import_name:18s} pip={pip_name} reason={exc}")

    segearth_root = _path("SEGEARTH_OV_ROOT", str(REPO_ROOT / "third_party" / "SegEarth-OV"))
    simfeatup_root = _path("SIMFEATUP_ROOT", str(REPO_ROOT / "third_party" / "SimFeatUp"))
    source_files = (
        segearth_root / "segearth_segmentor.py",
        segearth_root / "open_clip" / "factory.py",
        segearth_root / "prompts" / "imagenet_template.py",
        segearth_root / "simfeatup_dev" / "upsamplers.py",
        simfeatup_root / "setup.py",
        simfeatup_root / "featup" / "adaptive_conv_cuda" / "adaptive_conv.py",
    )
    for path in source_files:
        _check_file("official source", path, failures)
    _check_revision("SegEarth-OV", segearth_root, EXPECTED_SEGEARTH_COMMIT, failures)
    _check_revision("SimFeatUp", simfeatup_root, EXPECTED_SIMFEATUP_COMMIT, failures)
    featup_module = modules.get("featup")
    if featup_module is not None:
        featup_file = Path(str(getattr(featup_module, "__file__", ""))).resolve()
        if simfeatup_root not in featup_file.parents:
            failures.append("featup imported from unexpected checkout")
            print(
                "INCOMPATIBLE     featup import path="
                f"{featup_file} expected_root={simfeatup_root}"
            )
        else:
            print(f"ok               featup import path={featup_file}")

    checkpoint = _path(
        "SEGEARTH_OV_SIMFEATUP",
        str(segearth_root / "simfeatup_dev" / "weights" / "xclip_jbu_one_million_aid.ckpt"),
    )
    clip_vitb = _path(
        "SEGEARTH_OV_CLIP_VITB",
        "/root/data/weight/segearth_ov/pretrained/ViT-B-16.pt",
    )
    _check_file("SegEarth-OV SimFeatUp checkpoint", checkpoint, failures)
    _check_file("CLIP ViT-B/16", clip_vitb, failures)
    for label, path, expected in (
        ("SegEarth-OV SimFeatUp checkpoint", checkpoint, EXPECTED_SIMFEATUP_SHA256),
        ("CLIP ViT-B/16", clip_vitb, EXPECTED_CLIP_SHA256),
    ):
        if path.is_file() and path.stat().st_size:
            actual = _sha256(path)
            if actual != expected:
                failures.append(f"invalid {label} checksum")
                print(f"INCOMPATIBLE     {label} sha256={actual} expected={expected}")
            else:
                print(f"ok               {label} sha256={actual}")

    for module_name in ("adaptive_conv_cuda_impl", "adaptive_conv_cpp_impl", "mmcv._ext"):
        try:
            importlib.import_module(module_name)
            print(f"ok               compiled/runtime module={module_name}")
        except Exception as exc:
            failures.append(f"missing {module_name}")
            print(f"MISSING required module={module_name} reason={exc}")

    if segearth_root.is_dir():
        sys.path.insert(0, str(segearth_root))
        try:
            imported = importlib.import_module("segearth_segmentor")
            _ = imported.SegEarthSegmentation
            print("ok               official SegEarth-OV package imports")
        except Exception as exc:
            failures.append("official SegEarth-OV import failed")
            print(f"INCOMPATIBLE     official SegEarth-OV import reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("SEGEARTH_OV_REQUIRE_CUDA", "1").casefold() not in {
        "0",
        "false",
        "no",
    }
    if torch is not None:
        try:
            if require_cuda and not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("CUDA is unavailable")
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                tensor = torch.ones((8, 8), device="cuda")  # type: ignore[attr-defined]
                _ = (tensor @ tensor).sum().item()
                print("ok               CUDA device=" + torch.cuda.get_device_name(0))  # type: ignore[attr-defined]
            else:
                print("skipped          CUDA runtime check")
        except Exception as exc:
            failures.append("CUDA check failed")
            print(f"MISSING required working CUDA reason={exc}")

    if failures:
        print("\nSegEarth-OV dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-segearth-ov.txt")
        print("  bash scripts/bootstrap_segearth_ov.sh")
        return 1
    print("\nSegEarth-OV dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
