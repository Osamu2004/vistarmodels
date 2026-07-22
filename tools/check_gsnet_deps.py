from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "tqdm": "tqdm",
    "einops": "einops",
    "timm": "timm",
    "open-clip-torch": "open_clip",
    "scipy": "scipy",
    "imageio": "imageio",
    "tifffile": "tifffile",
    "ftfy": "ftfy",
    "regex": "regex",
    "fvcore": "fvcore",
    "iopath": "iopath",
    "yacs": "yacs",
    "pycocotools": "pycocotools",
    "cloudpickle": "cloudpickle",
    "detectron2": "detectron2",
}


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _check_file(label: str, path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing {label}")
        print(f"MISSING required {label}={path}")
        return
    print(
        f"ok               {label}={path} "
        f"size={path.stat().st_size / (1024 ** 2):.1f} MiB"
    )


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
            print(
                f"MISSING required {import_name:18s} "
                f"pip={pip_name} reason={exc}"
            )

    root = _path("GSNET_ROOT", "third_party/GSNet")
    source_files = (
        root / "gs_net" / "GSNet.py",
        root / "gs_net" / "modeling" / "transformer" / "GSNetPredictor.py",
        root / "configs" / "vitb_384.yaml",
        root / "datasets" / "landdiscover.json",
    )
    for path in source_files:
        _check_file("official GSNet source", path, failures)

    checkpoint = _path(
        "GSNET_CHECKPOINT",
        "/root/data/weight/gsnet/GSNet_base.pth",
    )
    clip_vitb = _path(
        "GSNET_CLIP_VITB",
        "/root/data/weight/gsnet/pretrained/ViT-B-16.pt",
    )
    rsib = _path(
        "GSNET_RSIB",
        "/root/data/weight/rsib/RSIB.pth",
    )
    _check_file("GSNet LandDiscover50K checkpoint", checkpoint, failures)
    _check_file("CLIP ViT-B/16", clip_vitb, failures)
    _check_file("RSIB/DINO", rsib, failures)

    class_json_value = os.environ.get("GSNET_CLASS_JSON", "")
    foreground_class = os.environ.get("GSNET_FOREGROUND_CLASS", "").strip()
    if class_json_value:
        class_json = Path(class_json_value).expanduser().resolve()
        _check_file("GSNet test taxonomy", class_json, failures)
        if class_json.is_file():
            try:
                class_names = json.loads(class_json.read_text(encoding="utf-8"))
                if not isinstance(class_names, list) or not class_names:
                    raise ValueError("taxonomy must be a non-empty list")
                if any(
                    not isinstance(class_name, str) or not class_name.strip()
                    for class_name in class_names
                ):
                    raise ValueError(
                        "taxonomy entries must be non-empty strings"
                    )
                normalized = [
                    class_name.strip().casefold()
                    for class_name in class_names
                ]
                if len(normalized) != len(set(normalized)):
                    raise ValueError("taxonomy contains duplicate classes")
                if "background" in normalized:
                    raise ValueError(
                        "taxonomy must exclude GSNet's ignored background "
                        "channel"
                    )
                if foreground_class.casefold() not in normalized:
                    raise ValueError(
                        f"foreground class {foreground_class!r} is absent"
                    )
                print(
                    "ok               GSNet prediction protocol="
                    f"{len(normalized)} non-background classes, "
                    f"foreground={foreground_class}"
                )
            except Exception as exc:
                failures.append("invalid GSNet test taxonomy")
                print(
                    "INCOMPATIBLE     GSNet test taxonomy "
                    f"reason={exc}"
                )

    if (root / ".git").is_dir():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            print(f"ok               official GSNet revision={commit}")
        except Exception as exc:
            print(f"warning          cannot read GSNet revision reason={exc}")

    if (root / "gs_net" / "__init__.py").is_file() and "detectron2" in modules:
        sys.path.insert(0, str(root))
        try:
            from gs_net import add_cat_seg_config
            from gs_net.third_party import clip as official_clip

            _ = add_cat_seg_config, official_clip
            print("ok               official GSNet package imports")
        except Exception as exc:
            failures.append("official GSNet import failed")
            print(f"INCOMPATIBLE     official GSNet import reason={exc}")

    try:
        import detectron2._C  # type: ignore[import-not-found]

        print("ok               Detectron2 compiled extension imports")
    except Exception as exc:
        failures.append("Detectron2 compiled extension missing")
        print(f"INCOMPATIBLE     detectron2._C reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("GSNET_REQUIRE_CUDA", "1").lower() not in {
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
                print(
                    "ok               CUDA device="
                    + torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
                )
            else:
                print("skipped          CUDA runtime check")
        except Exception as exc:
            failures.append("CUDA check failed")
            print(f"MISSING required working CUDA reason={exc}")

    if failures:
        print("\nGSNet dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-gsnet.txt")
        print("  bash scripts/bootstrap_gsnet.sh")
        print(
            "  If Detectron2 is missing: "
            "GSNET_INSTALL_DETECTRON2=1 bash scripts/bootstrap_gsnet.sh"
        )
        return 1

    print("\nGSNet dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
