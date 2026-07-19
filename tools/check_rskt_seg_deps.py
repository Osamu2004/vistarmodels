from __future__ import annotations

import importlib
import os
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
    "fvcore": "fvcore",
    "iopath": "iopath",
    "yacs": "yacs",
    "pycocotools": "pycocotools",
    "detectron2": "detectron2",
}


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _check_file(
    label: str,
    path: Path,
    failures: list[str],
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing {label}")
        print(f"MISSING required {label}={path}")
        return
    print(f"ok               {label}={path} size={path.stat().st_size / (1024 ** 2):.1f} MiB")


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

    root = _path("RSKT_ROOT", "third_party/RSKT-Seg")
    source_files = (
        root / "RSKT_Seg" / "RSKT_Seg.py",
        root / "RSKT_Seg" / "modeling" / "transformer" / "RSKT_Seg_Predictor.py",
        root / "configs" / "vitl_336_DLRSD.yaml",
        root / "datasets" / "DLRSD.json",
        root / "detectron2" / "setup.py",
    )
    for path in source_files:
        _check_file("official RSKT-Seg source", path, failures)

    checkpoint = _path(
        "RSKT_CHECKPOINT",
        "/root/data/weight/rskt_seg/RSKT_Seg_DLRSD_ViT_L/model_final.pth",
    )
    clip_vitl = _path(
        "RSKT_CLIP_VITL",
        "/root/data/weight/rskt_seg/pretrained/ViT-L-14-336px.pt",
    )
    clip_vitb = _path(
        "RSKT_CLIP_VITB",
        "/root/data/weight/rskt_seg/pretrained/ViT-B-32.pt",
    )
    remote_clip = _path(
        "RSKT_REMOTE_CLIP",
        "/root/data/weight/rskt_seg/pretrained/RemoteCLIP-ViT-B-32.pt",
    )
    rsib = _path(
        "RSKT_RSIB",
        "/root/data/weight/rskt_seg/pretrained/RSIB.pth",
    )
    _check_file("RSKT-Seg DLRSD+ViT-L checkpoint", checkpoint, failures)
    _check_file("CLIP ViT-L/14@336", clip_vitl, failures)
    _check_file("CLIP ViT-B/32", clip_vitb, failures)
    _check_file("RemoteCLIP ViT-B/32", remote_clip, failures)
    _check_file("RSIB/DINO", rsib, failures)

    if (root / "RSKT_Seg" / "__init__.py").is_file() and "detectron2" in modules:
        sys.path.insert(0, str(root))
        try:
            from RSKT_Seg import add_RSKT_seg_config
            from RSKT_Seg.third_party import clip as official_clip

            _ = add_RSKT_seg_config, official_clip
            print("ok               official RSKT-Seg package imports")
        except Exception as exc:
            failures.append("official RSKT-Seg import failed")
            print(f"INCOMPATIBLE     official RSKT-Seg import reason={exc}")

    try:
        import detectron2._C  # type: ignore[import-not-found]

        print("ok               Detectron2 compiled extension imports")
    except Exception as exc:
        failures.append("Detectron2 compiled extension missing")
        print(f"INCOMPATIBLE     detectron2._C reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("RSKT_REQUIRE_CUDA", "1").lower() not in {
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
        print("\nRSKT-Seg dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-rskt-seg.txt")
        print(
            "  RSKT_INSTALL_DETECTRON2=1 bash scripts/bootstrap_rskt_seg.sh"
        )
        print(
            "  The bootstrap downloads all auxiliary weights automatically. "
            "Download only the official DLRSD+ViT-L model_final.pth manually "
            "to /root/data/weight/rskt_seg/RSKT_Seg_DLRSD_ViT_L/"
        )
        return 1

    print("\nRSKT-Seg dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
