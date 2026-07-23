from __future__ import annotations

import gzip
import hashlib
import importlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPECTED_VIP_COMMIT = "5bd25ee03ec25c1538622cf7da661e8c0461e769"
EXPECTED_BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

DEPENDENCIES = {
    "torch": "torch",
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "ftfy": "ftfy",
    "regex": "regex",
    "omegaconf": "omegaconf",
}


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(label: str, path: Path, failures: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing {label}")
        print(f"MISSING required {label}={path}")
        return False
    print(f"ok               {label}={path} size={path.stat().st_size / 2**20:.1f} MiB")
    return True


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

    vip_root = _path("VIP_ROOT", str(REPO_ROOT / "third_party" / "VIP"))
    source_files = (
        vip_root / "dinosegmentor.py",
        vip_root / "dinov3" / "hub" / "backbones.py",
        vip_root / "dinov3" / "eval" / "text" / "dinotxt_model.py",
        vip_root / "prompts" / "imagenet_template.py",
    )
    for path in source_files:
        _check_file("official VIP source", path, failures)
    if (vip_root / ".git").is_dir():
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(vip_root), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != EXPECTED_VIP_COMMIT:
                failures.append("unexpected VIP source revision")
                print(
                    f"INCOMPATIBLE     VIP revision={revision} "
                    f"expected={EXPECTED_VIP_COMMIT}"
                )
            else:
                print(f"ok               VIP revision={revision}")
        except Exception as exc:
            failures.append("cannot read VIP source revision")
            print(f"INCOMPATIBLE     VIP revision reason={exc}")
    else:
        failures.append("missing VIP Git metadata")
        print(f"MISSING required VIP Git metadata={vip_root}")

    backbone = _path(
        "VIP_BACKBONE",
        "/root/data/weight/vip/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    dinotxt = _path(
        "VIP_DINOTXT",
        "/root/data/weight/vip/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    )
    bpe = _path(
        "VIP_BPE", "/root/data/weight/vip/bpe_simple_vocab_16e6.txt.gz"
    )
    for label, path, prefix in (
        ("DINOv3 ViT-L/16 backbone", backbone, "8aa4cbdd"),
        ("dino.txt head/text encoder", dinotxt, "a442d8f5"),
    ):
        if _check_file(label, path, failures):
            actual = _sha256(path)
            if not actual.startswith(prefix):
                failures.append(f"invalid {label} checksum")
                print(
                    f"INCOMPATIBLE     {label} sha256={actual} "
                    f"expected_prefix={prefix}"
                )
            else:
                print(f"ok               {label} sha256={actual}")
    if _check_file("dino.txt BPE vocabulary", bpe, failures):
        actual = _sha256(bpe)
        if actual != EXPECTED_BPE_SHA256:
            failures.append("invalid BPE checksum")
            print(
                f"INCOMPATIBLE     BPE sha256={actual} "
                f"expected={EXPECTED_BPE_SHA256}"
            )
        else:
            try:
                with gzip.open(bpe, "rb") as handle:
                    _ = handle.read(32)
                print(f"ok               dino.txt BPE sha256={actual}")
            except Exception as exc:
                failures.append("invalid BPE gzip payload")
                print(f"INCOMPATIBLE     BPE gzip reason={exc}")

    torch = modules.get("torch")
    require_cuda = os.environ.get("VIP_REQUIRE_CUDA", "1").casefold() not in {
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

    try:
        from baselines.vip.protocols import read_vip_class_groups

        for path in sorted((REPO_ROOT / "baselines" / "vip" / "configs").glob("*.txt")):
            groups = read_vip_class_groups(path)
            print(f"ok               VIP aliases={path.name} classes={len(groups)}")
    except Exception as exc:
        failures.append("VIP adapter import/config validation failed")
        print(f"INCOMPATIBLE     VIP adapter reason={exc}")

    if failures:
        print("\nVIP dependency check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSetup:")
        print("  python -m pip install -r requirements-vip.txt")
        print("  bash scripts/bootstrap_vip.sh")
        return 1
    print("\nVIP dependency check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
