from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class Dependency:
    pip_name: str
    import_name: str
    required: bool
    note: str


DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("torch", "torch", True, "CUDA-matched PyTorch runtime"),
    Dependency("torchvision", "torchvision", True, "DreamCD preview/grid utilities"),
    Dependency("numpy", "numpy", True, "image/tensor preprocessing"),
    Dependency("pillow", "PIL", True, "image IO"),
    Dependency("huggingface-hub", "huggingface_hub", False, "only needed for automatic checkpoint download"),
    Dependency("opencv-python", "cv2", True, "Albumentations interpolation constants"),
    Dependency("albumentations", "albumentations", True, "official DreamCD dataset resizing/cropping"),
    Dependency("omegaconf", "omegaconf", True, "DreamCD YAML config loading"),
    Dependency("pytorch-lightning", "pytorch_lightning", True, "LatentDiffusion LightningModule base class"),
    Dependency("einops", "einops", True, "tensor layout transforms"),
    Dependency("tqdm", "tqdm", True, "progress bars"),
    Dependency("termcolor", "termcolor", True, "official logger formatting"),
    Dependency("kornia", "kornia", True, "latent diffusion imports"),
    Dependency("torchmetrics", "torchmetrics", True, "pytorch-lightning compatibility in official env"),
    Dependency("test-tube", "test_tube", False, "only needed by the official training logger"),
    Dependency("taming-transformers", "taming", True, "VQ/VQGAN modules imported by DreamCD"),
    Dependency("transformers", "transformers", False, "official requirement; not needed by the wrapper path in normal inference"),
    Dependency("clip", "clip", True, "imported unconditionally by the official encoder module"),
    Dependency("imageio", "imageio", False, "official requirement"),
    Dependency("streamlit", "streamlit", False, "only needed for old official demos"),
    Dependency("pudb", "pudb", False, "debugger only"),
)


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _version(dep: Dependency, module: object) -> str:
    try:
        return metadata.version(dep.pip_name)
    except Exception:
        return str(getattr(module, "__version__", "unknown"))


def _dreamcd_root() -> Path:
    default_root = Path("third_party/DreamCD")
    return Path(_normalize_wsl_unc(os.environ.get("DREAMCD_ROOT", str(default_root)))).expanduser().resolve()


def _dreamcd_weight_root() -> Path:
    return Path(
        _normalize_wsl_unc(os.environ.get("DREAMCD_WEIGHT_ROOT", "/root/data/weight/dreamcd"))
    ).expanduser()


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _check_required_file(label: str, path: Path, failures: list[str], *, min_bytes: int = 1) -> None:
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.stat().st_size >= int(min_bytes):
        print(f"ok               {label:22s} path={resolved} size={resolved.stat().st_size}")
        return
    print(f"MISSING required {label:22s} path={resolved}")
    failures.append(f"{label}: {resolved}")


def _install_lightning_compat() -> None:
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except Exception:
        return
    module_name = "pytorch_lightning.utilities.distributed"
    if module_name in sys.modules:
        return
    import types

    compat_module = types.ModuleType(module_name)
    compat_module.rank_zero_only = rank_zero_only
    sys.modules[module_name] = compat_module


def main() -> int:
    missing_required: list[Dependency] = []
    missing_optional: list[Dependency] = []
    artifact_failures: list[str] = []
    runtime_failures: list[str] = []
    imported: dict[str, object] = {}

    root = _dreamcd_root()
    source_entry = root / "changeanywhere2_synthesis.py"
    if source_entry.is_file():
        sys.path.insert(0, str(root))
        print(f"ok               DreamCD source      path={root} entry={source_entry.name}")
    else:
        print(f"MISSING required DreamCD source      path={root}")
        artifact_failures.append(f"DreamCD source: {root}")

    config_path = Path(
        _normalize_wsl_unc(
            os.environ.get("DREAMCD_CONFIG", str(root / "configs/synthesis-wcsdm-second.yaml"))
        )
    ).expanduser()
    _check_required_file("DreamCD config", config_path, artifact_failures)

    weight_root = _dreamcd_weight_root()
    ldm_ckpt = Path(
        _normalize_wsl_unc(
            os.environ.get("DREAMCD_CKPT", str(weight_root / "second/ldm.ckpt"))
        )
    ).expanduser()
    vqvae_ckpt = Path(
        _normalize_wsl_unc(
            os.environ.get("DREAMCD_VQVAE_CKPT", str(weight_root / "second/vqvae.ckpt"))
        )
    ).expanduser()
    _check_required_file("LDM checkpoint", ldm_ckpt, artifact_failures, min_bytes=1024 * 1024)
    _check_required_file("VQ-VAE checkpoint", vqvae_ckpt, artifact_failures, min_bytes=1024 * 1024)

    for dep in DEPENDENCIES:
        try:
            module = importlib.import_module(dep.import_name)
        except Exception as exc:
            status = "MISSING required" if dep.required else "missing optional"
            print(f"{status:17s} {dep.import_name:22s} pip={dep.pip_name:24s} reason={exc}")
            if dep.required:
                missing_required.append(dep)
            else:
                missing_optional.append(dep)
            continue
        imported[dep.import_name] = module
        print(f"ok               {dep.import_name:22s} pip={dep.pip_name:24s} version={_version(dep, module)}")

    _install_lightning_compat()
    if (root / "changeanywhere2_synthesis.py").is_file():
        try:
            import changeanywhere2_synthesis  # noqa: F401
            import ldm.data.changeanywhere2  # noqa: F401
            import ldm.models.autoencoder  # noqa: F401
            import scripts.sample_diffusion  # noqa: F401

            print("ok               DreamCD modules     synthesis/dataset/VQ-VAE/model imports work")
        except Exception as exc:
            print(f"MISSING required DreamCD modules     reason={exc}")
            missing_required.append(Dependency("DreamCD modules", "ldm/scripts", True, "official module imports"))

    require_cuda = _is_truthy(os.environ.get("DREAMCD_REQUIRE_CUDA", "1"))
    torch_module = imported.get("torch")
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        print(
            "info             CUDA runtime          "
            f"required={int(require_cuda)} available={int(cuda_available)} "
            f"visible_devices={visible} device_count={cuda_count}"
        )
        if cuda_available:
            try:
                probe = torch_module.empty(1, device="cuda")
                device_index = int(probe.device.index or 0)
                device_name = str(torch_module.cuda.get_device_name(device_index))
                print(f"ok               CUDA allocation     device=cuda:{device_index} name={device_name}")
                del probe
            except Exception as exc:
                print(f"MISSING required CUDA allocation     reason={exc}")
                runtime_failures.append(f"CUDA allocation: {exc}")
        elif require_cuda:
            print("MISSING required CUDA runtime        torch.cuda.is_available() is false")
            runtime_failures.append("CUDA runtime unavailable")

    if missing_required:
        print("\nMissing required packages or broken official-module imports.")
        special_install_names = {"taming-transformers", "clip"}
        pip_missing = [
            dep
            for dep in missing_required
            if dep.pip_name not in {"torch", "torchvision", "DreamCD", "DreamCD modules"}
            and dep.pip_name not in special_install_names
        ]
        if pip_missing:
            install = " ".join(dep.pip_name for dep in pip_missing)
            print(f"pip install {install}")
        if any(dep.pip_name == "taming-transformers" for dep in missing_required):
            print(
                "The normal taming-transformers wheel can contain no importable `taming` source.\n"
                "Repair it with the pinned editable official checkout:\n"
                "python -m pip uninstall -y taming-transformers\n"
                "python -m pip install --no-deps -e "
                "'git+https://github.com/CompVis/taming-transformers.git@"
                "3ba01b241669f5ade541ce990f7650a3b8f65318#egg=taming-transformers'"
            )
        if any(dep.pip_name == "clip" for dep in missing_required):
            print(
                "Install OpenAI CLIP (not the unrelated PyPI package named `clip`):\n"
                "python -m pip install --no-deps --force-reinstall "
                "'git+https://github.com/openai/CLIP.git@"
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'"
            )
        if any(dep.pip_name in {"torch", "torchvision"} for dep in missing_required):
            print("Install torch/torchvision from your CUDA-matched PyTorch channel.")
        print("DreamCD's official requirement.txt pins an old CUDA 11.1 stack; prefer a separate conda env.")

    if artifact_failures:
        print("\nMissing required source/config/checkpoints.")
        for failure in artifact_failures:
            print(f"- {failure}")
        if any("checkpoint" in failure.lower() for failure in artifact_failures):
            print("Run: DREAMCD_DOWNLOAD_WEIGHTS=1 bash scripts/bootstrap_dreamcd.sh")

    if runtime_failures:
        print("\nCUDA runtime check failed.")
        print("Run with the intended GPU visible, for example: CUDA_VISIBLE_DEVICES=0 python tools/check_dreamcd_deps.py")
        print("For CPU-only static checking, set DREAMCD_REQUIRE_CUDA=0.")

    if missing_optional:
        print("\nOptional packages missing (not required for SECOND inference).")
        install = " ".join(dep.pip_name for dep in missing_optional)
        print(f"Only for official training/demo tools: pip install {install}")

    failed = bool(missing_required or artifact_failures or runtime_failures)
    if failed:
        print("\n[check_dreamcd_deps] FAIL")
        return 1
    print("\n[check_dreamcd_deps] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
