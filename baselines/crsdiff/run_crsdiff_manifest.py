from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

try:
    import cv2
    import einops
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        cv2 = einops = np = torch = Image = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "CRS-Diff inference dependencies are missing. Install this repo's "
            "requirements and the official CRS-Diff environment first."
        ) from exc


SLOT_TO_INDEX = {
    "mlsd": 0,
    "hed": 1,
    "sketch": 2,
    "road": 3,
    "midas": 4,
    "seg": 5,
}


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _load_rgb(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _save_rgb(array: np.ndarray, path: Path, size: int | None = None) -> None:
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _read_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["condition_image"] = _normalize_wsl_unc(row["condition_image"])
            if row.get("target_image"):
                row["target_image"] = _normalize_wsl_unc(row["target_image"])
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_local_control(condition_rgb: np.ndarray, slot: str, resolution: int, device: torch.device) -> torch.Tensor:
    if slot not in SLOT_TO_INDEX:
        raise ValueError(f"Unsupported CRS-Diff condition slot {slot!r}; choose from {sorted(SLOT_TO_INDEX)}")
    maps = [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(6)]
    maps[SLOT_TO_INDEX[slot]] = cv2.resize(condition_rgb, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    detected_maps = np.concatenate(maps, axis=2)
    local_control = torch.from_numpy(detected_maps.copy()).float().to(device) / 255.0
    local_control = einops.rearrange(local_control, "h w c -> 1 c h w").contiguous()
    return local_control


def _install_crsdiff_lightning_compat() -> None:
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except ImportError:
        return
    module_name = "pytorch_lightning.utilities.distributed"
    if module_name in sys.modules:
        return
    compat_module = types.ModuleType(module_name)
    compat_module.rank_zero_only = rank_zero_only
    sys.modules[module_name] = compat_module


def _make_patched_config(config_path: Path, clip_version: str) -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(config_path))
    cond_stage_config = config.model.params.cond_stage_config
    if "params" not in cond_stage_config or cond_stage_config.params is None:
        cond_stage_config.params = {}
    cond_stage_config.params.version = str(clip_version)

    with tempfile.NamedTemporaryFile("w", suffix="_crsdiff_config.yaml", delete=False) as f:
        OmegaConf.save(config=config, f=f.name)
        return Path(f.name)


def _load_crsdiff(
    crsdiff_root: Path,
    ckpt: Path,
    config_path: Path,
    device: torch.device,
    clip_version: str,
):
    sys.path.insert(0, str(crsdiff_root))
    os.chdir(crsdiff_root)
    _install_crsdiff_lightning_compat()
    from models.ddim_hacked import DDIMSampler
    from models.util import create_model, load_state_dict

    patched_config_path = _make_patched_config(config_path, clip_version)
    print(f"[run_crsdiff_manifest] clip_version={clip_version}")
    try:
        model = create_model(str(patched_config_path)).cpu()
    finally:
        patched_config_path.unlink(missing_ok=True)
    state = load_state_dict(str(ckpt), location=str(device))
    model.load_state_dict(state)
    model = model.to(device).eval()
    sampler = DDIMSampler(model)
    return model, sampler


def _generate_one(
    *,
    model: Any,
    sampler: Any,
    condition_rgb: np.ndarray,
    prompt: str,
    slot: str,
    resolution: int,
    ddim_steps: int,
    scale: float,
    strength: float,
    global_strength: float,
    eta: float,
    negative_prompt: str,
    added_prompt: str,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        local_control = _make_local_control(condition_rgb, slot, resolution, device)
        global_control = torch.zeros((1, 1536), dtype=torch.float32, device=device)
        metadata_control = torch.zeros((7,), dtype=torch.float32, device=device)

        prompt_full = f"{prompt}, {added_prompt}" if added_prompt else prompt
        cond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([prompt_full])],
            "global_control": [global_control],
        }
        un_cond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([negative_prompt])],
            "global_control": [torch.zeros_like(global_control)],
        }

        if hasattr(model, "local_control_scales"):
            model.local_control_scales = [float(strength)] * 13
        else:
            model.control_scales = [float(strength)] * 13
        shape = (4, resolution // 8, resolution // 8)
        samples, _ = sampler.sample(
            int(ddim_steps),
            1,
            shape,
            metadata_control,
            conditioning=cond,
            verbose=False,
            eta=float(eta),
            unconditional_guidance_scale=float(scale),
            unconditional_conditioning=un_cond,
            global_strength=float(global_strength),
        )
        decoded = model.decode_first_stage(samples)
        image = (einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5)
        return image[0].detach().cpu().numpy().clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official CRS-Diff inference from a JSONL manifest.")
    parser.add_argument("--crsdiff_root", default="third_party/CRS-Diff")
    parser.add_argument("--ckpt", default="/root/data/weight/crsdiff/last.ckpt")
    parser.add_argument("--config", default="", help="defaults to <crsdiff_root>/configs/crs.yaml")
    parser.add_argument(
        "--clip_version",
        default=os.environ.get("CRSDIFF_CLIP_VERSION", "openai/clip-vit-large-patch14"),
        help="HuggingFace repo id or local directory for CRS-Diff FrozenCLIPEmbedder.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_slot", choices=sorted(SLOT_TO_INDEX), default="seg")
    parser.add_argument("--resolution", type=int, default=512, help="native CRS-Diff generation resolution")
    parser.add_argument("--eval_size", type=int, default=256, help="saved pred_rgb size for fair metric comparison")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--global_strength", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--negative_prompt", default="Blurry, distorted, overexposed")
    parser.add_argument("--added_prompt", default="best quality, extremely detailed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    crsdiff_root = Path(_normalize_wsl_unc(args.crsdiff_root)).expanduser().resolve()
    ckpt = Path(_normalize_wsl_unc(args.ckpt)).expanduser().resolve()
    config_path = Path(_normalize_wsl_unc(args.config)).expanduser().resolve() if args.config else crsdiff_root / "configs/crs.yaml"
    manifest = Path(_normalize_wsl_unc(args.manifest)).expanduser().resolve()
    output_dir = Path(_normalize_wsl_unc(args.output_dir)).expanduser().resolve()

    if not crsdiff_root.is_dir():
        raise NotADirectoryError(f"CRS-Diff root not found: {crsdiff_root}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"CRS-Diff checkpoint not found: {ckpt}")
    if not config_path.is_file():
        raise FileNotFoundError(f"CRS-Diff config not found: {config_path}")

    rows = _read_manifest(manifest, args.max_samples)
    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sampler = _load_crsdiff(
        crsdiff_root,
        ckpt,
        config_path,
        device,
        _normalize_wsl_unc(args.clip_version),
    )

    resolved_manifest = output_dir / "manifest_resolved.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as manifest_f:
        for index, row in enumerate(tqdm(rows, desc="CRS-Diff inference")):
            name = str(row.get("name") or f"sample_{index:06d}")
            pred_path = output_dir / "pred_rgb" / f"{name}_pred_rgb.png"
            native_path = output_dir / "pred_rgb_native" / f"{name}_pred_rgb_{args.resolution}.png"
            cond_out = output_dir / "cond_mask" / f"{name}_cond_mask.png"
            gt_out = output_dir / "gt_rgb" / f"{name}_gt_rgb.png"

            condition_path = Path(row["condition_image"])
            condition_rgb = _load_rgb(condition_path, int(args.resolution))
            _save_rgb(condition_rgb, cond_out, size=int(args.eval_size))
            if row.get("target_image"):
                gt_rgb = _load_rgb(Path(row["target_image"]), int(args.eval_size))
                _save_rgb(gt_rgb, gt_out, size=int(args.eval_size))

            if pred_path.is_file() and not args.overwrite:
                status = "skipped_existing"
            else:
                image = _generate_one(
                    model=model,
                    sampler=sampler,
                    condition_rgb=condition_rgb,
                    prompt=str(row.get("prompt") or ""),
                    slot=args.condition_slot,
                    resolution=int(args.resolution),
                    ddim_steps=int(args.ddim_steps),
                    scale=float(args.scale),
                    strength=float(args.strength),
                    global_strength=float(args.global_strength),
                    eta=float(args.eta),
                    negative_prompt=str(args.negative_prompt),
                    added_prompt=str(args.added_prompt),
                    device=device,
                )
                _save_rgb(image, native_path)
                _save_rgb(image, pred_path, size=int(args.eval_size))
                status = "generated"

            manifest_f.write(
                json.dumps(
                    {
                        **row,
                        "name": name,
                        "pred_rgb": str(pred_path),
                        "pred_rgb_native": str(native_path),
                        "condition_slot": args.condition_slot,
                        "resolution": int(args.resolution),
                        "eval_size": int(args.eval_size),
                        "ddim_steps": int(args.ddim_steps),
                        "scale": float(args.scale),
                        "status": status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[run_crsdiff_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
