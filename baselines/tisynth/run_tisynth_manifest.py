from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import einops
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        einops = np = torch = OmegaConf = Image = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "TISynth dependencies are missing. Use the official tisynth environment and run "
            "python tools/check_tisynth_deps.py."
        ) from exc


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [part for part in text.strip("\\").split("\\") if part]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _resolve(path: str) -> Path:
    return Path(_normalize_wsl_unc(path)).expanduser().resolve()


def _read_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("condition_image", "target_image", "reference_image"):
                if row.get(key):
                    row[key] = _normalize_wsl_unc(str(row[key]))
            if not row.get("reference_image"):
                raise ValueError("Every TISynth manifest row must contain reference_image.")
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _load_rgb(path: Path, size: int, *, nearest: bool = False) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        image = image.resize((size, size), resample)
    return image


def _save_rgb(image: Image.Image, path: Path, size: int, *, nearest: bool = False) -> None:
    output = image.convert("RGB")
    if output.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        output = output.resize((size, size), resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def _is_valid_rgb_file(path: Path, size: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == (size, size) and image.mode in {"RGB", "RGBA", "P"}
    except Exception:
        return False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def _latent_noise(names: list[str], base_seed: int, shape: tuple[int, int, int]) -> torch.Tensor:
    latents = []
    for name in names:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(base_seed, name))
        latents.append(torch.randn(shape, generator=generator, dtype=torch.float32))
    return torch.stack(latents, dim=0)


def _install_lightning_compat() -> None:
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except ImportError:
        return
    module_name = "pytorch_lightning.utilities.distributed"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.rank_zero_only = rank_zero_only
        sys.modules[module_name] = module


def _patched_config(config_path: Path, clip_version: str) -> Path:
    config = OmegaConf.load(str(config_path))
    cond_stage = config.model.params.cond_stage_config
    if "params" not in cond_stage or cond_stage.params is None:
        cond_stage.params = {}
    cond_stage.params.version = str(clip_version)
    with tempfile.NamedTemporaryFile("w", suffix="_tisynth.yaml", delete=False) as handle:
        temp_path = Path(handle.name)
    OmegaConf.save(config=config, f=str(temp_path))
    return temp_path


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    return payload


def _load_model(
    *,
    root: Path,
    config_path: Path,
    checkpoint: Path,
    clip_version: str,
    device: torch.device,
) -> tuple[Any, Any]:
    sys.path.insert(0, str(root))
    os.chdir(root)
    _install_lightning_compat()
    from cldm.ddim_hacked_ssl import DDIMSampler
    from cldm.model import create_model

    patched = _patched_config(config_path, clip_version)
    try:
        model = create_model(str(patched)).cpu()
    finally:
        patched.unlink(missing_ok=True)
    missing, unexpected = model.load_state_dict(_load_checkpoint(checkpoint), strict=False)
    print(
        f"[run_tisynth_manifest] checkpoint={checkpoint} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )
    if missing:
        print(f"[run_tisynth_manifest] first missing keys: {list(missing)[:20]}")
    if unexpected:
        print(f"[run_tisynth_manifest] first unexpected keys: {list(unexpected)[:20]}")
    model = model.to(device).eval()
    return model, DDIMSampler(model)


def _to_reference_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(image, dtype=np.float32) / 127.5 - 1.0 for image in images], axis=0)
    tensor = torch.from_numpy(array).to(device=device)
    return einops.rearrange(tensor, "b h w c -> b c h w").contiguous()


def _to_mask_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(image, dtype=np.float32) / 255.0 for image in images], axis=0)
    tensor = torch.from_numpy(array).to(device=device)
    return einops.rearrange(tensor, "b h w c -> b c h w").contiguous()


def _amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _generate(
    *,
    model: Any,
    sampler: Any,
    names: list[str],
    masks: list[Image.Image],
    references: list[Image.Image],
    prompts: list[str],
    resolution: int,
    steps: int,
    scale: float,
    strength: float,
    eta: float,
    seed: int,
    precision: str,
    device: torch.device,
) -> list[Image.Image]:
    batch_size = len(names)
    shape = (4, resolution // 8, resolution // 8)
    x_t = _latent_noise(names, seed, shape).to(device=device)
    with torch.no_grad(), _amp_context(device, precision), model.ema_scope():
        ref_tensor = _to_reference_tensor(references, device)
        reference_condition = model.get_learned_ssl_conditioning(ref_tensor).detach()
        control = _to_mask_tensor(masks, device)
        text_condition = model.get_learned_conditioning(prompts)
        empty_condition = model.get_learned_conditioning([""] * batch_size)
        cond = {
            "c_concat": [control],
            "c_crossattn": [text_condition],
            "c_ssl": [reference_condition],
        }
        uncond = {
            "c_concat": [control],
            "c_crossattn": [empty_condition],
            "c_ssl": [reference_condition],
        }
        model.control_scales = [float(strength)] * 13
        samples, _ = sampler.sample(
            int(steps),
            batch_size,
            shape,
            conditioning=cond,
            verbose=False,
            eta=float(eta),
            x_T=x_t,
            unconditional_guidance_scale=float(scale),
            unconditional_conditioning=uncond,
        )
        decoded = model.decode_first_stage(samples)
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
    arrays = (
        einops.rearrange(decoded, "b c h w -> b h w c").float().cpu().numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def _batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official TISynth inference from a Vistar JSONL manifest.")
    parser.add_argument("--tisynth_root", default="third_party/TISynth")
    parser.add_argument("--config", default="")
    parser.add_argument("--ckpt", required=True, help="LoveDA-trained TISynth checkpoint")
    parser.add_argument("--clip_version", default="openai/clip-vit-large-patch14")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=9.0, help="official batch_infer.py effective CFG")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = _resolve(args.tisynth_root)
    checkpoint = _resolve(args.ckpt)
    manifest = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    config_path = _resolve(args.config) if args.config else root / "models/cldm_ssl_v15_aia_v0_augmentation.yaml"
    if not root.is_dir():
        raise NotADirectoryError(f"TISynth root not found: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TISynth checkpoint not found: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"TISynth config not found: {config_path}")
    if int(args.resolution) % 8 != 0:
        raise ValueError("TISynth resolution must be divisible by 8.")
    if int(args.eval_size) != 512:
        raise ValueError("The unified LoveDA comparison protocol requires --eval_size 512.")
    if not torch.cuda.is_available():
        raise RuntimeError("TISynth 512x512 inference requires a CUDA GPU.")

    rows = _read_manifest(manifest, int(args.max_samples))
    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb", "reference_rgb"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "tisynth_root": str(root),
        "config": str(config_path),
        "ckpt": str(checkpoint),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "num_manifest_rows": len(rows),
        "task": "LoveDA ordinary semantic-mask-to-image generation",
        "reference_is_model_input": True,
        "paired_target_is_model_input": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _seed_everything(int(args.seed))
    device = torch.device("cuda")
    model, sampler = _load_model(
        root=root,
        config_path=config_path,
        checkpoint=checkpoint,
        clip_version=_normalize_wsl_unc(args.clip_version),
        device=device,
    )

    resolved_path = output_dir / "manifest_resolved.jsonl"
    with resolved_path.open("w", encoding="utf-8") as resolved:
        for row_batch in tqdm(_batches(rows, max(1, int(args.batch_size))), desc="TISynth inference batches"):
            records: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            for row in row_batch:
                name = str(row.get("name") or Path(row["condition_image"]).stem)
                pred_path = output_dir / "pred_rgb" / f"{name}_pred_rgb.png"
                native_path = output_dir / "pred_rgb_native" / f"{name}_pred_rgb_{args.resolution}.png"
                cond_out = output_dir / "cond_mask" / f"{name}_cond_mask.png"
                gt_out = output_dir / "gt_rgb" / f"{name}_gt_rgb.png"
                ref_out = output_dir / "reference_rgb" / f"{name}_reference_rgb.png"
                mask = _load_rgb(Path(row["condition_image"]), int(args.resolution), nearest=True)
                reference = _load_rgb(Path(row["reference_image"]), int(args.resolution))
                _save_rgb(mask, cond_out, int(args.eval_size), nearest=True)
                _save_rgb(reference, ref_out, int(args.eval_size))
                if row.get("target_image"):
                    target = _load_rgb(Path(row["target_image"]), int(args.eval_size))
                    _save_rgb(target, gt_out, int(args.eval_size))
                valid = _is_valid_rgb_file(pred_path, int(args.eval_size))
                record = {
                    "row": row,
                    "name": name,
                    "mask": mask,
                    "reference": reference,
                    "pred_path": pred_path,
                    "native_path": native_path,
                    "status": "skipped_existing" if valid and not args.overwrite else "pending",
                }
                records.append(record)
                if record["status"] == "pending":
                    pending.append(record)

            if pending:
                images = _generate(
                    model=model,
                    sampler=sampler,
                    names=[record["name"] for record in pending],
                    masks=[record["mask"] for record in pending],
                    references=[record["reference"] for record in pending],
                    prompts=[str(record["row"].get("prompt") or "") for record in pending],
                    resolution=int(args.resolution),
                    steps=int(args.ddim_steps),
                    scale=float(args.scale),
                    strength=float(args.strength),
                    eta=float(args.eta),
                    seed=int(args.seed),
                    precision=str(args.precision),
                    device=device,
                )
                for record, image in zip(pending, images):
                    _save_rgb(image, record["native_path"], int(args.resolution))
                    _save_rgb(image, record["pred_path"], int(args.eval_size))
                    record["status"] = "generated"

            for record in records:
                resolved.write(
                    json.dumps(
                        {
                            **record["row"],
                            "name": record["name"],
                            "pred_rgb": str(record["pred_path"]),
                            "pred_rgb_native": str(record["native_path"]),
                            "sample_seed": _sample_seed(int(args.seed), record["name"]),
                            "resolution": int(args.resolution),
                            "eval_size": int(args.eval_size),
                            "ddim_steps": int(args.ddim_steps),
                            "scale": float(args.scale),
                            "strength": float(args.strength),
                            "eta": float(args.eta),
                            "precision": str(args.precision),
                            "status": record["status"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"[run_tisynth_manifest] wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
