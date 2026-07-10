from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        np = torch = Image = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "DreamCD inference dependencies are missing. Install the official "
            "DreamCD environment first, then run tools/check_dreamcd_deps.py."
        ) from exc


DREAMCD_PALETTE_VALUES = [
    [128, 0, 0],
    [0, 255, 36],
    [148, 148, 148],
    [255, 255, 255],
    [34, 97, 38],
    [0, 69, 255],
    [75, 181, 73],
    [222, 31, 7],
]
DREAMCD_PALETTE_U8 = np.asarray(DREAMCD_PALETTE_VALUES, dtype=np.uint8) if np is not None else DREAMCD_PALETTE_VALUES
VISTAR_OUTPUT_DIRS = {
    "source_rgb": "source_rgb",
    "cond_mask": "cond_mask",
    "cond_mask_official": "cond_mask_official",
    "cond_mask_ids": "cond_mask_ids",
    "gt_rgb": "gt_rgb",
    "pred_rgb": "pred_rgb",
    "absdiff": "absdiff",
    "prompt": "prompts",
}

PATH_ALIASES: dict[str, tuple[str, ...]] = {
    "source_image": ("source_image", "img_A", "image_A", "pre_image", "t1_image", "image1"),
    "target_image": ("target_image", "style_image", "img_B", "image_B", "post_image", "t2_image", "image2"),
    "source_mask": ("source_mask", "mask_A", "label_A", "pre_mask", "t1_mask", "label1"),
    "target_mask": (
        "target_mask",
        "condition_image",
        "mask_B",
        "label_B",
        "post_mask",
        "t2_mask",
        "label2",
    ),
    "change_mask": ("change_mask", "bcd_mask", "binary_change_mask", "mask_change", "change"),
}


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _resolve_path(path: str) -> Path:
    return Path(_normalize_wsl_unc(path)).expanduser().resolve()


def _safe_stem(name: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    if not stem:
        stem = f"sample_{index:06d}"
    return stem


def _path_from_row(row: dict[str, Any], canonical_key: str, *, required: bool = True) -> str:
    for key in PATH_ALIASES[canonical_key]:
        value = row.get(key)
        if value:
            return _normalize_wsl_unc(str(value))
    if required:
        raise KeyError(
            f"manifest row is missing {canonical_key}; accepted aliases: "
            f"{', '.join(PATH_ALIASES[canonical_key])}"
        )
    return ""


def _read_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for aliases in PATH_ALIASES.values():
                for key in aliases:
                    if row.get(key):
                        row[key] = _normalize_wsl_unc(str(row[key]))
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rgb(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size > 0 and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def _save_rgb(image: Image.Image, path: Path, size: int | None = None) -> None:
    out = image.convert("RGB")
    if size is not None and out.size != (size, size):
        out = out.resize((size, size), Image.Resampling.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _save_l(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_mask_vis(mask_ids: np.ndarray, path: Path, palette_u8: np.ndarray) -> None:
    ids = mask_ids.astype(np.int64).clip(0, len(palette_u8) - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(palette_u8[ids], mode="RGB").save(path)


def _save_absdiff(pred_path: Path, gt_path: Path, output_path: Path, size: int) -> None:
    pred = np.asarray(_load_rgb(pred_path, size), dtype=np.int16)
    gt = np.asarray(_load_rgb(gt_path, size), dtype=np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.abs(pred - gt).astype(np.uint8), mode="RGB").save(output_path)


def _vistar_change_condition(target_ids: np.ndarray, change_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if target_ids.shape != change_mask.shape:
        raise ValueError(
            f"target semantic/change mask shape mismatch: {target_ids.shape} vs {change_mask.shape}"
        )
    # Vistar reserves id 0 for unchanged pixels. Shift DreamCD semantic ids by one
    # so every changed target class remains distinguishable in cond_mask_ids.
    cond_ids = np.where(change_mask == 0, target_ids.astype(np.int64) + 1, 0).astype(np.uint8)
    palette_u8 = np.concatenate(
        [np.asarray([[255, 255, 255]], dtype=np.uint8), np.asarray(DREAMCD_PALETTE_U8, dtype=np.uint8)],
        axis=0,
    )
    return cond_ids, palette_u8


def _make_vistar_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {key: root / dirname for key, dirname in VISTAR_OUTPUT_DIRS.items()}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _is_valid_rgb_file(path: Path, size: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode in {"RGB", "RGBA", "P"} and image.size == (int(size), int(size))
    except Exception:
        return False


def _nearest_palette_ids(rgb: np.ndarray, palette_u8: np.ndarray) -> np.ndarray:
    rgb_i32 = rgb[..., :3].astype(np.int32)
    palette = palette_u8.astype(np.int32)
    diff = rgb_i32[..., None, :] - palette[None, None, :, :]
    return np.sum(diff * diff, axis=-1).argmin(axis=-1).astype(np.uint8)


def _load_semantic_ids(path: Path, *, size: int, num_labels: int, semantic_rgb_mode: str) -> np.ndarray:
    mask = Image.open(path)
    if mask.mode in {"L", "P", "I", "I;16"}:
        arr = np.asarray(mask)
        if arr.ndim != 2:
            arr = np.asarray(mask.convert("L"))
        ids = arr.astype(np.int64)
    else:
        if semantic_rgb_mode == "error":
            raise ValueError(
                f"semantic mask is not single-channel: {path}. "
                "Pass --semantic_rgb_mode nearest_dreamcd_palette to map RGB masks."
            )
        ids = _nearest_palette_ids(np.asarray(mask.convert("RGB"), dtype=np.uint8), DREAMCD_PALETTE_U8).astype(np.int64)

    ids_max = int(ids.max()) if ids.size else 0
    ids_min = int(ids.min()) if ids.size else 0
    if ids_max >= int(num_labels) or ids_min < 0:
        unique = np.unique(ids)
        preview = ", ".join(str(int(v)) for v in unique[:20])
        raise ValueError(
            f"semantic mask values must be in [0, {int(num_labels) - 1}] for DreamCD, "
            f"got values [{preview}] in {path}"
        )

    image = Image.fromarray(ids.astype(np.uint8), mode="L")
    if size > 0 and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _load_change_mask(path: Path, *, size: int, mode: str) -> np.ndarray:
    mask = Image.open(path)
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    unique = set(int(v) for v in np.unique(arr).tolist())

    if mode == "auto":
        if unique.issubset({0, 255}):
            official = np.where(arr == 255, 255, 0)
        elif unique.issubset({0, 1}):
            official = np.where(arr == 1, 0, 255)
        else:
            official = np.where(arr > 0, 0, 255)
    elif mode == "white_unchanged":
        official = np.where(arr >= 128, 255, 0)
    elif mode == "zero_changed":
        official = np.where(arr == 0, 0, 255)
    elif mode == "nonzero_changed":
        official = np.where(arr != 0, 0, 255)
    else:
        raise ValueError(f"Unsupported binary_change_mode={mode!r}")

    image = Image.fromarray(official.astype(np.uint8), mode="L")
    if size > 0 and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _derive_official_change_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"cannot derive change mask from mismatched masks: {mask_a.shape} vs {mask_b.shape}")
    return np.where(mask_a != mask_b, 0, 255).astype(np.uint8)


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


def _install_torch_load_compat() -> None:
    original = torch.load
    if getattr(original, "_dreamcd_compat", False):
        return

    def compat_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        try:
            return original(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return original(*args, **kwargs)

    setattr(compat_load, "_dreamcd_compat", True)
    torch.load = compat_load  # type: ignore[assignment]


def _make_patched_config(config_path: Path, vqvae_ckpt: Path, runtime_dir: Path) -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(config_path))
    config.model.params.first_stage_config.params.ckpt_path = str(vqvae_ckpt)
    patched_path = runtime_dir / "dreamcd_config_patched.yaml"
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(patched_path))
    return patched_path


def _run_official_dreamcd(
    *,
    dreamcd_root: Path,
    config_path: Path,
    ckpt_path: Path,
    data_csv: Path,
    batch_size: int,
    preview_path: Path,
    seed: int,
    ddim_steps: int,
    change_background: bool,
    with_adain: bool,
    content_correlation_scale_low: float,
    noise_cond: bool,
    preview_step: int,
    with_preview: bool,
    only_building: bool,
) -> None:
    if str(dreamcd_root) not in sys.path:
        sys.path.insert(0, str(dreamcd_root))

    _seed_everything(seed)
    _install_lightning_compat()
    _install_torch_load_compat()

    old_cwd = Path.cwd()
    os.chdir(dreamcd_root)
    try:
        import changeanywhere2_synthesis as official
        from ldm.data.changeanywhere2 import ChangeAnywhere2
        from scripts.utils import create_logger, seed_anything

        official.os = os
        seed_anything(seed)

        dataset = ChangeAnywhere2(
            data_csv=str(data_csv),
            only_building=bool(only_building),
            with_adain=bool(with_adain),
        )
        preview_path.mkdir(parents=True, exist_ok=True)
        if with_preview:
            official.logger = create_logger(output_dir=str(preview_path))

        official.changeanywhere2_synthesis(
            str(config_path),
            str(ckpt_path),
            dataset,
            batch_size=int(batch_size),
            preview_path=str(preview_path),
            ddim_steps=int(ddim_steps),
            change_background=bool(change_background),
            with_adain=bool(with_adain),
            content_correlation_scale_low=float(content_correlation_scale_low),
            noise_cond=bool(noise_cond),
            preview_step=int(preview_step),
            with_preview=bool(with_preview),
        )
    finally:
        os.chdir(old_cwd)


def _write_official_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                " ".join(
                    [
                        str(record["runtime_source_image"]),
                        str(record["runtime_target_image"]),
                        str(record["runtime_source_mask"]),
                        str(record["runtime_target_mask"]),
                        str(record["runtime_change_mask"]),
                        str(record["native_path"]),
                    ]
                )
                + "\n"
            )


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official DreamCD SECOND synthesis from a JSONL manifest.")
    parser.add_argument("--dreamcd_root", default="third_party/DreamCD")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--runtime_dir",
        default="",
        help="optional DreamCD working directory; defaults to a temporary directory outside output_dir",
    )
    parser.add_argument("--config", default="", help="defaults to <dreamcd_root>/configs/synthesis-wcsdm-second.yaml")
    parser.add_argument("--ckpt", default="", help="defaults to /root/data/weight/dreamcd/second/ldm.ckpt")
    parser.add_argument("--vqvae_ckpt", default="", help="defaults to /root/data/weight/dreamcd/second/vqvae.ckpt")
    parser.add_argument("--resolution", type=int, default=256, help="DreamCD native input/output size")
    parser.add_argument("--eval_size", type=int, default=256, help="saved pred_rgb size for metric comparison")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_labels", type=int, default=8)
    _add_bool_arg(parser, "with_adain", False, "use target image B as DreamCD AdaIN style reference")
    _add_bool_arg(parser, "noise_cond", True, "use DreamCD noise conditioning")
    _add_bool_arg(parser, "change_background", True, "let DreamCD synthesize changed background regions")
    parser.add_argument("--only_building", action="store_true")
    parser.add_argument("--content_correlation_scale_low", type=float, default=0.7)
    parser.add_argument("--preview_step", type=int, default=50)
    parser.add_argument("--with_preview", action="store_true")
    parser.add_argument(
        "--semantic_rgb_mode",
        choices=["nearest_dreamcd_palette", "error"],
        default="nearest_dreamcd_palette",
        help="how to handle RGB semantic masks; DreamCD itself expects grayscale class-id masks",
    )
    parser.add_argument(
        "--binary_change_mode",
        choices=["auto", "white_unchanged", "zero_changed", "nonzero_changed"],
        default="auto",
        help="normalize input binary change mask to DreamCD convention: 0=changed, 255=unchanged",
    )
    args = parser.parse_args()

    dreamcd_root = _resolve_path(args.dreamcd_root)
    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    config_path = _resolve_path(args.config) if args.config else dreamcd_root / "configs/synthesis-wcsdm-second.yaml"
    weight_root = _resolve_path(os.environ.get("DREAMCD_WEIGHT_ROOT", "/root/data/weight/dreamcd"))
    ckpt_default = os.environ.get("DREAMCD_CKPT", str(weight_root / "second/ldm.ckpt"))
    vqvae_default = os.environ.get("DREAMCD_VQVAE_CKPT", str(weight_root / "second/vqvae.ckpt"))
    ckpt_path = _resolve_path(args.ckpt or ckpt_default)
    vqvae_ckpt = _resolve_path(args.vqvae_ckpt or vqvae_default)

    if not dreamcd_root.is_dir():
        raise NotADirectoryError(f"DreamCD root not found: {dreamcd_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"DreamCD config not found: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"DreamCD LDM checkpoint not found: {ckpt_path}")
    if not vqvae_ckpt.is_file():
        raise FileNotFoundError(f"DreamCD VQ-VAE checkpoint not found: {vqvae_ckpt}")

    rows = _read_manifest(manifest, int(args.max_samples))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")

    dirs = _make_vistar_output_dirs(output_dir)
    temporary_runtime = None
    if args.runtime_dir:
        runtime_dir = _resolve_path(args.runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary_runtime = tempfile.TemporaryDirectory(prefix="dreamcd_runtime_")
        runtime_dir = Path(temporary_runtime.name)
    for subdir in ("img_A", "img_B", "mask_A", "mask_B", "bcd_mask", "pred_rgb_native", "preview"):
        (runtime_dir / subdir).mkdir(parents=True, exist_ok=True)

    patched_config = _make_patched_config(config_path, vqvae_ckpt, runtime_dir)
    official_csv = runtime_dir / "dreamcd_sample_list.txt"
    preview_path = runtime_dir / "preview"

    all_records: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for index, row in enumerate(tqdm(rows, desc="Preparing DreamCD inputs")):
        raw_name = str(row.get("name") or f"sample_{index:06d}")
        name = _safe_stem(raw_name, index)
        if name in used_names:
            name = f"{index:06d}_{name}"
        used_names.add(name)

        source_image_path = Path(_path_from_row(row, "source_image"))
        target_image_path_value = _path_from_row(row, "target_image")
        target_image_path = Path(target_image_path_value)
        source_mask_path = Path(_path_from_row(row, "source_mask"))
        target_mask_path = Path(_path_from_row(row, "target_mask"))
        change_mask_value = _path_from_row(row, "change_mask", required=False)
        change_mask_path = Path(change_mask_value) if change_mask_value else None

        pred_path = dirs["pred_rgb"] / f"{name}_pred_rgb.png"
        native_path = runtime_dir / "pred_rgb_native" / f"{name}_pred_rgb_{args.resolution}.png"
        source_rgb_out = dirs["source_rgb"] / f"{name}_source_rgb.png"
        gt_rgb_out = dirs["gt_rgb"] / f"{name}_gt_rgb.png"
        cond_mask_out = dirs["cond_mask"] / f"{name}_cond_mask.png"
        cond_mask_official_out = dirs["cond_mask_official"] / f"{name}_cond_mask_official.png"
        cond_mask_ids_out = dirs["cond_mask_ids"] / f"{name}_cond_mask_ids.png"
        absdiff_out = dirs["absdiff"] / f"{name}_absdiff.png"
        prompt_out = dirs["prompt"] / f"{name}.txt"

        runtime_source_image = runtime_dir / "img_A" / f"{name}.png"
        runtime_target_image = runtime_dir / "img_B" / f"{name}.png"
        runtime_source_mask = runtime_dir / "mask_A" / f"{name}.png"
        runtime_target_mask = runtime_dir / "mask_B" / f"{name}.png"
        runtime_change_mask = runtime_dir / "bcd_mask" / f"{name}.png"

        source_rgb = _load_rgb(source_image_path, int(args.resolution))
        target_rgb = _load_rgb(target_image_path, int(args.resolution))
        runtime_target_rgb = target_rgb if args.with_adain else source_rgb
        _save_rgb(source_rgb, runtime_source_image)
        # Keep real target B out of the official inference input unless AdaIN is explicitly enabled.
        _save_rgb(runtime_target_rgb, runtime_target_image)
        _save_rgb(source_rgb, source_rgb_out, size=int(args.eval_size))
        _save_rgb(target_rgb, gt_rgb_out, size=int(args.eval_size))

        source_ids = _load_semantic_ids(
            source_mask_path,
            size=int(args.resolution),
            num_labels=int(args.num_labels),
            semantic_rgb_mode=str(args.semantic_rgb_mode),
        )
        target_ids = _load_semantic_ids(
            target_mask_path,
            size=int(args.resolution),
            num_labels=int(args.num_labels),
            semantic_rgb_mode=str(args.semantic_rgb_mode),
        )
        if change_mask_path is not None:
            change_mask = _load_change_mask(
                change_mask_path,
                size=int(args.resolution),
                mode=str(args.binary_change_mode),
            )
            change_mask_source = str(change_mask_path)
        else:
            change_mask = _derive_official_change_mask(source_ids, target_ids)
            change_mask_source = "derived_from_source_target_masks"

        _save_l(source_ids, runtime_source_mask)
        _save_l(target_ids, runtime_target_mask)
        _save_l(change_mask, runtime_change_mask)
        cond_ids, cond_palette_u8 = _vistar_change_condition(target_ids, change_mask)
        _save_mask_vis(cond_ids, cond_mask_out, cond_palette_u8)
        _save_mask_vis(cond_ids, cond_mask_official_out, cond_palette_u8)
        _save_l(cond_ids, cond_mask_ids_out)
        prompt_text = str(row.get("prompt") or "DreamCD semantic change image synthesis.")
        prompt_out.write_text(prompt_text + "\n", encoding="utf-8")

        has_valid_prediction = _is_valid_rgb_file(pred_path, int(args.eval_size))
        status = "skipped_existing" if has_valid_prediction and not args.overwrite else "pending"
        record = {
            "row": row,
            "name": name,
            "raw_name": raw_name,
            "source_image": str(source_image_path),
            "target_image": str(target_image_path),
            "source_mask": str(source_mask_path),
            "target_mask": str(target_mask_path),
            "change_mask": change_mask_source,
            "runtime_source_image": runtime_source_image,
            "runtime_target_image": runtime_target_image,
            "runtime_source_mask": runtime_source_mask,
            "runtime_target_mask": runtime_target_mask,
            "runtime_change_mask": runtime_change_mask,
            "source_rgb": source_rgb_out,
            "gt_rgb": gt_rgb_out,
            "cond_mask": cond_mask_out,
            "cond_mask_official": cond_mask_official_out,
            "cond_mask_ids": cond_mask_ids_out,
            "absdiff": absdiff_out,
            "prompt_path": prompt_out,
            "prompt": prompt_text,
            "pred_path": pred_path,
            "native_path": native_path,
            "status": status,
        }
        if status == "pending":
            pending_records.append(record)
        all_records.append(record)

    if pending_records:
        _write_official_csv(pending_records, official_csv)
        _run_official_dreamcd(
            dreamcd_root=dreamcd_root,
            config_path=patched_config,
            ckpt_path=ckpt_path,
            data_csv=official_csv,
            batch_size=int(args.batch_size),
            preview_path=preview_path,
            seed=int(args.seed),
            ddim_steps=int(args.ddim_steps),
            change_background=bool(args.change_background),
            with_adain=bool(args.with_adain),
            content_correlation_scale_low=float(args.content_correlation_scale_low),
            noise_cond=bool(args.noise_cond),
            preview_step=int(args.preview_step),
            with_preview=bool(args.with_preview),
            only_building=bool(args.only_building),
        )

        for record in pending_records:
            native_path = Path(record["native_path"])
            if not native_path.is_file():
                raise FileNotFoundError(f"DreamCD did not produce expected output: {native_path}")
            image = Image.open(native_path).convert("RGB")
            _save_rgb(image, Path(record["pred_path"]), size=int(args.eval_size))
            record["status"] = "generated"
    else:
        _write_official_csv([], official_csv)

    for record in all_records:
        pred_path = Path(record["pred_path"])
        gt_path_value = record["gt_rgb"]
        if pred_path.is_file() and Path(gt_path_value).is_file():
            _save_absdiff(pred_path, Path(gt_path_value), Path(record["absdiff"]), int(args.eval_size))

    directions = list(
        dict.fromkeys(str(record["row"].get("direction") or "t1_to_t2") for record in all_records)
    )
    class_names = ["unchanged"] + [f"dreamcd_semantic_{idx}" for idx in range(len(DREAMCD_PALETTE_U8))]
    classes = [
        {"id": idx, "name": class_names[idx], "rgb": cond_palette_u8[idx].tolist()}
        for idx in range(len(cond_palette_u8))
    ]
    class_map = {
        "dataset": "SECOND",
        "split": str(all_records[0]["row"].get("split") or "unknown"),
        "task": "bidirectional_change_generation" if len(directions) > 1 else f"{directions[0]}_change_generation",
        "directions": directions,
        "label_mode": "dreamcd_semantic_pair",
        "eval_palette_mode": "official",
        "palette_seed": 0,
        "official_palette_for_gt": classes,
        "classes": classes,
        "model": "DreamCD",
        "with_adain": bool(args.with_adain),
    }
    with (output_dir / "class_map.json").open("w", encoding="utf-8") as class_map_f:
        json.dump(class_map, class_map_f, ensure_ascii=False, indent=2)

    for direction in directions:
        direction_prompt = next(
            record["prompt"]
            for record in all_records
            if str(record["row"].get("direction") or "t1_to_t2") == direction
        )
        (output_dir / f"prompt_{direction}_raw.txt").write_text(direction_prompt + "\n", encoding="utf-8")
        (output_dir / f"prompt_{direction}_effective.txt").write_text(direction_prompt + "\n", encoding="utf-8")

    prompts_manifest = output_dir / "prompts.jsonl"
    with prompts_manifest.open("w", encoding="utf-8") as manifest_f:
        for record in all_records:
            row = record["row"]
            payload = {
                "name": record["name"],
                "direction": str(row.get("direction") or "t1_to_t2"),
                "source_path": record["source_image"],
                "target_path": record["target_image"],
                "source_mask_path": record["source_mask"],
                "target_mask_path": record["target_mask"],
                "change_path": record["change_mask"],
                "prompt_raw": record["prompt"],
                "prompt_effective": record["prompt"],
                "resize_size": int(args.eval_size),
                "resumed": record["status"] == "skipped_existing",
                "source_rgb": str(record["source_rgb"]),
                "cond_mask": str(record["cond_mask"]),
                "cond_mask_official": str(record["cond_mask_official"]),
                "cond_mask_ids": str(record["cond_mask_ids"]),
                "gt_rgb": str(record["gt_rgb"]),
                "pred_rgb": str(record["pred_path"]),
                "absdiff": str(record["absdiff"]),
                "dreamcd_root": str(dreamcd_root),
                "config": str(config_path),
                "ckpt": str(ckpt_path),
                "vqvae_ckpt": str(vqvae_ckpt),
                "resolution": int(args.resolution),
                "eval_size": int(args.eval_size),
                "batch_size": int(args.batch_size),
                "ddim_steps": int(args.ddim_steps),
                "with_adain": bool(args.with_adain),
                "noise_cond": bool(args.noise_cond),
                "change_background": bool(args.change_background),
                "content_correlation_scale_low": float(args.content_correlation_scale_low),
                "semantic_rgb_mode": str(args.semantic_rgb_mode),
                "binary_change_mode": str(args.binary_change_mode),
                "status": record["status"],
            }
            manifest_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if pending_records:
        missing = [record["pred_path"] for record in pending_records if not Path(record["pred_path"]).is_file()]
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(f"DreamCD missing pred_rgb outputs:\n{preview}")

    if temporary_runtime is not None:
        temporary_runtime.cleanup()

    print(f"[run_dreamcd_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
