from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    import einops
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        einops = np = torch = Image = tqdm = None  # type: ignore[assignment]
    else:
        raise ImportError(
            "PLACE inference dependencies are missing. Install torch, einops, "
            "omegaconf, transformers, scipy, pillow, numpy, and tqdm first."
        ) from exc


BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
COMMA_TOKEN_ID = 267

LOVEDA_COLOR_TO_TEXT: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "background or unlabeled land-cover area",
    (255, 255, 255): "building roofs and built-up structures",
    (255, 0, 0): "roads and transportation surfaces",
    (0, 0, 255): "water bodies such as rivers ponds or lakes",
    (255, 255, 0): "barren land or bare soil",
    (0, 255, 0): "forest and tree-covered vegetation",
    (0, 255, 255): "agricultural fields and cropland",
}

LOVEDA_FOREGROUND_ORDER: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
)


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rgb(path: Path, size: int, *, nearest: bool) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        image = image.resize((size, size), resample)
    return image


def _save_rgb(image: Image.Image, path: Path, size: int | None = None, *, nearest: bool = False) -> None:
    out = image.convert("RGB")
    if size is not None and out.size != (size, size):
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
        out = out.resize((size, size), resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


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


def _label_texts(include_background: bool) -> dict[tuple[int, int, int], tuple[int, str]]:
    mapping: dict[tuple[int, int, int], tuple[int, str]] = {}
    next_label = 1
    for color in LOVEDA_FOREGROUND_ORDER:
        mapping[color] = (next_label, LOVEDA_COLOR_TO_TEXT[color])
        next_label += 1
    if include_background:
        mapping[(0, 0, 0)] = (next_label, LOVEDA_COLOR_TO_TEXT[(0, 0, 0)])
    return mapping


def _source_from_loveda_mask(
    mask_rgb: Image.Image,
    *,
    include_background: bool,
    include_unknown: bool,
) -> tuple[np.ndarray, dict[int, str], list[dict[str, Any]]]:
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.uint8)
    source = np.zeros(arr.shape[:2], dtype=np.float32)
    id_to_text: dict[int, str] = {}
    regions: list[dict[str, Any]] = []
    color_mapping = _label_texts(include_background)
    next_label = max([label for label, _ in color_mapping.values()] + [0]) + 1

    colors = np.unique(arr.reshape(-1, 3), axis=0)
    for color_arr in colors:
        color = tuple(int(v) for v in color_arr.tolist())
        mask = (
            (arr[..., 0] == color[0])
            & (arr[..., 1] == color[1])
            & (arr[..., 2] == color[2])
        )
        area = int(mask.sum())
        if color in color_mapping:
            label_id, text = color_mapping[color]
        elif include_unknown:
            if next_label > 182:
                continue
            label_id = next_label
            text = "land-cover region"
            next_label += 1
        else:
            continue
        source[mask] = float(label_id)
        id_to_text[label_id - 1] = text
        regions.append(
            {
                "color": [color[0], color[1], color[2]],
                "label_id": int(label_id),
                "class_index": int(label_id - 1),
                "text": text,
                "area": area,
            }
        )

    regions.sort(key=lambda item: item["label_id"])
    return source, id_to_text, regions


def _content_tokens(tokenizer: Any, text: str, max_length: int) -> list[int]:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    token_ids = encoding["input_ids"][0].tolist()
    out: list[int] = []
    for token_id in token_ids:
        if token_id == BOS_TOKEN_ID:
            continue
        if token_id == EOS_TOKEN_ID:
            break
        if token_id == COMMA_TOKEN_ID:
            continue
        out.append(int(token_id))
    return out


def _make_place_batch(
    *,
    source: np.ndarray,
    id_to_text: dict[int, str],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], str]:
    tokens = np.full((1, 77), EOS_TOKEN_ID, dtype=np.int64)
    tokens[0, 0] = BOS_TOKEN_ID
    tokens_cls = np.full((1, 77), EOS_TOKEN_ID, dtype=np.int64)
    tokens_cls[0, 0] = BOS_TOKEN_ID

    token_list: list[int] = []
    token_cls_list: list[int] = []
    prompt_parts: list[str] = []
    source_minus_1 = source - 1.0
    for class_index in sorted(np.unique(source_minus_1).astype(int).tolist()):
        if class_index == -1:
            continue
        text = id_to_text.get(class_index)
        if text is None:
            continue
        class_tokens = _content_tokens(tokenizer, text, max_length=max_length)
        if not class_tokens:
            continue
        token_list.extend(class_tokens)
        token_cls_list.extend([class_index] * len(class_tokens))
        prompt_parts.append(text)

    max_content = tokens.shape[1] - 1
    token_list = token_list[:max_content]
    token_cls_list = token_cls_list[:max_content]
    if token_list:
        tokens[0, 1 : len(token_list) + 1] = np.asarray(token_list, dtype=np.int64)
        tokens_cls[0, 1 : len(token_cls_list) + 1] = np.asarray(token_cls_list, dtype=np.int64)

    viewsource = np.repeat(source[:, :, None], 3, axis=2).astype(np.uint8).astype(np.float32) / 255.0
    hint = source_minus_1[:, :, None].astype(np.float32)
    # PLACE only uses `jpg` to infer batch shape in the official data path; this wrapper builds cond directly.
    jpg = np.zeros((source.shape[0], source.shape[1], 3), dtype=np.float32)

    batch = {
        "jpg": torch.from_numpy(jpg[None]).to(device),
        "tks": torch.from_numpy(tokens).to(device),
        "hint": torch.from_numpy(hint[None]).to(device),
        "viewcontrol": torch.from_numpy(viewsource[None]).to(device),
        "tokens_cls": torch.from_numpy(tokens_cls).to(device),
    }
    return batch, ",".join(prompt_parts)


def _install_place_source(place_root: Path) -> None:
    if not place_root.is_dir():
        raise NotADirectoryError(f"PLACE root not found: {place_root}")
    sys.path.insert(0, str(place_root))


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return state_dict.get("state_dict", state_dict)


def _load_place_model(
    *,
    place_root: Path,
    config_path: Path,
    ckpt_path: Path,
    device: torch.device,
):
    _install_place_source(place_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"PLACE config not found: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"PLACE checkpoint not found: {ckpt_path}")

    from ldm.models.diffusion.plms import PLMSSampler
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

    cwd = os.getcwd()
    os.chdir(place_root)
    try:
        config = OmegaConf.load(str(config_path))
        model = instantiate_from_config(config.model).cpu()
    finally:
        os.chdir(cwd)

    state_dict = _load_state_dict(ckpt_path, device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    sampler = PLMSSampler(model)
    return model, sampler


def _prepare_condition(model: Any, batch: dict[str, torch.Tensor]) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]:
    device = model.device
    tokens = batch["tks"].to(device)
    control = einops.rearrange(batch["hint"].to(device), "b h w c -> b c h w").contiguous().float()
    tokens_cls = batch["tokens_cls"].to(device).contiguous().float()

    cond_cross = model.get_learned_conditioning(tokens)

    uc_tokens = torch.zeros_like(tokens) + EOS_TOKEN_ID
    uc_tokens[:, 0] = BOS_TOKEN_ID
    uc_cross = model.get_learned_conditioning(uc_tokens)
    uc_tkscls = torch.zeros_like(tokens_cls) + EOS_TOKEN_ID
    uc_tkscls[:, 0] = BOS_TOKEN_ID

    cond = {"c_concat": [control], "c_crossattn": [cond_cross], "tkscls": [tokens_cls]}
    un_cond = {"c_concat": [control], "c_crossattn": [uc_cross], "tkscls": [uc_tkscls]}
    return cond, un_cond


def _generate_one(
    *,
    model: Any,
    sampler: Any,
    batch: dict[str, torch.Tensor],
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> Image.Image:
    _seed_everything(seed)
    cond, un_cond = _prepare_condition(model, batch)
    shape = (4, 512 // 8, 512 // 8)
    with torch.inference_mode():
        samples, _ = sampler.sample(
            int(steps),
            conditioning=cond,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=float(guidance_scale),
            unconditional_conditioning=un_cond,
            eta=0.0,
            x_T=None,
        )
        decoded = model.decode_first_stage(samples)
        image = (einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5)
        array = image.detach().cpu().numpy().clip(0, 255).astype(np.uint8)[0]
    return Image.fromarray(array).convert("RGB")


def _write_place_input(
    *,
    name: str,
    input_dir: Path,
    source: np.ndarray,
    regions: list[dict[str, Any]],
    prompt: str,
) -> tuple[Path, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    label_path = input_dir / f"{name}_label.png"
    json_path = input_dir / f"{name}.json"
    Image.fromarray(source.astype(np.uint8), mode="L").save(label_path)
    payload = {
        "prompt": prompt,
        "regions": regions,
        "label_png": str(label_path),
        "note": "PLACE source labels are 1-based foreground ids; 0 is ignored background.",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return label_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLACE inference from a Vistar-style JSONL manifest.")
    parser.add_argument("--place_root", default="third_party/PLACE")
    parser.add_argument("--config", default="", help="defaults to <place_root>/configs/stable-diffusion/PLACE.yaml")
    parser.add_argument("--ckpt", default="/root/data/weight/place/coco_best.ckpt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=512, help="PLACE native generation resolution; must stay 512")
    parser.add_argument("--eval_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="PLACE wrapper runs one image at a time")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--include_background", action="store_true")
    parser.add_argument("--include_unknown", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    place_root = _resolve_path(args.place_root)
    config_path = _resolve_path(args.config) if args.config else place_root / "configs/stable-diffusion/PLACE.yaml"
    ckpt_path = _resolve_path(args.ckpt)
    manifest = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)

    if int(args.resolution) != 512:
        raise ValueError("PLACE code has hard-coded 512x512 semantic attention; keep --resolution 512.")
    if int(args.batch_size) != 1:
        raise ValueError("PLACE wrapper only supports batch_size=1.")
    if not torch.cuda.is_available():
        raise RuntimeError("PLACE official PLMS sampler assumes CUDA. Run on a CUDA GPU.")

    rows = _read_manifest(manifest, int(args.max_samples))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest}")

    for subdir in ("pred_rgb", "pred_rgb_native", "cond_mask", "gt_rgb", "place_inputs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = torch.device("cuda")
    model, sampler = _load_place_model(
        place_root=place_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
    )
    tokenizer = model.cond_stage_model.tokenizer
    max_length = int(getattr(model.cond_stage_model, "max_length", 77))

    resolved_manifest = output_dir / "manifest_resolved.jsonl"
    with resolved_manifest.open("w", encoding="utf-8") as manifest_f:
        for index, row in enumerate(tqdm(rows, desc="PLACE inference")):
            name = str(row.get("name") or f"sample_{index:06d}")
            pred_path = output_dir / "pred_rgb" / f"{name}_pred_rgb.png"
            native_path = output_dir / "pred_rgb_native" / f"{name}_pred_rgb_512.png"
            cond_out = output_dir / "cond_mask" / f"{name}_cond_mask.png"
            gt_out = output_dir / "gt_rgb" / f"{name}_gt_rgb.png"

            condition_image = _load_rgb(Path(row["condition_image"]), 512, nearest=True)
            _save_rgb(condition_image, cond_out, size=int(args.eval_size), nearest=True)
            if row.get("target_image"):
                gt_rgb = _load_rgb(Path(row["target_image"]), int(args.eval_size), nearest=False)
                _save_rgb(gt_rgb, gt_out, size=int(args.eval_size), nearest=False)

            source, id_to_text, regions = _source_from_loveda_mask(
                condition_image,
                include_background=bool(args.include_background),
                include_unknown=bool(args.include_unknown),
            )
            batch, place_prompt = _make_place_batch(
                source=source,
                id_to_text=id_to_text,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
            )
            label_path, input_json = _write_place_input(
                name=name,
                input_dir=output_dir / "place_inputs",
                source=source,
                regions=regions,
                prompt=place_prompt,
            )

            has_valid_prediction = _is_valid_rgb_file(pred_path, int(args.eval_size))
            status = "skipped_existing" if has_valid_prediction and not args.overwrite else "pending"

            if status == "pending":
                image = _generate_one(
                    model=model,
                    sampler=sampler,
                    batch=batch,
                    steps=int(args.steps),
                    guidance_scale=float(args.guidance_scale),
                    seed=int(args.seed) + index,
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
                        "place_input_label": str(label_path),
                        "place_input_json": str(input_json),
                        "place_prompt": place_prompt,
                        "place_regions": regions,
                        "resolution": int(args.resolution),
                        "eval_size": int(args.eval_size),
                        "batch_size": int(args.batch_size),
                        "steps": int(args.steps),
                        "guidance_scale": float(args.guidance_scale),
                        "place_root": str(args.place_root),
                        "config": str(config_path),
                        "ckpt": str(args.ckpt),
                        "include_background": bool(args.include_background),
                        "include_unknown": bool(args.include_unknown),
                        "status": status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[run_place_manifest] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
