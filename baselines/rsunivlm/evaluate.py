"""Restartable LoveDA/LEVIR-CD inference with the same RSUniVLM weights."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from baselines.binary_boundary_wfm import (
    aggregate_binary_boundary_wfm,
    score_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)
from baselines.levircd_common import (
    binary_counts,
    binary_metrics,
    make_output_dirs,
    save_artifacts,
)
from baselines.segearth_ov.protocols import (
    EvalSample,
    LOVEDA_CLASSES,
    colorize_mask,
    confusion_matrix,
    load_target,
    metrics_from_confusion,
    normalize_path,
)
from baselines.rsunivlm.protocols import (
    CD_PROMPT, GRID_SIZE, LOVEDA_PROMPT, MAX_DESCRIPTOR_SIDE, PROTOCOL_VERSION, RLE_GRID_POLICY,
    Sample, checkpoint_identity, decode_rle, discover, expected_samples,
    fingerprint, fuse_change, input_identity, loveda_spatial_settings, palette, restore_grid, source_revision,
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(prediction.astype(np.uint8)).save(temporary, format="PNG")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def cached_prediction(path: Path, receipt: dict[str, Any] | None, *,
                      identity: str, shape: tuple[int, int], num_classes: int) -> np.ndarray | None:
    if receipt is None or receipt.get("identity") != identity:
        return None
    try:
        with Image.open(path) as image:
            prediction = np.asarray(image)
        if prediction.ndim != 2 or prediction.shape != shape:
            return None
        if num_classes == 2:
            if not np.all(np.isin(prediction, (0, 1, 255))):
                return None
            prediction = (prediction > 0).astype(np.uint8)
        if not np.all((prediction >= 0) & (prediction < num_classes)):
            return None
        # A valid but manually replaced PNG must not reuse an old receipt.
        if receipt.get("prediction_sha256") != hashlib.sha256(prediction.astype(np.uint8).tobytes()).hexdigest():
            return None
        return prediction.astype(np.uint8)
    except (OSError, ValueError):
        return None


def decode_saved_response(record: dict[str, Any], dataset: str) -> np.ndarray:
    generation = record.get("generation") or {}
    if generation.get("hit_token_limit") and not generation.get("ended_with_eos"):
        raise ValueError("Generation reached MAX_NEW_TOKENS without EOS; refusing a truncated mask")
    return decode_rle(record["response"], dataset, grid_size=None)


def query_grid(backend: Any, images: list[Image.Image], prompt: str, dataset: str,
               raw_path: Path, identity: str, *, overwrite: bool,
               spatial_metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, bool]:
    cached = None if overwrite else read_json(raw_path)
    if cached and cached.get("identity") == identity:
        try:
            return decode_saved_response(cached, dataset), True
        except (ValueError, KeyError):
            pass
    response = backend.generate(images, prompt)
    # Keep the unmodified text even if strict decoding subsequently fails.
    record = {"identity": identity, "prompt": prompt, "response": response}
    generation = getattr(backend, "last_generation", None)
    if isinstance(generation, dict):
        record["generation"] = generation
    if spatial_metadata is not None:
        record["spatial"] = spatial_metadata
    write_json(raw_path, record)
    try:
        grid = decode_saved_response(record, dataset)
    except (ValueError, KeyError) as exc:
        write_json(raw_path, {**record, "decode_error": str(exc)})
        raise
    write_json(raw_path, {**record, "grid_shape": list(grid.shape), "grid_policy": RLE_GRID_POLICY})
    return grid, False


def predict_loveda_tiled(backend: Any, image: Image.Image, sample: Sample,
                         raw_dir: Path, identity: str, *, tile_size: int,
                         overwrite: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict native RGB crops only; each padded crop is restored before trim."""
    settings = loveda_spatial_settings(tile_size)
    if not tile_size:
        raise ValueError("Tiled prediction needs a positive source tile size")
    width, height = image.size
    if min(width, height) <= 0 or image.mode != "RGB":
        raise ValueError("Expected a nonempty RGB source image")
    prediction = np.empty((height, width), dtype=np.uint8)
    queries, reused_queries = 0, 0
    descriptor_shapes: list[list[int]] = []
    # Iterate row-major like Alters. Only one padded source tile is live at a
    # time; GT is not read or passed into this function.
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            valid_h, valid_w = min(tile_size, height-y), min(tile_size, width-x)
            tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            tile.paste(image.crop((x, y, x+valid_w, y+valid_h)), (0, 0))
            tile_metadata = dict(source_box=[x, y, x+valid_w, y+valid_h],
                source_extent=[width, height], source_tile_size=tile_size,
                padding=settings["rgb_padding"], valid_height=valid_h, valid_width=valid_w)
            tile_id = fingerprint([identity, "loveda_source_tile", tile_metadata])
            raw_path = raw_dir / f"{sample.name}_tile{tile_size}_y{y}_x{x}_seg.json"
            grid, reused = query_grid(backend, [tile], LOVEDA_PROMPT, "loveda",
                raw_path, tile_id, overwrite=overwrite, spatial_metadata=tile_metadata)
            # Do not resize the descriptor directly to a partial border crop:
            # that would stretch the prediction for the padded region into GT.
            restored = restore_grid(grid, (tile_size, tile_size))
            descriptor_shapes.append(list(grid.shape))
            prediction[y:y+valid_h, x:x+valid_w] = restored[:valid_h, :valid_w]
            queries += 1
            reused_queries += int(reused)
    return prediction, dict(queries=queries, reused_queries=reused_queries,
                            num_tiles=queries, source_tile_size=tile_size,
                            descriptor_shapes=descriptor_shapes)


def predict_sample(backend: Any, images: list[Image.Image], sample: Sample, dataset: str,
                   raw_dir: Path, identity: str, *, cd_direction: str,
                   overwrite: bool, loveda_tile_size: int = 512) -> tuple[np.ndarray, dict[str, Any]]:
    if dataset == "loveda":
        loveda_spatial_settings(loveda_tile_size)
        if len(images) != 1:
            raise ValueError("LoveDA segmentation takes exactly one RGB source image")
        if loveda_tile_size:
            return predict_loveda_tiled(backend, images[0], sample, raw_dir, identity,
                                       tile_size=loveda_tile_size, overwrite=overwrite)
        grid, reused = query_grid(backend, images, LOVEDA_PROMPT, dataset,
                                  raw_dir / f"{sample.name}_seg.json", identity, overwrite=overwrite)
        return restore_grid(grid, (images[0].height, images[0].width)), {
            "queries": 1, "reused_queries": int(reused), "descriptor_shapes": [list(grid.shape)],
        }
    grids, reused_count = [], 0
    directions = ("forward", "reverse") if cd_direction == "symmetric_union" else ("forward",)
    for direction in directions:
        temporal_images = images if direction == "forward" else list(reversed(images))
        grid, reused = query_grid(
            backend, temporal_images, CD_PROMPT, dataset,
            raw_dir / f"{sample.name}_{direction}.json",
            fingerprint([identity, direction]), overwrite=overwrite,
        )
        grids.append(grid)
        reused_count += int(reused)
    restored = [restore_grid(grid, (images[0].height, images[0].width)) for grid in grids]
    return fuse_change(restored), {
        "queries": len(directions), "reused_queries": reused_count,
        "descriptor_shapes": [list(grid.shape) for grid in grids],
    }


def load_sample(sample: Sample, dataset: str) -> tuple[list[Image.Image], np.ndarray]:
    images = []
    for path in sample.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if len({image.size for image in images}) != 1:
        raise ValueError(f"Temporal image extents differ: {sample.name}")
    with Image.open(sample.label) as image:
        raw = np.asarray(image)
    shape = (images[0].height, images[0].width)
    if raw.ndim != 2 or raw.shape != shape:
        raise ValueError(f"Expected native-extent, single-channel GT at {sample.label}, got {raw.shape}")
    if dataset == "loveda":
        record = EvalSample(sample.name, sample.images[0], sample.label)
        target = load_target(record, "loveda", height=shape[0], width=shape[1])
    else:
        if not np.all(np.isin(raw, (0, 1, 255))):
            raise ValueError(f"Unexpected binary LEVIR-CD label values at {sample.label}")
        target = (raw > 0).astype(np.uint8)
    return images, target


def make_dirs(root: Path) -> dict[str, Path]:
    directories = make_output_dirs(root)
    for name in ("input", "gt_rgb", "overlay", "raw_text", "receipts", "errors"):
        directories[name] = root / name
        directories[name].mkdir(parents=True, exist_ok=True)
    return directories


def save_images(dirs: dict[str, Path], sample: Sample, images: list[Image.Image],
                prediction: np.ndarray, target: np.ndarray, dataset: str) -> None:
    if dataset == "levircd":
        save_artifacts(dirs, sample.name, np.asarray(images[0]), np.asarray(images[1]),
                       target, prediction, save_images=True)
        return
    colors = palette(dataset)
    prediction_rgb = colorize_mask(prediction, colors, ignore_index=None)
    target_rgb = colorize_mask(target, colors, ignore_index=255)
    images[0].save(dirs["input"] / f"{sample.name}.png")
    Image.fromarray(prediction_rgb).save(dirs["pred_rgb"] / f"{sample.name}_pred_rgb.png")
    Image.fromarray(target_rgb).save(dirs["gt_rgb"] / f"{sample.name}.png")
    Image.fromarray(target.astype(np.uint8)).save(dirs["gt_mask"] / f"{sample.name}.png")
    # Prediction visualization must not depend on GT validity/support.
    overlay = 0.55 * np.asarray(images[0], dtype=np.float32) + 0.45 * prediction_rgb
    Image.fromarray(overlay.astype(np.uint8)).save(dirs["overlay"] / f"{sample.name}.png")


def score(prediction: np.ndarray, target: np.ndarray, dataset: str,
          *, boundary_wfm: bool) -> tuple[np.ndarray, dict[str, Any]]:
    classes = LOVEDA_CLASSES if dataset == "loveda" else ("unchanged", "change")
    confusion = confusion_matrix(prediction, target, num_classes=len(classes),
                                 ignore_index=255 if dataset == "loveda" else None)
    metrics = metrics_from_confusion(confusion, classes)
    if dataset == "levircd":
        metrics.update(binary_metrics(binary_counts(prediction, target)))
    if boundary_wfm:
        if dataset == "loveda":
            # All-ignored images contribute zero confusion and no boundary score.
            if np.any(target != 255):
                metrics.update(score_semantic_boundary_wfm(prediction, target, num_classes=7))
        else:
            metrics.update(score_binary_boundary_wfm(prediction, target))
    return confusion, metrics


def evaluate_dataset(args: argparse.Namespace, dataset: str, backend: Any,
                     base_identity: dict[str, Any]) -> dict[str, Any]:
    tile_size = args.loveda_tile_size if dataset == "loveda" else 0
    spatial = loveda_spatial_settings(tile_size) if dataset == "loveda" else None
    root_value = args.data_root or (args.loveda_root if dataset == "loveda" else args.levircd_root)
    split = "val" if dataset == "loveda" else args.split
    samples, audit = discover(dataset, normalize_path(root_value), split)
    total = len(samples)
    expected = expected_samples(dataset, split)
    if not args.allow_nonstandard_population and total != expected:
        raise ValueError(f"{dataset}/{split}: found {total}, expected {expected}; "
                         "use --allow_nonstandard_population only for a labelled development subset")
    if len({x.name for x in samples}) != total:
        raise ValueError("Dataset contains colliding output names")
    input_manifest = [{"name": sample.name, "files": input_identity(sample)} for sample in samples]
    protocol = {
        **base_identity, "dataset": dataset, "split": split,
        "dataset_population": fingerprint(input_manifest), "discovered_samples": total,
        "prompt": LOVEDA_PROMPT if dataset == "loveda" else CD_PROMPT,
        "cd_direction": args.cd_direction if dataset == "levircd" else None,
        "nominal_grid_size": GRID_SIZE, "rle_grid_policy": RLE_GRID_POLICY,
        "max_descriptor_side": MAX_DESCRIPTOR_SIDE, "restoration": "nearest_to_native_extent",
        "nearest_implementation": "opencv_INTER_NEAREST",
        "classes": list(LOVEDA_CLASSES) if dataset == "loveda" else ["unchanged", "change"],
        "label_protocol": "raw0/255 ignored; raw1..7 -> 0..6" if dataset == "loveda" else "raw0 unchanged; raw1/255 change",
    }
    if spatial is not None:
        protocol.update(loveda_spatial=spatial, restoration=spatial["restoration"])
    config_id = fingerprint(protocol)
    output = normalize_path(args.output_dir)
    if args.dataset == "both":
        output = output / ("loveda_val" if dataset == "loveda" else f"levircd_{split}")
    dirs = make_dirs(output)
    # Single writer per output directory. The OS releases this advisory lock
    # on a crash; no stale lock/PID deletion is required.
    import fcntl
    with (output / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluation owns {output}") from exc
        existing = read_json(output / "run_config.json")
        has_cache = any(dirs["pred_mask"].glob("*.png")) or any(dirs["raw_text"].glob("*.json"))
        if has_cache and not args.overwrite and (existing is None or existing.get("identity") != config_id):
            raise RuntimeError("Cached outputs use a different model/data/prompt/source protocol; "
                               "choose a new OUTPUT_DIR or explicitly set OVERWRITE=1")
        write_json(output / "run_config.json", {
            "identity": config_id, "protocol": protocol, "audit": audit,
            "metric_settings": {"boundary_wfm": args.boundary_wfm, "wfm_ignore_margin": 2},
        })
        write_json(output / "input_manifest.json", {"samples": input_manifest})
        selected = samples[:args.max_samples] if args.max_samples else samples
        classes = tuple(protocol["classes"])
        confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        rows: list[dict[str, Any]] = []
        result: dict[str, Any] = {
            "model": "RSUniVLM", "dataset": audit["dataset"], "split": split,
            "protocol_version": PROTOCOL_VERSION, "rle_grid_policy": RLE_GRID_POLICY,
            "nominal_grid_size": GRID_SIZE,
            "complete": False, "is_full_benchmark": False, "discovered_samples": total,
            "selected_samples": len(selected), "expected_samples": expected,
            "identity": config_id, "metric_scale": "0..1 (percent fields explicitly suffixed)",
            "official_upstream_benchmark_result": False,
            "protocol_note": "VISTAR adapter; no fine-tuning; upstream training includes LoveDA and LEVIR-MCI; not asserted to be zero-shot",
        }
        if spatial is not None:
            result["loveda_spatial"] = spatial
        write_json(output / "metrics.json", result)
        progress = tqdm(selected, desc=f"RSUniVLM {dataset}/{split}", unit="image")
        for sample in progress:
            started = time.monotonic()
            try:
                images, target = load_sample(sample, dataset)
                item_id = fingerprint([config_id, input_identity(sample)])
                pred_path = dirs["pred_mask"] / f"{sample.name}_pred_mask.png"
                receipt_path = dirs["receipts"] / f"{sample.name}.json"
                saved_receipt = read_json(receipt_path)
                prediction = None if args.overwrite else cached_prediction(
                    pred_path, saved_receipt, identity=item_id,
                    shape=target.shape, num_classes=len(classes),
                )
                reused = prediction is not None
                query_info: dict[str, Any] = {"descriptor_shapes": saved_receipt.get("descriptor_shapes", [])} if reused else {}
                if prediction is None:
                    prediction, query_info = predict_sample(
                        backend, images, sample, dataset, dirs["raw_text"], item_id,
                        cd_direction=args.cd_direction, overwrite=args.overwrite,
                        loveda_tile_size=tile_size,
                    )
                    save_prediction(pred_path, prediction if dataset == "loveda" else prediction * 255)
                    write_json(receipt_path, {"identity": item_id,
                               "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                               "descriptor_shapes": query_info["descriptor_shapes"]})
                if args.save_images:
                    save_images(dirs, sample, images, prediction, target, dataset)
                item_confusion, metrics = score(prediction, target, dataset, boundary_wfm=args.boundary_wfm)
                confusion += item_confusion
                row = {"sample": sample.name, "reused_prediction": reused,
                       "height": target.shape[0], "width": target.shape[1],
                       "seconds": time.monotonic() - started, **query_info, **metrics}
                if dataset == "loveda":
                    row["source_tile_size"] = tile_size
                    row["num_tiles"] = (((images[0].height+tile_size-1)//tile_size)
                                        * ((images[0].width+tile_size-1)//tile_size)) if tile_size else 1
                rows.append(row)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                progress.set_postfix(cached=int(reused), miou=f"{metrics['miou']:.4f}")
            except Exception as exc:
                write_json(dirs["errors"] / f"{sample.name}.json", {
                    "sample": sample.name, "error": str(exc), "traceback": traceback.format_exc(),
                })
                write_json(output / "metrics.json", {**result, "processed_samples": len(rows),
                                                    "failed_sample": sample.name, "error": str(exc)})
                raise
        scores = metrics_from_confusion(confusion, classes)
        if dataset == "levircd":
            tn, fp, fn, tp = (int(x) for x in confusion.reshape(-1))
            scores.update(binary_metrics({"tp": tp, "tn": tn, "fp": fp, "fn": fn}))
        if args.boundary_wfm:
            boundary_rows = [row for row in rows if "wfm_3px_mean_present" in row]
            if boundary_rows:
                scores.update(aggregate_binary_boundary_wfm(boundary_rows))
        full = total == expected and len(rows) == total
        shape_counts = Counter(f"{shape[0]}x{shape[1]}" for row in rows for shape in row.get("descriptor_shapes", []))
        result.update({"complete": True, "is_full_benchmark": full,
                       "processed_samples": len(rows), "scores": scores,
                       "confusion_matrix": confusion.tolist(), "descriptor_shape_counts": dict(shape_counts)})
        atomic_text(output / "predictions.jsonl", "".join(json.dumps(x, allow_nan=False) + "\n" for x in rows))
        csv_rows = []
        for class_id, name in enumerate(classes):
            key = name.replace(" ", "_")
            csv_rows.append({"class_id": class_id, "class": name, "iou": scores[f"iou_{key}"],
                             "f1": scores[f"f1_{key}"], "accuracy": scores[f"acc_{key}"]})
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
        atomic_text(output / "per_class_metrics.csv", buffer.getvalue())
        write_json(output / "metrics.json", result)
        print(json.dumps({"output_dir": str(output), **result}, ensure_ascii=False, indent=2))
        return result


class LazyModel:
    """Read and rescore caches without loading CUDA weights unnecessarily."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.model = args, None

    @property
    def last_generation(self) -> dict[str, Any]:
        return getattr(self.model, "last_generation", {})

    def generate(self, images: list[Image.Image], prompt: str) -> str:
        if self.model is None:
            from baselines.rsunivlm.model import RSUniVLM
            self.model = RSUniVLM(
                normalize_path(self.args.source_root), normalize_path(self.args.model_path),
                device=self.args.device, dtype=self.args.dtype, attention=self.args.attention,
                max_new_tokens=self.args.max_new_tokens, vision_tower=self.args.vision_tower,
            )
        return self.model.generate(images, prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loveda", "levircd", "both"), required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--loveda_root", default="/root/data/LoveDA")
    parser.add_argument("--levircd_root", default="/root/data/LEVIR-CD")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_root", default=str(REPO_ROOT / "third_party" / "RSUniVLM"))
    parser.add_argument("--model_path", default="/root/data/weight/rsunivlm/llava-qwen-gmoe")
    parser.add_argument("--vision_tower", default="", help="Optional local SigLIP path; otherwise retain checkpoint config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--cd_direction", choices=("symmetric_union", "forward"), default="symmetric_union")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--loveda_tile_size", type=int, choices=(0, 512), default=512,
                        help="Native LoveDA source crop size; 512 aligns with Alters, 0 reproduces whole-image processing")
    parser.add_argument("--allow_nonstandard_population", action="store_true")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary_wfm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate paths/populations/source, without loading weights")
    args = parser.parse_args()
    if args.dataset == "both" and args.data_root:
        parser.error("Use --loveda_root and --levircd_root with --dataset both")
    if args.max_samples < 0 or args.max_new_tokens <= 0:
        parser.error("max_samples must be nonnegative; max_new_tokens must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Use one process per output directory; run datasets on separate GPUs if desired")
    return args


def main() -> None:
    args = parse_args()
    source = normalize_path(args.source_root)
    base = {
        "protocol_version": PROTOCOL_VERSION, "source_revision": source_revision(source),
        "checkpoint": checkpoint_identity(normalize_path(args.model_path)),
        "dtype": args.dtype, "attention": args.attention, "max_new_tokens": args.max_new_tokens,
        "vision_tower_override": args.vision_tower,
        "adapter_source": fingerprint([path.read_text() for path in sorted(Path(__file__).parent.glob("*.py"))]),
    }
    datasets = ("loveda", "levircd") if args.dataset == "both" else (args.dataset,)
    if args.preflight:
        for dataset in datasets:
            data_root = args.data_root or (args.loveda_root if dataset == "loveda" else args.levircd_root)
            split = "val" if dataset == "loveda" else args.split
            samples, audit = discover(dataset, normalize_path(data_root), split)
            expected = expected_samples(dataset, split)
            if len(samples) != expected and not args.allow_nonstandard_population:
                raise ValueError(f"{dataset}: {len(samples)} samples != expected {expected}")
            info = {"preflight": "passed (paths only; model not loaded)", **audit, "model": base}
            if dataset == "loveda":
                info["loveda_spatial"] = loveda_spatial_settings(args.loveda_tile_size)
            print(json.dumps(info, indent=2))
        return
    backend = LazyModel(args)
    for dataset in datasets:
        evaluate_dataset(args, dataset, backend, base)


if __name__ == "__main__":
    main()
