from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SECOND_PALETTE = np.asarray([
    [255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0],
    [0, 255, 0], [128, 0, 0], [255, 255, 0],
], dtype=np.uint8)
DIRS = ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts")


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def prediction_for(root: Path, name: str) -> Path:
    exact = [root / f"{name}.png", root / f"{name}_pred_rgb.png"]
    for path in exact:
        if path.is_file():
            return path
    matches = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} and name in path.stem)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one ChangeBridge prediction for {name} under {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize ChangeBridge samples to the common SECOND output contract.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_size", type=int, default=256)
    args = parser.parse_args()

    rows = [json.loads(line) for line in resolve(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    pred_root, output = resolve(args.prediction_dir), resolve(args.output_dir)
    folders = {name: output / name for name in DIRS}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    metadata = []
    for row in rows:
        name = str(row["name"])
        source = Image.open(resolve(row["source_image"])).convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
        target = Image.open(resolve(row["target_image"])).convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
        ids = Image.open(resolve(row["condition_mask_ids"])).convert("L").resize((args.eval_size, args.eval_size), Image.Resampling.NEAREST)
        ids_np = np.asarray(ids, dtype=np.uint8)
        pred_source = prediction_for(pred_root, name)
        pred = Image.open(pred_source).convert("RGB").resize((args.eval_size, args.eval_size), Image.Resampling.BICUBIC)
        source.save(folders["source_rgb"] / f"{name}_source_rgb.png")
        target.save(folders["gt_rgb"] / f"{name}_gt_rgb.png")
        ids.save(folders["cond_mask_ids"] / f"{name}_cond_mask_ids.png")
        Image.fromarray(SECOND_PALETTE[ids_np.clip(0, 6)], "RGB").save(folders["cond_mask_official"] / f"{name}_cond_mask_official.png")
        Image.fromarray(np.repeat(np.where(ids_np[..., None] > 0, 255, 0).astype(np.uint8), 3, axis=2), "RGB").save(folders["cond_mask"] / f"{name}_cond_mask.png")
        pred.save(folders["pred_rgb"] / f"{name}_pred_rgb.png")
        Image.fromarray(np.abs(np.asarray(pred, np.int16) - np.asarray(target, np.int16)).astype(np.uint8), "RGB").save(folders["absdiff"] / f"{name}_absdiff.png")
        prompt = "ChangeBridge semantic-map conditioned SECOND synthesis."
        (folders["prompts"] / f"{name}.txt").write_text(prompt + "\n", encoding="utf-8")
        metadata.append({
            "name": name, "dataset": "SECOND", "direction": row.get("direction"),
            "condition_passed_to_model": ["source_rgb", "semantic_change_mask"],
            "ground_truth_change_mask_passed_to_model": True,
            "prediction_source": str(pred_source), "model_outputs": ["pred_rgb"],
        })
    with (output / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[collect_changebridge_outputs] wrote {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
