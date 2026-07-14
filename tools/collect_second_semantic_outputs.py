from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SPADE/OASIS outputs into the Table 4 folder contract.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    for folder in ("source_rgb", "cond_mask", "cond_mask_official", "cond_mask_ids", "gt_rgb", "pred_rgb", "absdiff", "prompts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        name = row["name"]
        candidates = [pred_dir / f"{name}.png", pred_dir / f"{name}_synthesized_image.png"]
        pred_source = next((path for path in candidates if path.is_file()), None)
        if pred_source is None:
            raise FileNotFoundError(f"No prediction for {name} under {pred_dir}")
        source = Path(row["source_image"])
        target = Path(row["target_image"])
        mask_ids = Path(row["target_mask_ids"])
        mask_rgb = Path(row["target_mask_rgb"])
        shutil.copy2(source, output / "source_rgb" / f"{name}_source_rgb.png")
        shutil.copy2(target, output / "gt_rgb" / f"{name}_gt_rgb.png")
        shutil.copy2(mask_ids, output / "cond_mask_ids" / f"{name}_cond_mask_ids.png")
        shutil.copy2(mask_rgb, output / "cond_mask_official" / f"{name}_cond_mask_official.png")
        binary = (np.asarray(Image.open(mask_ids).convert("L")) > 0).astype(np.uint8) * 255
        Image.fromarray(np.repeat(binary[..., None], 3, axis=2), mode="RGB").save(
            output / "cond_mask" / f"{name}_cond_mask.png"
        )
        pred = Image.open(pred_source).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        pred_path = output / "pred_rgb" / f"{name}_pred_rgb.png"
        pred.save(pred_path)
        gt = np.asarray(Image.open(target).convert("RGB"), dtype=np.int16)
        pa = np.asarray(pred, dtype=np.int16)
        Image.fromarray(np.abs(gt - pa).astype(np.uint8), mode="RGB").save(output / "absdiff" / f"{name}_absdiff.png")
        (output / "prompts" / f"{name}.txt").write_text(str(row["prompt"]) + "\n")
        written += 1
    print(f"[collect_second_semantic_outputs] {args.method}: {written}")


if __name__ == "__main__":
    main()
