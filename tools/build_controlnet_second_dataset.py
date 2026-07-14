from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    os.symlink(source.resolve(), target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an imagefolder dataset for SECOND ControlNet training.")
    parser.add_argument("--manifest", default="/data/vistar/runs/paper_baselines/data/second/train.jsonl")
    parser.add_argument("--output", default="/data/vistar/runs/paper_baselines/controlnet_second_train")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    output = Path(args.output).expanduser().resolve()
    metadata = []
    for row in rows:
        name = row["name"] + ".png"
        link(Path(row["target_image"]), output / "images" / name)
        link(Path(row["target_mask_rgb"]), output / "conditioning" / name)
        metadata.append({
            "file_name": f"images/{name}",
            "conditioning_image_file_name": f"conditioning/{name}",
            "text": row["prompt"],
        })
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")
    print(f"[build_controlnet_second_dataset] wrote {len(metadata)} rows to {output}")


if __name__ == "__main__":
    main()
