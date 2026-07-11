from __future__ import annotations

import argparse
import json

from build_tisynth_loveda_manifest import _prompt_for_mask, _resolve, _sample_name, _validate_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TISynth training/validation JSONL from saved LoveDA mask/RGB pairs."
    )
    parser.add_argument("--source_dir", required=True, help="directory containing cond_mask and gt_rgb")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt_prefix",
        default="a high-resolution remote sensing satellite image",
    )
    parser.add_argument(
        "--strict_palette",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expected_size", type=int, default=512)
    args = parser.parse_args()

    source_dir = _resolve(args.source_dir)
    output = _resolve(args.output)
    cond_dir = source_dir / "cond_mask"
    gt_dir = source_dir / "gt_rgb"
    if not cond_dir.is_dir() or not gt_dir.is_dir():
        raise NotADirectoryError(
            f"TISynth LoveDA source must contain cond_mask and gt_rgb: {source_dir}"
        )

    rows = []
    for cond_path in sorted(cond_dir.glob("*_cond_mask.png")):
        name = _sample_name(cond_path)
        target_path = gt_dir / f"{name}_gt_rgb.png"
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing paired RGB target: {target_path}")
        _validate_size(cond_path, int(args.expected_size))
        _validate_size(target_path, int(args.expected_size))
        rows.append(
            {
                "source": str(cond_path.resolve()),
                "target": str(target_path.resolve()),
                "prompt": _prompt_for_mask(
                    cond_path,
                    str(args.prompt_prefix),
                    bool(args.strict_palette),
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No *_cond_mask.png files found under: {cond_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_tisynth_loveda_train_manifest] wrote {len(rows)} rows to: {output}")


if __name__ == "__main__":
    main()
