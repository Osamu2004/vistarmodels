from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize_wsl_unc(path: str) -> str:
    text = str(path)
    for prefix in ("\\\\wsl.localhost\\", "\\wsl.localhost\\"):
        if text.startswith(prefix):
            parts = [p for p in text.strip("\\").split("\\") if p]
            if len(parts) >= 3:
                return "/" + "/".join(parts[2:])
    return text


def _name_from_cond(path: Path) -> str:
    suffix = "_cond_mask.png"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CRS-Diff JSONL manifest from a Vistar generation eval directory."
    )
    parser.add_argument("--eval_dir", required=True, help="Vistar eval directory containing cond_mask, gt_rgb, and optionally prompts")
    parser.add_argument("--output", required=True, help="output JSONL manifest")
    parser.add_argument("--prompt", default="A high-resolution remote-sensing image matching the given semantic mask.")
    parser.add_argument("--prompt_dir", default="", help="optional prompt folder; defaults to eval_dir/prompts")
    parser.add_argument("--overwrite_prompt_from_file", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(_normalize_wsl_unc(args.eval_dir)).expanduser().resolve()
    cond_dir = eval_dir / "cond_mask"
    gt_dir = eval_dir / "gt_rgb"
    prompt_dir = Path(_normalize_wsl_unc(args.prompt_dir)).expanduser().resolve() if args.prompt_dir else eval_dir / "prompts"
    out_path = Path(_normalize_wsl_unc(args.output)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cond_dir.is_dir():
        raise NotADirectoryError(f"missing condition folder: {cond_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"missing GT folder: {gt_dir}")

    rows = []
    for cond_path in sorted(cond_dir.glob("*_cond_mask.png")):
        name = _name_from_cond(cond_path)
        gt_path = gt_dir / f"{name}_gt_rgb.png"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing GT image for {name}: {gt_path}")
        prompt = args.prompt
        prompt_path = prompt_dir / f"{name}.txt"
        if args.overwrite_prompt_from_file and prompt_path.is_file():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "name": name,
                "condition_image": str(cond_path),
                "target_image": str(gt_path),
                "prompt": prompt,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[build_manifest_from_vistar_eval] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

