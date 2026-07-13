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


def _canonical_split(value: str) -> str:
    split = str(value).strip().lower()
    aliases = {
        "training": "train",
        "validation": "val",
        "valid": "val",
        "eval": "val",
    }
    split = aliases.get(split, split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split {value!r}; use train, val, test, or all")
    return split


def _parse_splits(value: str) -> list[str]:
    text = str(value).strip().lower()
    if not text or text in {"all", "both", "*"}:
        return []
    splits: list[str] = []
    for item in text.replace("+", ",").split(","):
        if not item.strip():
            continue
        split = _canonical_split(item)
        if split not in splits:
            splits.append(split)
    if not splits:
        raise ValueError(f"no valid split selected from {value!r}")
    return splits


def _split_from_name(name: str) -> str:
    """Recover the split naming convention used by Vistar LoveDA generation."""
    lowered = str(name).lower()
    for split in ("train", "val", "test"):
        if lowered.startswith(f"{split}_"):
            return split
    # Vistar intentionally keeps the legacy <domain>_<id> names for Val.
    return "val"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a baseline JSONL manifest from a Vistar generation eval directory."
    )
    parser.add_argument("--eval_dir", required=True, help="Vistar eval directory containing cond_mask, gt_rgb, and optionally prompts")
    parser.add_argument("--output", required=True, help="output JSONL manifest")
    parser.add_argument("--prompt", default="A high-resolution remote-sensing image matching the given semantic mask.")
    parser.add_argument("--prompt_dir", default="", help="optional prompt folder; defaults to eval_dir/prompts")
    parser.add_argument("--overwrite_prompt_from_file", action="store_true")
    parser.add_argument(
        "--splits",
        default="all",
        help="comma-separated Vistar LoveDA splits to include (train,val,test), or all",
    )
    parser.add_argument(
        "--require_splits",
        action="store_true",
        help="fail if any explicitly selected split has no samples",
    )
    args = parser.parse_args()

    eval_dir = Path(_normalize_wsl_unc(args.eval_dir)).expanduser().resolve()
    cond_dir = eval_dir / "cond_mask"
    gt_dir = eval_dir / "gt_rgb"
    prompt_dir = Path(_normalize_wsl_unc(args.prompt_dir)).expanduser().resolve() if args.prompt_dir else eval_dir / "prompts"
    out_path = Path(_normalize_wsl_unc(args.output)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_splits = _parse_splits(args.splits)

    if not cond_dir.is_dir():
        raise NotADirectoryError(f"missing condition folder: {cond_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"missing GT folder: {gt_dir}")

    rows = []
    split_counts: dict[str, int] = {}
    for cond_path in sorted(cond_dir.glob("*_cond_mask.png")):
        name = _name_from_cond(cond_path)
        split = _split_from_name(name)
        if selected_splits and split not in selected_splits:
            continue
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
                "split": split,
                "condition_image": str(cond_path),
                "target_image": str(gt_path),
                "prompt": prompt,
            }
        )
        split_counts[split] = split_counts.get(split, 0) + 1

    if args.require_splits and selected_splits:
        missing_splits = [split for split in selected_splits if split_counts.get(split, 0) == 0]
        if missing_splits:
            raise FileNotFoundError(
                f"Vistar eval directory {eval_dir} has no samples for required split(s): "
                f"{', '.join(missing_splits)}. Found counts: {split_counts}"
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[build_manifest_from_vistar_eval] wrote {len(rows)} rows to {out_path}; "
        f"split_counts={split_counts}"
    )


if __name__ == "__main__":
    main()
