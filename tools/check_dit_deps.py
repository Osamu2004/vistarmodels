from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


PINNED_DIT_REVISION = "ed81ce2229091fd4ecc9a223645f95cf379d582b"


def resolve(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the conditioned DiT-B/2 SECOND training dependencies.")
    parser.add_argument("--dit_root", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--vae", default="")
    parser.add_argument("--allow_unpinned_dit", action="store_true")
    args = parser.parse_args()

    missing_modules = []
    versions = {}
    for module_name in ("torch", "torchvision", "diffusers", "timm", "numpy", "PIL", "tqdm"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # import failures can include binary ABI errors
            missing_modules.append(f"{module_name}: {error}")
    if missing_modules:
        raise RuntimeError("missing/broken dependencies:\n  " + "\n  ".join(missing_modules))

    dit_root = resolve(args.dit_root)
    for relative in ("models.py", "diffusion"):
        if not (dit_root / relative).exists():
            raise FileNotFoundError(f"missing official DiT component: {dit_root / relative}")
    if (dit_root / ".git").is_dir():
        revision = subprocess.check_output(
            ["git", "-C", str(dit_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != PINNED_DIT_REVISION and not args.allow_unpinned_dit:
            raise RuntimeError(
                f"DiT revision mismatch: expected {PINNED_DIT_REVISION}, found {revision}. "
                "Pass --allow_unpinned_dit only after auditing the source."
            )
    else:
        revision = "not-a-git-checkout"

    if args.manifest:
        manifest = resolve(args.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"manifest is empty: {manifest}")
        required = {"name", "source_image", "target_image", "target_mask_ids", "target_mask_rgb"}
        missing = required - rows[0].keys()
        if missing:
            raise ValueError(f"manifest first row is missing fields: {sorted(missing)}")
    else:
        rows = []

    if args.vae:
        vae = resolve(args.vae)
        config_candidates = (vae / "vae" / "config.json", vae / "config.json")
        if not any(path.is_file() for path in config_candidates):
            raise FileNotFoundError(f"cannot find VAE config under {vae}")

    print(
        json.dumps(
            {
                "status": "ok",
                "python": sys.version.split()[0],
                "versions": versions,
                "dit_root": str(dit_root),
                "dit_revision": revision,
                "manifest_rows": len(rows),
                "vae": str(resolve(args.vae)) if args.vae else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
