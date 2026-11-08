"""Preserve an existing CUDA Torch stack during OVCD dependency setup.

Reads installed distribution metadata only: no torch import or CUDA call.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def installed():
    versions = {}
    for name in ("torch", "torchvision", "torchaudio"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name != "torchaudio":
                raise RuntimeError(f"The cloned environment must already contain {name}")
    return {"prefix": sys.prefix, "versions": versions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("capture", "verify"))
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--constraints", type=Path)
    args = p.parse_args()
    current = installed()
    if args.state.exists():
        expected = json.loads(args.state.read_text())
        if current != expected:
            raise RuntimeError(f"Torch stack/prefix changed; preserve the environment and inspect {args.state}: "
                               f"expected={expected}, actual={current}")
    elif args.operation == "verify":
        raise FileNotFoundError(args.state)
    else:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(current, indent=2) + "\n")
    if args.operation == "capture":
        if args.constraints is None:
            p.error("capture requires --constraints")
        args.constraints.parent.mkdir(parents=True, exist_ok=True)
        args.constraints.write_text("\n".join(f"{name}=={version}" for name, version in current["versions"].items()) + "\n")
    print(json.dumps({"operation": args.operation, **current, "models_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
