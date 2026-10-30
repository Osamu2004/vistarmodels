"""Check RSUniVLM imports/checkpoint metadata without loading model weights."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from baselines.rsunivlm.protocols import checkpoint_identity, source_revision

    errors = []
    source = Path(os.environ.get("RSUNIVLM_ROOT", str(ROOT / "third_party" / "RSUniVLM"))).expanduser().resolve()
    checkpoint = Path(os.environ.get("RSUNIVLM_MODEL_PATH", "/root/data/weight/rsunivlm/llava-qwen-gmoe")).expanduser().resolve()
    for name in ("torch", "torchvision", "transformers", "accelerate", "numpy", "PIL", "scipy",
                 "tqdm", "einops", "timm", "sentencepiece", "decord", "av", "cv2"):
        try:
            module = importlib.import_module(name)
            print(f"ok {name}: {getattr(module, '__version__', 'installed')}")
            if name == "transformers" and module.__version__ != "4.45.2":
                errors.append("Expected transformers==4.45.2 for pinned GMoE internals")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    try:
        print("ok upstream revision:", source_revision(source))
        sys.path.insert(0, str(source))
        import llava
        if not Path(llava.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"llava resolves to another package: {llava.__file__}")
        from llava.model.builder import load_pretrained_model  # noqa: F401
        from llava.mm_utils import process_images  # noqa: F401
        print("ok official GMoE imports (no weights loaded)")
    except Exception as exc:
        errors.append(f"upstream source/imports: {exc}")
    try:
        identity = checkpoint_identity(checkpoint)
        print("ok checkpoint metadata:", json.dumps({"path": identity["path"], "vision_tower": identity["vision_tower"]}))
        tower = os.environ.get("RSUNIVLM_VISION_TOWER", "")
        if tower and not (Path(tower) / "config.json").is_file():
            raise FileNotFoundError(f"Local SigLIP config missing: {tower}")
    except Exception as exc:
        errors.append(f"checkpoint: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA unavailable")
        else:
            print("ok CUDA visible:", torch.cuda.get_device_name(0))
    except Exception as exc:
        errors.append(f"CUDA visibility: {exc}")
    for error in errors:
        print("FAILED", error, file=sys.stderr)
    if errors:
        return 1
    print("Dependency/metadata checks passed. No model forward or checkpoint loading was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
