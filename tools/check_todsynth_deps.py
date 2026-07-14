from __future__ import annotations

import importlib
import os
from pathlib import Path


def main() -> int:
    failures = []
    root = Path(os.environ.get("TODSYNTH_ROOT", Path(__file__).resolve().parents[1] / "third_party/crfm"))
    for module in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "PIL", "numpy", "tqdm", "peft"):
        try:
            importlib.import_module(module)
            print(f"ok               {module}")
        except Exception as exc:
            failures.append(module)
            print(f"MISSING required {module}: {exc}")
    for relative in ("train.py", "preprocess/vectorize.py", "src/models/sd3_mmdit.py", "src/utils/inference.py", "src/utils/crfm.py"):
        if not (root / relative).is_file():
            failures.append(relative)
            print(f"MISSING required {root / relative}")
    model = Path(os.environ.get("SD35_MODEL_DIR", "/data/vistar/weights/todsynth/sd3.5-medium"))
    for relative in ("transformer/config.json", "vae/config.json", "scheduler/scheduler_config.json"):
        if not (model / relative).is_file():
            failures.append(relative)
            print(f"MISSING gated SD3.5 file {model / relative}")
    checkpoint = os.environ.get("TODSYNTH_CHECKPOINT", "")
    if checkpoint and not (Path(checkpoint) / "model.safetensors").is_file() and not Path(checkpoint).is_file():
        failures.append("TODSYNTH_CHECKPOINT")
        print(f"MISSING trained checkpoint {checkpoint}")
    if not checkpoint:
        print("TRAINING REQUIRED no TODSYNTH_CHECKPOINT set; no official task checkpoint is published")
    print("\nTODSynth dependency check " + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
