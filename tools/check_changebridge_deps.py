from __future__ import annotations

import importlib
import os
from pathlib import Path


def main() -> int:
    failures = []
    root = Path(os.environ.get("CHANGEBRIDGE_ROOT", Path(__file__).resolve().parents[1] / "third_party/ChangeBridge"))
    for module in ("torch", "torchvision", "yaml", "omegaconf", "cv2", "PIL", "numpy", "tqdm"):
        try:
            importlib.import_module(module)
            print(f"ok               {module}")
        except Exception as exc:
            failures.append(module)
            print(f"MISSING required {module}: {exc}")
    for relative in ("main.py", "configs/Template_LBBDM_f4_cd_semantic.yaml", "datasets/custom_cd.py"):
        if not (root / relative).is_file():
            failures.append(relative)
            print(f"MISSING required {root / relative}")
    if not (root / "runners").is_dir():
        failures.append("upstream runners/")
        print("INCOMPLETE upstream source: runners/ is absent in the public repository; main.py cannot import utils.py.")
    for env_name in ("CHANGEBRIDGE_VQGAN_CKPT", "CHANGEBRIDGE_CLIP_CKPT"):
        path = Path(os.environ.get(env_name, ""))
        if not path.is_file():
            failures.append(env_name)
            print(f"MISSING required {env_name}={path}")
    print("\nChangeBridge dependency check " + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
