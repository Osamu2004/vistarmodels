from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    data_root = Path(os.environ.get("PAPER_BASELINE_DATA_ROOT", "/data/vistar/runs/paper_baselines/data")).expanduser()
    weight_root = Path(os.environ.get("PAPER_BASELINE_WEIGHT_ROOT", "/data/vistar/weights")).expanduser()
    eval_root = Path(os.environ.get("PAPER_BASELINE_EVAL_ROOT", "/data/vistar/runs")).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", str(weight_root / "hf_cache"))).expanduser()
    checks = {
        "common_data_summary": (data_root / "summary.json").is_file(),
        "syntheticgen_source": (ROOT / "third_party/SyntheticGen/src/scripts/sample_pair.py").is_file(),
        "syntheticgen_layout": (weight_root / "syntheticgen/layout/checkpoint-79000/model.safetensors").is_file(),
        "syntheticgen_controlnet": (weight_root / "syntheticgen/controlnet/checkpoint-112000/model.safetensors").is_file(),
        "rsedit_weights": (weight_root / "rsedit/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14/unet/diffusion_pytorch_model.safetensors").is_file(),
        "rsedit_inputs": (eval_root / "second_test_bidirectional_inputs_256/manifest_from_eval.jsonl").is_file(),
        "spade_source": (ROOT / "third_party/SPADE/train.py").is_file(),
        "oasis_source": (ROOT / "third_party/OASIS/train.py").is_file(),
        "dit_source": (ROOT / "third_party/DiT/models.py").is_file(),
        "changen_public_repo": (ROOT / "third_party/Changen/hubconf.py").is_file(),
        "diffusers_training_examples": (ROOT / "third_party/diffusers/examples/controlnet/train_controlnet.py").is_file(),
        "sd15_local_snapshot": any(hf_home.glob("models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*/unet/diffusion_pytorch_model.safetensors")),
    }
    print(json.dumps({"checks": checks, "all_local_prerequisites": all(checks.values())}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
