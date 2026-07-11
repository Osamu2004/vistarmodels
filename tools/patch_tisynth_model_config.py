from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from build_tisynth_loveda_manifest import _resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Set TISynth FrozenCLIPEmbedder path without editing upstream source.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip_version", required=True)
    args = parser.parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    config = OmegaConf.load(str(input_path))
    cond_stage = config.model.params.cond_stage_config
    if "params" not in cond_stage or cond_stage.params is None:
        cond_stage.params = {}
    cond_stage.params.version = str(args.clip_version)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(output_path))
    print(f"[patch_tisynth_model_config] wrote: {output_path}")


if __name__ == "__main__":
    main()
