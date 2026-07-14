from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the official ChangeBridge semantic YAML for Vistar SECOND.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--vqgan_ckpt", required=True)
    parser.add_argument("--clip_ckpt", required=True)
    parser.add_argument("--model_ckpt", default="")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    args = parser.parse_args()

    with Path(args.template).open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["data"]["dataset_name"] = "VistarSECOND"
    config["data"]["dataset_type"] = "change_detection_semantic"
    config["data"]["dataset_config"]["condition_type"] = "semantic map"
    config["data"]["dataset_config"]["dataset_path"] = str(Path(args.dataset_path).expanduser().resolve())
    config["data"]["train"]["batch_size"] = args.train_batch_size
    config["data"]["val"]["batch_size"] = args.val_batch_size
    config["data"]["test"]["batch_size"] = args.test_batch_size
    config["model"]["VQGAN"]["params"]["ckpt_path"] = str(Path(args.vqgan_ckpt).expanduser().resolve())
    config["model"]["CLIP"]["ckpt_path"] = str(Path(args.clip_ckpt).expanduser().resolve())
    if args.model_ckpt:
        config["model"]["model_load_path"] = str(Path(args.model_ckpt).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False)
    print(f"[configure_changebridge] wrote {output}")


if __name__ == "__main__":
    main()
