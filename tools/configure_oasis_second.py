from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oasis_root", required=True)
    args = parser.parse_args()
    root = Path(args.oasis_root).expanduser().resolve()
    source = Path(__file__).resolve().parents[1] / "baselines/semantic_synthesis/oasis_second_dataset.py"
    target = root / "dataloaders/SecondDataset.py"
    shutil.copy2(source, target)
    loader = root / "dataloaders/dataloaders.py"
    text = loader.read_text(encoding="utf-8")
    needle = '    if mode == "ade20k":\n'
    replacement = '    if mode == "second":\n        return "SecondDataset"\n' + needle
    if 'mode == "second"' not in text:
        if needle not in text:
            raise RuntimeError("Unexpected OASIS dataloader dispatcher")
        loader.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    train = root / "train.py"
    train_text = train.read_text(encoding="utf-8")
    if "OASIS_SKIP_FID" not in train_text:
        train_text = train_text.replace("import torch\n", "import os\nimport torch\n", 1)
        train_text = train_text.replace(
            "fid_computer = fid_pytorch(opt, dataloader_val)",
            'fid_computer = None if os.environ.get("OASIS_SKIP_FID", "0") == "1" else fid_pytorch(opt, dataloader_val)',
        )
        train_text = train_text.replace(
            "if cur_iter % opt.freq_fid == 0 and cur_iter > 0:",
            "if fid_computer is not None and cur_iter % opt.freq_fid == 0 and cur_iter > 0:",
        )
        train_text = train_text.replace(
            "is_best = fid_computer.update(model, cur_iter)\nif is_best:\n    utils.save_networks(opt, cur_iter, model, best=True)",
            "if fid_computer is not None:\n    is_best = fid_computer.update(model, cur_iter)\n    if is_best:\n        utils.save_networks(opt, cur_iter, model, best=True)",
        )
        train.write_text(train_text, encoding="utf-8")
    print(f"[configure_oasis_second] configured {root}")


if __name__ == "__main__":
    main()
