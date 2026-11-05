"""Explicit WSL-side downloads of unchanged official OVCD feature weights."""
import argparse
import os
from pathlib import Path
import subprocess


def download(url, path, minimum_size):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        if path.stat().st_size<minimum_size:
            raise ValueError(f"Existing checkpoint looks truncated; preserve and inspect: {path}")
        print(f"Reuse local checkpoint: {path}",flush=True)
        return
    partial=path.with_suffix(path.suffix+".part")
    subprocess.run(["curl","--fail","-L","--retry","5","--continue-at","-","--output",str(partial),url],check=True)
    if partial.stat().st_size<minimum_size:
        raise ValueError(f"Downloaded checkpoint unexpectedly small: {partial}")
    os.replace(partial,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=("opendpr","dynamicearth"),required=True)
    args=p.parse_args()
    download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        os.environ.get("SAM_CHECKPOINT","/root/data/weight/dynamicearth/sam_vit_h_4b8939.pth"),1_000_000_000)
    if args.method=="opendpr":
        from huggingface_hub import snapshot_download
        local=Path(os.environ.get("DINOV2_DIR","/root/data/weight/dinov2-base"))
        metadata=all((local/name).is_file() for name in ("config.json","preprocessor_config.json"))
        weights=any((local/name).is_file() and (local/name).stat().st_size>100_000_000
                    for name in ("model.safetensors","pytorch_model.bin"))
        if metadata and weights:
            print(f"Reuse local DINOv2, without a remote metadata request: {local}",flush=True)
        else:
            snapshot_download("facebook/dinov2-base",local_dir=str(local),
                allow_patterns=["config.json","preprocessor_config.json","model.safetensors"],max_workers=2)
    else:
        download("https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            os.environ.get("DINO_CHECKPOINT","/root/data/weight/dynamicearth/dino_vitbase16_pretrain.pth"),100_000_000)


if __name__=="__main__":main()
