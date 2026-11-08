"""WSL import/extension check; intentionally separate from real model forwards."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=("opendpr","dynamicearth"),required=True)
    p.add_argument("--source_root")
    p.add_argument("--cuda",action="store_true")
    args=p.parse_args()
    from baselines.ovcd.backends import configure_source
    root=args.source_root or str(ROOT/"third_party"/("OpenDPR" if args.method=="opendpr" else "DynamicEarth"))
    configure_source(root)
    names=["torch","torchvision","transformers","scipy","sklearn","cv2","segment_anything"]
    names+=["dynamic_earth.identifier.prototype_retrieval","dynamic_earth.mask_generator"] if args.method=="opendpr" else [
        "mmcv._ext","adaptive_conv_cuda_impl","dynamic_earth.identifier.segearth_ov_ext","dynamic_earth.sam_ext"]
    failed=[]
    for name in names:
        print("Importing "+name,flush=True)
        try:
            module=importlib.import_module(name)
            print("OK",name,getattr(module,"__version__",""),flush=True)
        except Exception as error:
            failed.append(name);print("FAILED",name,repr(error),flush=True)
    if args.cuda and not failed:
        import torch
        print("Checking CUDA initialization",flush=True)
        if not torch.cuda.is_available():
            failed.append("CUDA unavailable")
        else:
            print(torch.cuda.get_device_name(0),flush=True)
            print("CUDA runtime:",torch.version.cuda,"Torch:",torch.__version__,flush=True)
    print("Import checks only; no checkpoint loaded or model forward executed.",flush=True)
    return 2 if failed else 0


if __name__=="__main__":raise SystemExit(main())
