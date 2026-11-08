"""Resumable DynamicEarth/OpenDPR: LEVIR test128 and exact SECOND val593.

Keep model inference separate from native-resolution membership IoU and the
same whole-semantic boundary WFm used by the LoveDA/RSUniVLM evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from tqdm import tqdm

from baselines.levircd_common import binary_counts, binary_metrics, load_binary
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from tools import rsunivlm_paper_protocol as data
from baselines.ovcd.backends import REVISIONS, NAMES

SPLIT_SHA = "eed4dc3956022e18f426ba9efd30fd0927cd4fa0c065b7058e86e439624ed346"
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sources(args, dataset):
    root = Path(args.source_root).resolve()
    required = [root / "dynamic_earth/comparator/bi_match.py"]
    required += [root / ("dynamic_earth/identifier/prototype_retrieval.py" if args.method == "opendpr"
                         else "dynamic_earth/identifier/segearth_ov_ext.py")]
    files = {str(path): sha(path) for path in required}
    archive_receipt = root / ".ovcd_upstream_source.json"
    if archive_receipt.is_file():
        files[str(archive_receipt)] = sha(archive_receipt)
    folders = [root / "dynamic_earth", root / "third_party/segment_anything"]
    if args.method == "dynamicearth":
        folders += [root / "third_party/SegEarth_OV", root / "third_party/SimFeatUp"]
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h", ".so"}:
                files[str(path)] = sha(path)
    if args.method == "opendpr":
        support = root / "support_set" / ("SECOND" if dataset == "second" else "LEVIR-CD")
        classes = NAMES if dataset == "second" else ("background", "building")
        for category in classes:
            folder, masks = support / category, Path(str(support) + "_mask") / category
            pairs = [(path, masks / (path.stem + ".png")) for path in sorted(folder.iterdir())
                     if path.suffix.lower() in {".png", ".jpg", ".jpeg"}] if folder.is_dir() else []
            if len(pairs) < 3 or any(not target.is_file() for _, target in pairs):
                raise ValueError(f"Incomplete official OpenDPR support image/mask pairs: {category}")
            for pair in pairs:
                for path in pair:
                    files[str(path)] = sha(path)
        checkpoint_dir = Path(args.dinov2_dir)
        for name in ("config.json", "preprocessor_config.json"):
            files[str(checkpoint_dir / name)] = sha(checkpoint_dir / name)
        weights = [p for name in ("model.safetensors", "pytorch_model.bin")
                   if (p := checkpoint_dir / name).is_file()]
        if not weights:
            raise FileNotFoundError(f"Missing local DINOv2 weights: {checkpoint_dir}")
        files[str(weights[0])] = sha(weights[0])
        config = json.loads((checkpoint_dir / "config.json").read_text())
        if config.get("hidden_size") != 768 or config.get("patch_size") != 14:
            raise ValueError("OpenDPR requires the published DINOv2-B/14 encoder")
    else:
        files[args.dynamic_config] = sha(args.dynamic_config)
        if getattr(args, "clip_cache_dir", None):
            clip = Path(args.clip_cache_dir) / "ViT-B-16.pt"
            files[str(clip)] = sha(clip)
        for name in (args.dino_checkpoint, args.segearth_weight):
            files[name] = sha(name)
        repo = Path(args.dino_repo)
        for name in ("hubconf.py", "vision_transformer.py", "utils.py"):
            files[str(repo / name)] = sha(repo / name)
    if Path(args.sam_checkpoint).stat().st_size < 1_000_000_000:
        raise ValueError("SAM ViT-H checkpoint is missing/truncated or the wrong variant")
    files[args.sam_checkpoint] = sha(args.sam_checkpoint)
    return files


def plan(args, dataset):
    if sha(data.SPLIT_FILE) != SPLIT_SHA:
        raise ValueError("SECOND canonical val593 list differs from Alters")
    samples, population = data.discover(dataset, getattr(args, dataset + "_root"), strict=True)
    selected = samples[:args.max_samples] if args.max_samples else samples
    if any(Path(sample.name).name != sample.name for sample in selected):
        raise ValueError("Unsafe sample identity")
    sealed = [{str(p): sha(p) for p in (*sample.images, *sample.labels)} for sample in selected]
    local = ("baselines/ovcd/evaluate.py", "baselines/ovcd/backends.py",
             "baselines/levircd_common.py", "baselines/binary_boundary_wfm.py", "tools/rsunivlm_paper_protocol.py")
    try:
        if not (Path(args.source_root).resolve() / ".git").exists():
            raise OSError("Archive source has no own Git repository")
        revision = subprocess.run(["git","-C",str(Path(args.source_root).resolve()),"rev-parse","HEAD"],
            check=True,capture_output=True,text=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        receipt = Path(args.source_root).resolve() / ".ovcd_upstream_source.json"
        revision = (json.loads(receipt.read_text())["revision"] if receipt.is_file()
                    else "unavailable; actual source files are content-hashed")
    if revision != REVISIONS[args.method] and not revision.startswith("unavailable"):
        raise ValueError("Upstream revision differs; use the pinned bootstrap or a separate explicitly audited adapter")
    versions = {}
    for name in ("numpy","scipy","Pillow","torch","torchvision","transformers","scikit-learn","mmcv","opencv-python-headless","opencv-python"):
        try:versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name] = "unavailable"
    return selected, dict(schema="ovcd_paper_val593_v1", method=args.method, dataset=dataset,
        source_revision=revision,expected_source_revision=REVISIONS[args.method],upstream_files_sha256=sources(args, dataset),
        population=population, expected_samples=data.spec(dataset)["expected"],
        selected_samples=len(selected), samples=[asdict(sample) for sample in selected], input_sha256=sealed,
        tile_size=args.tile_size, stride=args.tile_size, padding="black right/bottom; native stitching/crop",
        whole_image_official_equivalence="SECOND native512 unchanged; LEVIR tile512 is an explicit adapter",
        prediction="single ordered pair; six independent memberships on SECOND; no reversed inference",
        levir_gt="Vistar PIL grayscale >0, including original intermediate nonzero values",
        second_gt="sparse original label1/label2 class unions, unchanged remains0",
        boundary="whole dense labels; ascending class1..6, last-positive wins for prediction and GT; LoveDA WFm",
        code_sha256={name: sha(ROOT / name) for name in local}, metric_versions=versions)


class RGBDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for path in sample.images:
            with Image.open(path) as opened:
                images.append(np.asarray(opened.convert("RGB"), np.uint8).copy())
        if images[0].shape != images[1].shape:
            raise ValueError(f"Paired image extents differ: {sample.name}")
        return index, images


def passthrough(value):
    return value


def predict(backend, images, tile_size):
    h, w = images[0].shape[:2]
    if tile_size == 0:
        return backend.predict(*images)
    result = None
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles = []
            for image in images:
                tile = np.zeros((tile_size, tile_size, 3), np.uint8)
                patch = image[y:y+tile_size, x:x+tile_size]
                tile[:patch.shape[0], :patch.shape[1]] = patch
                tiles.append(tile)
            current = backend.predict(*tiles)
            if result is None:
                result = [np.zeros((h,w),np.uint8) for _ in current]
            if len(current) != len(result):
                raise ValueError("Backend category count changed between tiles")
            hh, ww = min(tile_size,h-y), min(tile_size,w-x)
            for output, array in zip(result, current):
                if array.shape != (tile_size,tile_size) or not np.all((array==0)|(array==1)):
                    raise ValueError("Backend must return native binary category memberships")
                output[y:y+hh,x:x+ww] = array[:hh,:ww]
    return result


def dense(memberships):
    labels = np.zeros(memberships[0].shape, np.uint8)
    for index, mask in enumerate(memberships, 1):
        labels[np.asarray(mask,dtype=bool)] = index
    return labels


def score_masks(dataset, sample, masks):
    shape = masks[0].shape
    if dataset == "second":
        targets = [data.load_second_label(p) for p in sample.labels]
        if any(target.shape != shape for target in targets) or len(masks) != 6:
            raise ValueError("SECOND requires six memberships at original label resolution")
        memberships = [(targets[0]==c)|(targets[1]==c) for c in range(1,7)]
        counts = {str(c):binary_counts(masks[c-1],memberships[c-1]) for c in range(1,7)}
    else:
        target = load_binary(Path(sample.labels[0]))
        if target.shape != shape or len(masks) != 1:
            raise ValueError("LEVIR binary prediction/GT extent differs")
        memberships = [target]
        counts = {"1":binary_counts(masks[0],target)}
    boundary = score_semantic_boundary_wfm(dense(masks),dense(memberships),
        num_classes=7 if dataset=="second" else 2,ignore_index=255,ignore_margin=2)
    return dict(name=sample.name,class_counts=counts,**boundary)


def evaluate(args, dataset, samples, protocol):
    root = Path(args.output_dir).resolve() / dataset
    identity = fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    with (root/".eval.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = root/"protocol.json"
        record = dict(identity=identity,protocol=protocol)
        if config.exists() and json.loads(config.read_text()) != record:
            raise ValueError(f"Inference/source/population changed; choose a new output root: {root}")
        write(config,record)
        import torch
        from torch.utils.data import DataLoader
        from baselines.ovcd.backends import OpenDPR, DynamicEarth
        loader = DataLoader(RGBDataset(samples),batch_size=None,num_workers=args.num_workers,
            collate_fn=passthrough,**(dict(prefetch_factor=1,multiprocessing_context="spawn") if args.num_workers else {}))
        rows, failures, backend, cached = [], [], None, 0
        for index, images in tqdm(loader,total=len(samples),desc=f"{args.method}/{dataset}",unit="pair"):
            sample = samples[index]
            before = protocol["input_sha256"][index]
            if any(sha(p)!=value for p,value in before.items()):
                raise ValueError(f"Evaluation input changed: {sample.name}")
            receipt = root/"rows"/(sample.name+".json")
            old = json.loads(receipt.read_text()) if receipt.is_file() else {}
            masks, reused = [], old.get("identity")==identity
            names = [f"pred_mask/{sample.name}_class_{c:02d}.png" for c in range(1,7 if dataset=="second" else 2)]
            for name in names:
                if not reused or not (root/name).is_file() or sha(root/name)!=old.get("sha256",{}).get(name):
                    reused=False
                    break
                with Image.open(root/name) as opened:
                    mask=np.asarray(opened).copy()
                if mask.shape!=images[0].shape[:2] or not np.all((mask==0)|(mask==1)):
                    raise ValueError(f"Invalid saved prediction: {name}")
                masks.append(mask)
            try:
                if not reused:
                    if backend is None:
                        if not torch.cuda.is_available() and args.device.startswith("cuda"):
                            raise RuntimeError("CUDA unavailable; no model inference was started")
                        torch.manual_seed(0); np.random.seed(0)
                        backend=(OpenDPR if args.method=="opendpr" else DynamicEarth)(args,dataset,root)
                    with torch.inference_mode():
                        masks=predict(backend,images,args.tile_size)
                    if len(masks)!=(6 if dataset=="second" else 1):
                        raise ValueError("Missing semantic category outputs")
                    checksums={}
                    for name,array in zip(names,masks):
                        path=root/name;path.parent.mkdir(exist_ok=True)
                        temporary=path.with_name(path.stem+".tmp.png")
                        Image.fromarray(array.astype(np.uint8)).save(temporary)
                        os.replace(temporary,path);checksums[name]=sha(path)
                    write(receipt,dict(identity=identity,sha256=checksums))
                row=score_masks(dataset,sample,masks)
                if any(sha(p)!=value for p,value in before.items()):
                    raise ValueError("Source changed while scoring")
                rows.append(row);cached+=int(reused)
                write(root/"scores"/(sample.name+".json"),row)
            except Exception as error:
                failures.append(dict(sample=sample.name,error=repr(error)))
                write(root/"failures.json",failures)
                # Runtime/model errors need a fresh process, never a fabricated mask.
                if isinstance(error,RuntimeError):
                    raise
        if any(sha(p)!=value for values in protocol["input_sha256"] for p,value in values.items()):
            raise ValueError("A source input changed before final aggregation")
        if any(sha(p)!=value for p,value in protocol["upstream_files_sha256"].items()):
            raise ValueError("Upstream source/weights/support set changed while evaluating")
        classes=range(1,7) if dataset=="second" else (1,)
        totals={str(c):{k:sum(row["class_counts"][str(c)][k] for row in rows) for k in ("tp","tn","fp","fn")} for c in classes}
        scores={}
        if rows:
            values=[binary_metrics(counts) for counts in totals.values()]
            scores=dict(miou_c=float(np.mean([v["iou_change"] for v in values])),
                        mf1_c=float(np.mean([v["f1_score_change"] for v in values])),
                        **aggregate_binary_boundary_wfm(rows))
            if dataset=="levircd":scores["building_iou"]=scores["miou_c"]
        result=dict(method=args.method,dataset=dataset,identity=identity,complete=len(rows)==len(samples) and not failures,
            is_full_benchmark=len(samples)==protocol["expected_samples"],processed_samples=len(rows),selected_samples=len(samples),
            num_failed=len(failures),failures=failures,cached=cached,scores=scores,class_counts=totals)
        write(root/"metrics.json",result)
        print(json.dumps(result,indent=2),flush=True)
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=("dynamicearth","opendpr"),required=True)
    p.add_argument("--datasets",nargs="+",choices=("levircd","second"),default=["levircd","second"])
    p.add_argument("--source_root")
    p.add_argument("--levircd_root",default="/root/data/LEVIR-CD")
    p.add_argument("--second_root",default="/root/data/second_dataset")
    p.add_argument("--sam_checkpoint",default="/root/data/weight/dynamicearth/sam_vit_h_4b8939.pth")
    p.add_argument("--dinov2_dir",default="/root/data/weight/dinov2-base")
    p.add_argument("--dino_repo",default=str(ROOT/"third_party/DINO"))
    p.add_argument("--dino_checkpoint",default="/root/data/weight/dynamicearth/dino_vitbase16_pretrain.pth")
    p.add_argument("--segearth_weight",default="/root/data/weight/dynamicearth/xclip_jbu_one_million_aid.ckpt")
    p.add_argument("--clip_cache_dir",help="Existing official CLIP cache; same OpenAI ViT-B/16 weights")
    p.add_argument("--dynamic_config",default=str(ROOT/"configs/ovcd/dynamicearth_second_mci.json"))
    p.add_argument("--output_dir",required=True)
    p.add_argument("--tile_size",type=int,choices=(0,512),default=512)
    p.add_argument("--max_samples",type=int,default=0)
    p.add_argument("--num_workers",type=int,default=2)
    p.add_argument("--device",default="cuda")
    p.add_argument("--plan_only",action="store_true")
    return p


def main():
    args=parser().parse_args()
    if args.num_workers<0 or args.max_samples<0 or len(set(args.datasets))!=len(args.datasets):
        raise ValueError("Require nonnegative counts and unique datasets")
    args.source_root=args.source_root or str(ROOT/"third_party"/("OpenDPR" if args.method=="opendpr" else "DynamicEarth"))
    prepared=[(d,*plan(args,d)) for d in args.datasets]
    if args.plan_only:
        for dataset,samples,protocol in prepared:
            print(json.dumps(dict(method=args.method,dataset=dataset,selected=len(samples),expected=protocol["expected_samples"],
                                 population=protocol["population"],identity=fingerprint(protocol),model_loaded=False)),flush=True)
        return 0
    results={d:evaluate(args,d,samples,protocol) for d,samples,protocol in prepared}
    complete=all(r["complete"] and r["is_full_benchmark"] for r in results.values())
    write(Path(args.output_dir)/"paper_summary.json",dict(complete=complete,results=results))
    return 0 if all(r["complete"] for r in results.values()) else 2


if __name__=="__main__":
    raise SystemExit(main())
