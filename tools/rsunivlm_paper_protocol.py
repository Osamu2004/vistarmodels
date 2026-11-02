"""Dataset contracts for the seven dense-prediction columns in the CVPR draft."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image

from baselines.segearth_ov import protocols as seg
from baselines.rsunivlm import protocols as original

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("loveda","flair","uavid","xbd_pre","chn6_cug","levircd","second")
DEFAULT_ROOTS = dict(loveda="/root/data/LoveDA",flair="/root/data/FLAIR-1-2/data/flair#1-test",
    uavid="/root/data/OVSISBenchDataset/uavid",xbd_pre="/root/data/xview2/test",
    chn6_cug="/root/data/CHN6-CUG/val",levircd="/root/data/LEVIR-CD",second="/root/data/second_dataset")
SECOND_CLASSES = ("unchanged","water","bare land","low vegetation","tree","building","playground")
# The zero class belongs to the source annotations. Model queries retain six
# independent change categories, each with its own binary foreground mask.
SECOND_QUERIES = (("water",1,"water"),("bare_land",2,"bare land"),
    ("low_vegetation",3,"low vegetation"),("tree",4,"tree"),
    ("building",5,"building"),("playground",6,"playground"))
SECOND_RGB = {(255,255,255):0,(0,0,0):0,(0,0,255):1,(128,128,128):2,(159,129,183):2,
              (192,192,192):2,(0,128,0):3,(0,255,0):4,(128,0,0):5,(255,0,0):6,(165,0,165):6}
SPLIT_FILE = ROOT / "assets/splits/second_bisrnet_dynamic_earth_val.txt"
FLAIR_BACKGROUND_ID = 12  # Prediction-only class; GT remains the same twelve classes.


def background_policy(dataset):
    if dataset != "flair": return None
    return dict(policy="others_as_prediction_background_v1",id=FLAIR_BACKGROUND_ID,name="background",
        aliases=["others","other","background"],include_in_class_mean=False,
        scoring="background predictions on valid GT remain false negatives; only original GT ignore is excluded")


def prediction_class_count(dataset):
    if dataset == "second": return 2
    return len(spec(dataset)["classes"]) + int(dataset == "flair")


def empty_confusion(dataset):
    n=len(spec(dataset)["classes"])
    return np.zeros((n,n+int(dataset=="flair")),np.int64)


@dataclass(frozen=True)
class Sample:
    name: str
    images: tuple[str,...]
    labels: tuple[str,...]
    mask_id_base: str = ""

    def serialize(self):
        return asdict(self)

    @classmethod
    def restore(cls, value):
        return cls(value["name"],tuple(value["images"]),tuple(value["labels"]),value.get("mask_id_base",""))


def spec(dataset):
    if dataset in seg.DATASET_SPECS:
        value=seg.DATASET_SPECS[dataset]
        result=dict(classes=list(value["classes"]),expected=value["expected_samples"],
            split=value["split"],primary=value["primary_metric"],kind="semantic",label_protocol=value["label_protocol"])
        if dataset=="flair": result["prediction_background"]=background_policy(dataset)
        return result
    if dataset=="levircd":
        return dict(classes=["unchanged","building"],expected=128,split="test",primary="building_iou",
                    kind="binary_change",label_protocol="0 unchanged; raw1/255 building change; ordered A/B forward query only")
    if dataset=="second":
        return dict(classes=list(SECOND_CLASSES[1:]),expected=593,split="DynamicEarth/Bi-SRNet val593",
            primary="miou_c",kind="classwise_undirected_semantic_change",
            label_protocol="six independent change classes; GT class membership in either annotation; one input order, no reverse prediction")
    raise ValueError(dataset)


def metadata(sample):
    return [dict(path=str(Path(path).resolve()),size=Path(path).stat().st_size,mtime_ns=Path(path).stat().st_mtime_ns)
            for path in (*sample.images,*sample.labels)]


def _index(folder):
    out={}
    for path in sorted(folder.iterdir()):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in seg.IMAGE_EXTENSIONS:
            key=path.stem.casefold()
            if key in out:
                raise ValueError(f"Duplicate sample stem under {folder}: {key}")
            out[key]=path.resolve()
    return out


def discover_second(root,split_file=SPLIT_FILE,strict=True):
    names=[Path(line.split("#",1)[0].strip()).stem for line in Path(split_file).read_text().splitlines()
           if line.split("#",1)[0].strip()]
    if len({name.casefold() for name in names})!=len(names) or (strict and len(names)!=593):
        raise ValueError("SECOND requires the unique canonical 593-pair split")
    groups=(("im1","img1","Image1","image1","images1","T1","t1","A","images/T1","images/t1","images/A"),
            ("im2","img2","Image2","image2","images2","T2","t2","B","images/T2","images/t2","images/B"),
            ("label1","Label1","labels1","Labels1","semantic1","Semantic1","labels/label1","labels/Label1"),
            ("label2","Label2","labels2","Labels2","semantic2","Semantic2","labels/label2","labels/Label2"))
    bases=[Path(root)/name for name in ("val","Val","valid","Valid","validation","Validation",
           "train","Train","training","Training","test","Test","testing","Testing")]+[Path(root)]
    checked=[]
    for base in bases:
        folders=[next((base/name for name in aliases if (base/name).is_dir()),None) for aliases in groups]
        if any(folder is None for folder in folders):
            continue
        indices=[_index(folder) for folder in folders]
        missing=[name for name in names if any(name.casefold() not in index for index in indices)]
        if missing:
            checked.append(f"{base}: missing {len(missing)} split members")
            continue
        samples=[Sample(name,tuple(str(index[name.casefold()]) for index in indices[:2]),
                              tuple(str(index[name.casefold()]) for index in indices[2:])) for name in names]
        return samples,dict(resolved_root=str(base.resolve()),split_file=str(Path(split_file).resolve()),
            split_sha256=hashlib.sha256(Path(split_file).read_bytes()).hexdigest(),num_samples=len(samples))
    raise FileNotFoundError(f"SECOND semantic pairs do not cover the required split under {root}: {checked}")


def discover(dataset,root,strict=True,split_file=SPLIT_FILE):
    if dataset=="second":
        return discover_second(root,split_file,strict)
    if dataset=="levircd":
        values,audit=original.discover("levircd",Path(root).resolve(),"test")
        samples=[Sample(value.name,tuple(map(str,value.images)),(str(value.label),)) for value in values]
    else:
        values,audit=seg.discover_dataset(dataset,Path(root).resolve(),strict=strict,mask_id_base="auto")
        samples=[Sample(value.name,(str(value.image_path),),(str(value.mask_path),),value.mask_id_base) for value in values]
    if len({sample.name for sample in samples})!=len(samples):
        raise ValueError(f"Colliding output names in {dataset}")
    if strict and len(samples)!=spec(dataset)["expected"]:
        raise ValueError(f"{dataset}: {len(samples)} != {spec(dataset)['expected']}")
    return samples,audit


def load_images(sample,dataset):
    if dataset in seg.DATASET_SPECS:
        value=seg.EvalSample(sample.name,Path(sample.images[0]),Path(sample.labels[0]),mask_id_base=sample.mask_id_base)
        return [Image.fromarray(seg.load_rgb(value,dataset))]
    images=[]
    for path in sample.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if len({image.size for image in images})!=1:
        raise ValueError(f"Temporal extents differ: {sample.name}")
    return images


def load_second_label(path):
    with Image.open(path) as opened:
        raw=np.asarray(opened.convert("RGB") if opened.mode in ("RGB","RGBA","P") else opened)
    if raw.ndim==2:
        if not np.all((raw>=0)&(raw<=6)):
            raise ValueError(f"SECOND IDs must be 0..6, got {np.unique(raw)} at {path}")
        return raw.astype(np.uint8)
    ids=np.full(raw.shape[:2],255,np.uint8)
    for rgb,value in SECOND_RGB.items():
        ids[np.all(raw[...,:3]==rgb,axis=-1)]=value
    if np.any(ids==255):
        raise ValueError(f"Unknown SECOND GT colors at {path}: {np.unique(raw[ids==255],axis=0)[:8]}")
    return ids


def load_targets(sample,dataset,shape):
    if dataset in seg.DATASET_SPECS:
        value=seg.EvalSample(sample.name,Path(sample.images[0]),Path(sample.labels[0]),mask_id_base=sample.mask_id_base)
        return [seg.load_target(value,dataset,height=shape[0],width=shape[1])]
    if dataset=="second":
        targets=[load_second_label(path) for path in sample.labels]
    else:
        with Image.open(sample.labels[0]) as opened:
            raw=np.asarray(opened)
        if raw.ndim!=2 or not np.all(np.isin(raw,(0,1,255))):
            raise ValueError(f"Expected binary native LEVIR-CD GT: {sample.labels[0]}")
        targets=[(raw>0).astype(np.uint8)]
    if any(target.shape!=shape for target in targets):
        raise ValueError(f"Native target extent mismatch: {sample.name}")
    return targets


def second_query(query_key):
    for key,index,name in SECOND_QUERIES:
        if key==query_key: return index,name
    raise ValueError("SECOND requires an explicit category query key")


def prompt(dataset,attempt=0,query_key=None):
    if dataset=="second":
        _,name=second_query(query_key)
        if attempt not in (0,1,2): raise ValueError("Expected at most two fixed format retries")
        # Adapt the native per-category binary interface to an undirected
        # semantic change query; never enumerate seven categorical outputs.
        task=f"[SEG] Please segment the {name} areas that changed between these two images."
        if attempt==1:
            task+=f" Return only a rectangular mask using others and {name}, as category*count runs, one row per line."
        elif attempt==2:
            task+=f" Use others for background and {name} for the requested changed areas. Return only the complete category*count mask."
        return task
    if dataset=="levircd":
        # The official demo asks for one building-change foreground. It does
        # not request a categorical map or a reverse temporal prediction.
        return original.CD_PROMPT if attempt==0 else original.format_retry_prompt(dataset,attempt)
    names=", ".join(spec(dataset)["classes"])
    task="Segment this remote-sensing image."
    suffix=("" if attempt==0 else " Return only the complete mask, no explanation." if attempt==1 else
            " Check that every encoded row has exactly 24 positions and every category is in the list.")
    return (f"[SEG] {task} Use only these categories: {names}. Return a 24 by 24 semantic mask, one encoded row per line. "
            "Write each category followed by an asterisk and a positive integer run length; separate runs with commas."
            +suffix)


def decode(record,dataset,query_key=None):
    generation=record.get("generation",{})
    if generation.get("hit_token_limit") and not generation.get("ended_with_eos"):
        raise ValueError("Refusing a truncated descriptor without EOS")
    names=["unchanged",second_query(query_key)[1]] if dataset=="second" else spec(dataset)["classes"]
    mapping={original.normalize_label(name):index for index,name in enumerate(names)}
    aliases={"buildings":"building","roads":"road","trees":"tree","playgrounds":"playground",
             "bareland":"bare land","barren land":"barren","people":"human","person":"human"}
    if dataset=="loveda": aliases.update({"bareland":"barren","bare land":"barren","agricultural":"agriculture"})
    if dataset=="second": aliases.update({"ground":"bare land","barren":"bare land","barren land":"bare land"})
    background="background" if "background" in names else "unchanged" if "unchanged" in names else "background clutter" if "background clutter" in names else None
    if background: aliases.update({"others":background,"other":background})
    for alias,target in aliases.items():
        if target in names: mapping[alias]=names.index(target)
    policy=background_policy(dataset)
    if policy:
        mapping.update({alias:policy["id"] for alias in policy["aliases"]})
    text=record["response"]
    if not isinstance(text,str) or len(text)>1_000_000: raise ValueError("Expected bounded descriptor text")
    text=text.strip()
    if text.startswith("```"):
        lines=text.splitlines()
        if lines[-1].strip()!="```": raise ValueError("Unclosed code fence")
        text="\n".join(lines[1:-1])
    rows=[line.strip() for line in text.splitlines() if line.strip()]
    if rows and rows[0].casefold() in ("mask:","segmentation mask:","[seg]"):
        rows=rows[1:]
    if not 1<=len(rows)<=64: raise ValueError("Expected 1..64 descriptor rows")
    decoded=[]
    width=None
    for number,row in enumerate(rows,1):
        values=[]
        for item in row.split(","):
            match=re.fullmatch(r"\s*([A-Za-z][A-Za-z _'-]*)\s*\*\s*([0-9]+)\s*",item)
            if not match: raise ValueError(f"Invalid RLE item at row {number}: {item!r}")
            word,count=match.groups()
            key=original.normalize_label(word)
            if key not in mapping: raise ValueError(f"Unknown {dataset} category {word!r}")
            count=int(count)
            if count<=0 or len(values)+count>64: raise ValueError("Invalid descriptor run length")
            values.extend([mapping[key]]*count)
        if width is None: width=len(values)
        if len(values)!=width: raise ValueError("Ragged descriptor rows")
        decoded.append(values)
    return np.asarray(decoded,np.uint8)


def counts(prediction,target):
    return dict(tp=int(np.sum(prediction&target)),tn=int(np.sum(~prediction&~target)),
                fp=int(np.sum(prediction&~target)),fn=int(np.sum(~prediction&target)))


def second_metrics(class_counts):
    values={}
    for index,name in enumerate(SECOND_CLASSES[1:],1):
        row=class_counts[str(index)]
        tp,tn,fp,fn=(float(row[key]) for key in ("tp","tn","fp","fn"))
        precision=tp/(tp+fp+1e-7); recall=tp/(tp+fn+1e-7)
        values[name]=dict(**row,iou_change=tp/(tp+fp+fn+1e-7),
            f1_change=2*precision*recall/(precision+recall+1e-7),precision=precision,recall=recall)
    return dict(miou_c=float(np.mean([value["iou_change"] for value in values.values()])),
                mf1_c=float(np.mean([value["f1_change"] for value in values.values()])),per_class=values,
                epsilon=1e-7,protocol="DynamicEarth six class-wise undirected semantic change masks; macro over six changed classes")
