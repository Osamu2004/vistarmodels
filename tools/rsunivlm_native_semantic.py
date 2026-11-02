"""Paper single-class prompts and official-demo binary masks; no invented fusion."""
from __future__ import annotations

from collections import Counter, deque
import hashlib
from pathlib import Path
import re
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from baselines.segearth_ov import protocols as seg
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from tools import rsunivlm_paper_protocol as protocol

MODE = "native_classwise"
SEMANTIC_DATASETS = tuple(seg.DATASET_SPECS)
PAPER = "https://arxiv.org/html/2412.05679v2#S10"
UPSTREAM = "https://github.com/xuliu-cyber/RSUniVLM/blob/66dce1dc68b81a6c62f02fb8e5b306571092a864/app.py"


def enabled(mode, dataset):
    return mode == MODE and dataset in SEMANTIC_DATASETS


def queries(dataset):
    names = protocol.spec(dataset)["classes"]
    # The two binary benchmarks need only their named foreground query.
    ids = (1,) if len(names) == 2 else range(len(names))
    return tuple((f"class_{index:02d}_" + re.sub(r"[^a-z0-9]+", "_", names[index].casefold()).strip("_"),
                  index, names[index]) for index in ids)


def query(dataset, key):
    for item in queries(dataset):
        if item[0] == key: return item
    raise ValueError(f"Unknown native semantic query: {dataset}/{key}")


def prompt(dataset, key, variant=0):
    name = query(dataset, key)[2]
    templates = (
        "[SEG] Can you segment the {name} in this image?",
        "[SEG] Please segment the {name} in this image.",
        "[SEG] What is {name} in this image? Please respond with segmentation mask.",
    )
    if not 0 <= variant < len(templates): raise ValueError("At most two explicit prompt retries")
    return templates[variant].format(name=name)


def contract(dataset):
    result = protocol.spec(dataset).copy()
    result.pop("prediction_background", None)  # Each query has its own binary background.
    multi = len(result["classes"]) > 2
    result.update(prediction_kind="independent binary category masks", query_classes=[dict(key=k,id=i,name=n) for k,i,n in queries(dataset)],
        standard_exclusive_metrics_available=not multi, multiclass_fusion="not supplied by public sources; not performed",
        primary="miou" if multi else result["primary"],table7_protocol_verified=False)
    return result


def settings(dataset, retries):
    return dict(mode=MODE, prompt_source=PAPER, decoder_source=UPSTREAM,
        queries=[dict(key=k,id=i,name=n,prompts=[prompt(dataset,k,v) for v in range(retries+1)]) for k,i,n in queries(dataset)],
        decoder="others -> 0; every other syntactically valid label -> 1 (official demo)",
        restoration="expand each descriptor cell to 20x20; cv2.INTER_NEAREST to source tile; crop valid extent",
        multiclass_fusion="none", standard_exclusive_metrics_available=len(protocol.spec(dataset)["classes"])==2,
        scoring="independent one-vs-rest query IoU/F1; binary benchmarks also retain standard dense metrics",
        target_reference="RSUniVLM Table 7: Vaihingen 54.46, UDD5 65.41, VDD 49.60",
        table7_protocol_verified=False,
        scoring_evidence="mIoU is reported but aggregation/fusion script is not public; this adapter pools TP/FP/FN per fixed evaluation class",
        deviations=["user-requested source512 crops and direct multi-GPU batching", "bounded RLE validation and EOS audit",
                    "explicit data/score adapter; public multiclass scoring script unavailable"])


def decode_binary(record):
    generation = record.get("generation", {})
    if generation.get("hit_token_limit") and not generation.get("ended_with_eos"):
        raise ValueError("Refusing a truncated descriptor without EOS")
    text = record.get("response")
    if not isinstance(text, str) or len(text) > 1_000_000: raise ValueError("Expected bounded descriptor text")
    # Native app emits bare RLE. Keep benign fence/header wrappers accepted by
    # the existing adapter, but never invent runs or complete malformed rows.
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```": raise ValueError("Unclosed code fence")
        text = "\n".join(lines[1:-1])
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if rows and rows[0].casefold() in ("mask:", "segmentation mask:", "[seg]"): rows = rows[1:]
    if not 1 <= len(rows) <= 64: raise ValueError("Expected 1..64 descriptor rows")
    decoded, labels, width = [], Counter(), None
    for number, row in enumerate(rows, 1):
        values = []
        for item in row.split(","):
            match = re.fullmatch(r"\s*([A-Za-z][A-Za-z _'-]*)\s*\*\s*([0-9]+)\s*", item)
            if not match: raise ValueError(f"Invalid RLE item at row {number}: {item!r}")
            label, count = match.groups(); count = int(count); label = label.strip().lower()
            if count <= 0 or len(values)+count > 64: raise ValueError("Invalid descriptor run length")
            labels[label] += count
            values.extend([0 if label == "others" else 1] * count)
        if width is None: width = len(values)
        if len(values) != width: raise ValueError("Ragged descriptor rows")
        decoded.append(values)
    return np.asarray(decoded, np.uint8), dict(labels)


def restore(grid, size):
    # Use the actual public-demo sequence, including patch_size=20.
    expanded = np.repeat(np.repeat(grid, 20, axis=0), 20, axis=1)
    return cv2.resize(expanded, (size,size), interpolation=cv2.INTER_NEAREST)


def aggregate(dataset, rows):
    per_class = {}
    for key, index, name in queries(dataset):
        counts = {field:sum(row["query_counts"][key][field] for row in rows) for field in ("tp","tn","fp","fn")}
        tp,fp,fn = counts["tp"],counts["fp"],counts["fn"]
        union, denom = tp+fp+fn, 2*tp+fp+fn
        per_class[key] = dict(id=index,name=name,**counts,iou=tp/union if union else None,f1=2*tp/denom if denom else None)
    ious = [value["iou"] for value in per_class.values() if value["iou"] is not None]
    f1s = [value["f1"] for value in per_class.values() if value["f1"] is not None]
    result = dict(query_miou=float(np.mean(ious)) if ious else 0.0,query_mf1=float(np.mean(f1s)) if f1s else 0.0,
        per_query=per_class,metric_protocol="independent binary category queries; no exclusive multiclass fusion",
        standard_exclusive_metrics_available=len(protocol.spec(dataset)["classes"])==2,
        query_valid_pixels=sum(row["query_valid_pixels"] for row in rows),
        zero_positive_query_pixels=sum(row["zero_positive_query_pixels"] for row in rows),
        overlapping_query_pixels=sum(row["overlapping_query_pixels"] for row in rows))
    result.update(miou=result["query_miou"],mf1=result["query_mf1"],
        metric_formula="mean_c TP_c/(TP_c+FP_c+FN_c), counts pooled over all valid native-resolution pixels",
        metric_scope="independent_class_queries",table7_protocol_verified=False)
    if len(protocol.spec(dataset)["classes"]) == 2:
        matrix = sum((np.asarray(row["confusion_matrix"],np.int64) for row in rows),np.zeros((2,2),np.int64))
        result.update(seg.metrics_for_dataset(matrix,dataset))
        boundary = [row for row in rows if "wfm_3px_mean_present" in row]
        if boundary: result.update(aggregate_binary_boundary_wfm(boundary))
    return result


def evaluate(engine, entry, output):
    dataset = entry["dataset"]; query_list = queries(dataset)
    size = engine.plan["protocol"]["tile_size"]; budget = engine.plan["protocol"]["format_retries"]
    full = [protocol.Sample.restore(row["sample"]) for row in entry["samples"]]
    selected = full[:engine.plan["max_samples"]] if engine.plan["max_samples"] else full
    assigned = selected[engine.rank::engine.world_size]
    expected_meta = {row["sample"]["name"]:row["files"] for row in entry["samples"]}
    root = Path(output)/dataset/"shards"/f"rank{engine.rank:03d}"
    for folder in ("pred_mask","raw_text","receipts","errors"): (root/folder).mkdir(parents=True,exist_ok=True)
    dataset_id = fingerprint([engine.plan["identity"],dataset])
    previous = io.read_json(root/"run_config.json")
    if previous and previous.get("identity") != dataset_id: raise ValueError("Shard has another protocol")
    io.write_json(root/"run_config.json",dict(identity=dataset_id,rank=engine.rank,world_size=engine.world_size,
        assigned_samples=[sample.name for sample in assigned],contract=contract(dataset),native_semantic=settings(dataset,budget)))
    states, pending, rows, failed = deque(), {key:[] for key,_,_ in query_list}, [], []
    progress = tqdm(total=len(assigned),desc=f"GPU{engine.rank} {dataset} native queries",unit="image")

    def apply(job, grid):
        if job.get("done"): raise ValueError("Duplicate query tile completion")
        x,y,w,h = job["box"]
        job["case"]["predictions"][job["key"]][y:y+h,x:x+w] = restore(grid,size)[:h,:w]
        job["case"]["remaining"] -= 1
        job["case"]["shapes"].append(list(grid.shape)); job["done"] = True

    def flush(key=None):
        for name in ((key,) if key is not None else tuple(pending)):
            jobs = pending[name]; pending[name] = []
            try:
                for variant in range(budget+1):
                    waiting = [job for job in jobs if not job.get("done") and len(job["attempts"])==variant]
                    instruction = prompt(dataset,name,variant)
                    for group, answers in engine.forward(waiting,instruction):
                        for job, answer in zip(group,answers):
                            attempt = dict(prompt=instruction,prompt_variant=variant,**answer)
                            job["attempts"].append(attempt)
                            _, index, label = query(dataset,name)
                            record = dict(identity=job["identity"],dataset=dataset,query_key=name,query_class_id=index,
                                query_class=label,decoder="official_demo_binary",spatial=job["spatial"],
                                attempts=job["attempts"],**attempt)
                            io.write_json(job["path"],record)
                            try: grid, labels = decode_binary(record)
                            except (ValueError,KeyError) as error:
                                attempt["decode_error"] = str(error)
                                io.write_json(job["path"],dict(record,decode_error=str(error)))
                                print(f"[native format] {job['path'].name}: {error}; attempt={variant+1}/{budget+1}",flush=True)
                            else:
                                io.write_json(job["path"],dict(record,grid_shape=list(grid.shape),decoded_label_counts=labels))
                                apply(job,grid)
                for job in jobs:
                    if not job.get("done"):
                        job["case"]["error"] = f"Native format budget exhausted: {job['path']}"
                        job["case"].setdefault("format_failures",[]).append(str(job["path"]))
            finally:
                for job in jobs:
                    for image in job["images"]: image.close()

    def finish(case):
        sample = case["sample"]
        if protocol.metadata(sample) != expected_meta[sample.name]: raise ValueError(f"Inputs changed: {sample.name}")
        predictions = case["predictions"]
        if not case["cached"]:
            for key, value in predictions.items(): io.save_prediction(root/"pred_mask"/f"{sample.name}_{key}.png",value)
            io.write_json(root/"receipts"/f"{sample.name}.json",dict(identity=case["identity"],shape=list(case["shape"]),
                sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in predictions.items()},
                descriptor_shapes=case["shapes"],mask_encoding="each query: 0 others, 1 foreground"))
        # All fixed class queries are saved before reading any GT pixels.
        start = time.monotonic()
        target = protocol.load_targets(sample,dataset,case["shape"])[0]
        valid = (target != 255) & (target >= 0) & (target < len(protocol.spec(dataset)["classes"]))
        count_map = sum((value.astype(np.uint16) for value in predictions.values()),np.zeros(case["shape"],np.uint16))
        row = dict(sample=sample.name,native_semantic_query=True,reused_prediction=case["cached"],descriptor_shapes=case["shapes"],
            query_counts={key:protocol.counts(predictions[key][valid].astype(bool),(target[valid]==index)) for key,index,_ in query_list},
            query_valid_pixels=int(valid.sum()),zero_positive_query_pixels=int(np.sum(valid & (count_map==0))),
            overlapping_query_pixels=int(np.sum(valid & (count_map>1))))
        if len(protocol.spec(dataset)["classes"]) == 2:
            prediction = predictions[query_list[0][0]]
            row["confusion_matrix"] = seg.confusion_matrix(prediction,target,num_classes=2,ignore_index=255).tolist()
            if np.any(valid): row.update(score_semantic_boundary_wfm(prediction,target,num_classes=2))
        row["scoring_seconds"] = time.monotonic()-start; rows.append(row)
        with (root/"progress.jsonl").open("a") as stream:
            import json
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        progress.update(1)
        progress.set_postfix(batch=engine.effective_batch,query_miou=f"{aggregate(dataset,[row])['query_miou']:.4f}",cached=int(case["cached"]),failed=len(failed))
        io.write_json(root/"status.json",dict(processed=len(rows),attempted=len(rows)+len(failed),failed=len(failed),
            assigned=len(assigned),last_sample=sample.name,complete=False))

    def drain():
        while states and (states[0].get("error") or states[0]["remaining"]==0):
            case = states.popleft()
            if case.get("error"):
                failure=dict(sample=case["sample"].name,identity=case["identity"],error=case["error"],
                    raw_paths=case.get("format_failures",[]),reason="format_budget_exhausted",scored=False)
                failed.append(failure)
                io.write_json(root/"errors"/f"{case['sample'].name}.json",failure)
                progress.update(1); progress.set_postfix(batch=engine.effective_batch,scored=len(rows),failed=len(failed))
                io.write_json(root/"status.json",dict(processed=len(rows),attempted=len(rows)+len(failed),failed=len(failed),
                    assigned=len(assigned),last_sample=case["sample"].name,complete=False))
                print(f"[native skip] {case['sample'].name}: format failed; continuing with remaining samples (failed={len(failed)})",flush=True)
                continue
            finish(case)

    try:
        for sample in assigned:
            if protocol.metadata(sample) != expected_meta[sample.name]: raise ValueError(f"Inputs changed: {sample.name}")
            images = protocol.load_images(sample,dataset)
            try:
                width,height = images[0].size
                identity = fingerprint([dataset_id,sample.serialize(),expected_meta[sample.name]])
                receipt = io.read_json(root/"receipts"/f"{sample.name}.json"); cached = {}
                if receipt and receipt.get("identity") == identity:
                    for key,_,_ in query_list:
                        try:
                            with Image.open(root/"pred_mask"/f"{sample.name}_{key}.png") as opened: value = np.asarray(opened).copy()
                            if value.shape!=(height,width) or value.dtype!=np.uint8 or not np.all(value<2): raise ValueError("Invalid query mask")
                            if hashlib.sha256(value.tobytes()).hexdigest()!=receipt["sha256"][key]: raise ValueError("Query mask hash changed")
                            cached[key] = value
                        except (OSError,KeyError,ValueError): cached={}; break
                case = dict(sample=sample,identity=identity,shape=(height,width),cached=bool(cached),
                    remaining=0 if cached else ((height+size-1)//size)*((width+size-1)//size)*len(query_list),
                    shapes=receipt.get("descriptor_shapes",[]) if cached else [],
                    predictions=cached or {key:np.empty((height,width),np.uint8) for key,_,_ in query_list})
                states.append(case)
                if not cached:
                    for key,_,_ in query_list:
                        for y in range(0,height,size):
                            for x in range(0,width,size):
                                w,h = min(size,width-x),min(size,height-y)
                                spatial = dict(source_box=[x,y,x+w,y+h],extent=[width,height],tile_size=size,padding="black_right_bottom")
                                job = dict(case=case,key=key,box=(x,y,w,h),spatial=spatial,
                                    path=root/"raw_text"/f"{sample.name}_{key}_y{y}_x{x}.json",
                                    identity=fingerprint([identity,key,spatial]),attempts=[])
                                record = io.read_json(job["path"])
                                if record and record.get("identity")==job["identity"]:
                                    job["attempts"] = record.get("attempts",[])
                                    for attempt in job["attempts"]:
                                        try: grid,_ = decode_binary(attempt)
                                        except (ValueError,KeyError): continue
                                        apply(job,grid); break
                                    if job.get("done"): continue
                                    if len(job["attempts"])>=budget+1:
                                        case["error"]=f"Stored native format budget exhausted: {job['path']}"
                                        case.setdefault("format_failures",[]).append(str(job["path"])); continue
                                tile = Image.new("RGB",(size,size),(0,0,0)); tile.paste(images[0].crop((x,y,x+w,y+h)),(0,0))
                                job["images"]=[tile]; pending[key].append(job)
                                if len(pending[key])>=engine.effective_batch: flush(key); drain()
                if cached or case["remaining"]==0 or case.get("error"): flush()
                drain()
            finally:
                for image in images: image.close()
        flush(); drain()
        if states or len(rows)+len(failed)!=len(assigned): raise ValueError("Unaccounted native query population")
        for failure in failed: io.write_json(root/"errors"/f"{failure['sample']}.json",failure)
        io.write_json(root/"failed_samples.json",dict(failed_samples=failed,num_failed=len(failed)))
        result = dict(identity=dataset_id,dataset=dataset,rank=engine.rank,world_size=engine.world_size,complete=not failed,
            is_full_benchmark=False,scope="native query rank shard",processed=len(rows),assigned_samples=[sample.name for sample in assigned],
            attempted=len(rows)+len(failed),exhausted_population=True,num_failed=len(failed),failed_samples=failed,
            error=f"{len(failed)} samples failed format validation; remaining samples were processed" if failed else None,
            scores_scope="successfully decoded complete samples only" if failed else "all assigned samples",
            scores=aggregate(dataset,rows),prediction_keys=[key for key,_,_ in query_list],effective_batch=engine.effective_batch,
            actual_batch_sizes_this_process=sorted(engine.actual_batches),oom_splits=engine.oom_splits,rows=rows)
        io.write_json(root/"metrics.json",result)
        io.write_json(root/"status.json",dict(processed=len(rows),attempted=len(rows)+len(failed),failed=len(failed),
            assigned=len(assigned),complete=not failed,exhausted_population=True))
        return result
    except Exception as error:
        io.write_json(root/"metrics.json",dict(identity=dataset_id,dataset=dataset,rank=engine.rank,complete=False,
            is_full_benchmark=False,processed=len(rows),attempted=len(rows)+len(failed),num_failed=len(failed),
            failed_samples=failed,exhausted_population=False,rows=rows,error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        progress.close()
        for jobs in pending.values():
            for job in jobs:
                for image in job.get("images",[]): image.close()
