"""Recover failed query jobs in a separate run, retaining successful predictions."""
from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import eval_rsunivlm_paper_queue as queue
from tools import eval_rsunivlm_guarded as guarded
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker
from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint, checkpoint_identity


def keys(plan, dataset):
    if native.enabled(plan["protocol"].get("semantic_mode"),dataset):
        return tuple(key for key,_,_ in native.queries(dataset))
    return worker.outputs(dataset)


def instruction(plan, dataset, key, variant):
    if native.enabled(plan["protocol"].get("semantic_mode"),dataset):
        return native.prompt(dataset,key,variant)
    return protocol.prompt(dataset,variant,query_key=key if dataset=="second" else None)


def decoded(plan, dataset, key, attempt):
    if native.enabled(plan["protocol"].get("semantic_mode"),dataset):
        grid=native.decode_binary(attempt)[0]
    else:
        grid=protocol.decode(attempt,dataset,query_key=key if dataset=="second" else None)
    from tools.rsunivlm_constrained_rle import validate_grid
    validate_grid(attempt,grid)
    return grid


def sample_id(plan, dataset, item):
    return fingerprint([fingerprint([plan["identity"],dataset]),item["sample"],item["files"]])


def read_masks(folder, receipt, identity, expected_keys, limit):
    if not receipt or receipt.get("identity")!=identity: return None
    result={}
    try:
        for key in expected_keys:
            path=folder/"pred_mask"/f"{receipt['_sample']}_{key}.png"
            if path.is_symlink(): raise ValueError("Unexpected source prediction symlink")
            with Image.open(path) as opened: array=np.asarray(opened).copy()
            if array.dtype!=np.uint8 or list(array.shape)!=receipt["shape"] or not np.all(array<limit): return None
            if hashlib.sha256(array.tobytes()).hexdigest()!=receipt["sha256"][key]: return None
            result[key]=path
    except (OSError,ValueError,KeyError): return None
    return result


def link_or_copy(source, destination):
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or source.read_bytes()!=destination.read_bytes():
            raise ValueError(f"Conflicting destination mask: {destination}")
        return "already_present"
    temporary=destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source,temporary); method="hardlink"
        except OSError as error:
            if error.errno!=errno.EXDEV: raise
            shutil.copy2(source,temporary); method="copy"
        temporary.replace(destination)
        return method
    finally:
        if temporary.exists(): temporary.unlink()


def prepare(source, requested, gpu_ids=None, retries=2):
    old=io.read_json(source/"plan.json"); config=io.read_json(source/"run_config.json")
    if not old or not config or old.get("identity")!=fingerprint(old.get("protocol")) or config.get("protocol")!=old["protocol"] or config.get("identity")!=old["identity"]:
        raise ValueError("Source plan/config identity is missing or inconsistent")
    offline_finished=io.read_json(source/"offline_runtime_finished.json")
    valid_offline=bool(offline_finished and offline_finished.get("identity")==old["identity"] and offline_finished.get("mode")=="offline_format_repair")
    if not (source/"runtime_finished.json").is_file() and not valid_offline:
        raise ValueError("Wait for the original whole queue to finish; runtime_finished.json is missing")
    if old["protocol"].get("code_sha256")!=queue.source_identity():
        raise ValueError("Numerical source changed since the original run; do not import caches across unverified code")
    if retries not in (1,2) or retries<=old["protocol"]["format_retries"]:
        raise ValueError("Recovery must increase the original fixed-template retry budget, at most to 2")
    chosen=[]; fatal=[]; failed_counts={}
    for entry in old["entries"]:
        dataset=entry["dataset"]
        if requested and dataset not in requested: continue
        if fingerprint(entry["samples"])!=old["protocol"]["populations"].get(dataset): raise ValueError(f"Source manifest differs: {dataset}")
        population=entry["samples"][:old["max_samples"]] if old["max_samples"] else entry["samples"]
        count=0; usable=True
        for rank in range(len(old["gpu_ids"])):
            result=io.read_json(source/dataset/"shards"/f"rank{rank:03d}"/"metrics.json")
            if not result or result.get("identity")!=fingerprint([old["identity"],dataset]) or not result.get("exhausted_population"):
                usable=False; break
            expected={row["sample"]["name"] for row in population[rank::len(old["gpu_ids"]) ]}
            names=[row["sample"] for row in result.get("rows",[])]+[row["sample"] for row in result.get("failed_samples",[])]
            if len(names)!=len(set(names)) or set(names)!=expected or result.get("num_failed",0)!=len(result.get("failed_samples",[])):
                raise ValueError(f"Source shard accounting differs: {dataset}/rank{rank}")
            count+=result.get("num_failed",0)
        if not usable:
            fatal.append(dataset); continue
        if count:
            chosen.append(copy.deepcopy(entry)); failed_counts[dataset]=count
    if requested and set(requested)-{entry["dataset"] for entry in old["entries"]}: raise ValueError("Unknown requested source dataset")
    if requested and fatal: raise ValueError(f"These datasets stopped for a non-format error or were interrupted: {fatal}")
    if not chosen: raise ValueError(f"No completed dataset population has recoverable format failures; fatal/incomplete datasets: {fatal}")
    new=copy.deepcopy(old); value=new["protocol"]
    value["datasets"]=[entry["dataset"] for entry in chosen]
    value["format_retries"]=retries
    for field in ("populations","prediction_background","native_semantic","prompts"):
        value[field]={name:item for name,item in value.get(field,{}).items() if name in value["datasets"]}
    for name in value["datasets"]:
        value["prompts"][name]=({key:[instruction(new,name,key,index) for index in range(retries+1)] for key in keys(new,name)}
            if native.enabled(value.get("semantic_mode"),name) or name=="second" else
            [protocol.prompt(name,index) for index in range(retries+1)])
        if native.enabled(value.get("semantic_mode"),name): value["native_semantic"][name]=native.settings(name,retries)
    if "second" not in value["datasets"]: value["second_query"]=None
    if "levircd" not in value["datasets"]: value["levircd_query"]=None
    value["recovery"]=dict(source_root=str(source),source_identity=old["identity"],source_plan_sha256=hashlib.sha256((source/"plan.json").read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy="reuse_valid_masks_and_answers; continue_invalid_queries_with_remaining_fixed_templates; no_GT_selection")
    if gpu_ids:
        gpus=[value.strip() for value in gpu_ids.split(",")]
        if len(gpus)!=len(old["gpu_ids"]) or len(set(gpus))!=len(gpus) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",s) for s in gpus):
            raise ValueError("Use the original number of distinct GPU IDs; sample sharding remains fixed")
        new["gpu_ids"]=gpus
    new["entries"]=chosen; new["identity"]=fingerprint(value)
    return old,new,dict(source_failed_counts=failed_counts,excluded_fatal_or_interrupted_datasets=fatal)


def verify_runtime_assets(plan):
    value=plan["protocol"]; settings=value["model_settings"]
    if queue.upstream_files(settings["source_root"])!=value["upstream_code_sha256"]:
        raise ValueError("Upstream model source changed since the original run")
    if checkpoint_identity(Path(settings["model_path"]))!=value["checkpoint"]:
        raise ValueError("Checkpoint identity changed since the original run")
    vision=[]
    for path in sorted(Path(settings["vision_tower"]).iterdir()):
        if path.is_file() and path.suffix in (".json",".safetensors",".bin"):
            item=dict(path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            if path.suffix==".json": item["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            vision.append(item)
    if vision!=value["vision_files"]: raise ValueError("Vision checkpoint files changed since the original run")


def import_predictions(source, output, old, new):
    counters=dict(reused_complete_samples=0,reused_valid_query_answers=0,imported_failed_query_histories=0,missing_queries=0,
        hardlinked_masks=0,copied_masks=0,already_present_masks=0)
    size=old["protocol"]["tile_size"]
    for entry in new["entries"]:
        dataset=entry["dataset"]; expected_keys=keys(new,dataset)
        selected=entry["samples"][:new["max_samples"]] if new["max_samples"] else entry["samples"]
        for position,item in enumerate(tqdm(selected,desc=f"Import {dataset} cache",unit="image")):
            sample=protocol.Sample.restore(item["sample"])
            if Path(sample.name).name!=sample.name or sample.name in (".","..") or "\\" in sample.name: raise ValueError("Unsafe sample name")
            if protocol.metadata(sample)!=item["files"]: raise ValueError(f"Source inputs changed: {dataset}/{sample.name}")
            rank=position%len(new["gpu_ids"])
            origin=source/dataset/"shards"/f"rank{rank:03d}"; target=output/dataset/"shards"/f"rank{rank:03d}"
            for folder in ("pred_mask","raw_text","receipts"): (target/folder).mkdir(parents=True,exist_ok=True)
            old_id,new_id=sample_id(old,dataset,item),sample_id(new,dataset,item)
            limit=2 if native.enabled(new["protocol"].get("semantic_mode"),dataset) else protocol.prediction_class_count(dataset)
            present=io.read_json(target/"receipts"/f"{sample.name}.json")
            if present and read_masks(target,dict(present,_sample=sample.name),new_id,expected_keys,limit): continue
            receipt=io.read_json(origin/"receipts"/f"{sample.name}.json")
            valid=read_masks(origin,dict(receipt,_sample=sample.name) if receipt else None,old_id,expected_keys,limit)
            if valid:
                for key,path in valid.items():
                    method=link_or_copy(path,target/"pred_mask"/f"{sample.name}_{key}.png")
                    counters[{"hardlink":"hardlinked_masks","copy":"copied_masks","already_present":"already_present_masks"}[method]]+=1
                imported=dict(receipt,identity=new_id,reused_from=dict(root=str(source),identity=old_id,receipt=str(origin/"receipts"/f"{sample.name}.json")))
                io.write_json(target/"receipts"/f"{sample.name}.json",imported)
                counters["reused_complete_samples"]+=1; continue
            images=protocol.load_images(sample,dataset)
            try: width,height=images[0].size
            finally:
                for image in images: image.close()
            for key in expected_keys:
                for y in range(0,height,size):
                    for x in range(0,width,size):
                        w,h=min(size,width-x),min(size,height-y)
                        spatial=dict(source_box=[x,y,x+w,y+h],extent=[width,height],tile_size=size,padding="black_right_bottom")
                        filename=f"{sample.name}_{key}_y{y}_x{x}.json"
                        path=origin/"raw_text"/filename; record=io.read_json(path)
                        expected=fingerprint([old_id,key,spatial]); target_id=fingerprint([new_id,key,spatial])
                        destination=target/"raw_text"/filename
                        if destination.exists():
                            current=io.read_json(destination)
                            if not current or current.get("identity")!=target_id: raise ValueError(f"Destination raw answer conflicts: {destination}")
                            continue
                        if not record: counters["missing_queries"]+=1; continue
                        if record.get("identity")!=expected or record.get("spatial")!=spatial:
                            raise ValueError(f"Source query identity differs: {path}")
                        attempts=record.get("attempts",[])
                        from tools.rsunivlm_constrained_rle import validate_history
                        try: validate_history(old,dataset,key,attempts,instruction)
                        except ValueError as error: raise ValueError(f"{error}: {path}") from error
                        is_valid=False
                        for index,attempt in enumerate(attempts):
                            try: decoded(old,dataset,key,attempt); is_valid=True
                            except (ValueError,KeyError): pass
                        imported=copy.deepcopy(record); imported["identity"]=target_id
                        provenance=dict(path=str(path),source_identity=expected,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        imported["reused_from"]=provenance
                        for attempt in imported["attempts"]: attempt["reused_from"]=provenance
                        io.write_json(destination,imported)
                        counters["reused_valid_query_answers" if is_valid else "imported_failed_query_histories"]+=1
    return counters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run",required=True,help="Actual OUTPUT_DIR printed by the finished original queue")
    parser.add_argument("--datasets",default="",help="Comma-separated subset; default selects datasets with format failures")
    parser.add_argument("--output_dir",default="")
    parser.add_argument("--gpu_ids",default="")
    parser.add_argument("--format_retries",type=int,choices=(1,2),default=2)
    parser.add_argument("--plan_only",action="store_true")
    args=parser.parse_args(); source=Path(args.source_run).expanduser().resolve()
    if not source.is_dir(): raise FileNotFoundError(source)
    with (source/".queue.lock").open("a") as source_lock:
        try: fcntl.flock(source_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("The original queue is still running. Wait; do not alter its files.")
        requested=[value.strip() for value in args.datasets.split(",")] if args.datasets else []
        if len(requested)!=len(set(requested)): raise ValueError("Choose distinct datasets")
        old,new,report=prepare(source,requested,args.gpu_ids or None,args.format_retries)
        verify_runtime_assets(old)
        requested_output=Path(args.output_dir).expanduser().resolve() if args.output_dir else source.with_name(source.name+f"_failed_retry_r{args.format_retries}")
        if requested_output==source or source in requested_output.parents or requested_output in source.parents:
            raise ValueError("Recovery output must be separate from, and not contain or be inside, the source run")
        output,resolution=queue.resolve_output_root(requested_output,new["protocol"])
        new["output_dir"]=str(output)
        print(f"[recovery] datasets={new['protocol']['datasets']}; source failures={report['source_failed_counts']}; output={output}",flush=True)
        if args.plan_only:
            print("Plan only: no files imported, model loaded, or GPU process started.",flush=True); return 0
        output.mkdir(parents=True,exist_ok=True)
        with (output/".queue.lock").open("a") as output_lock:
            fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            io.write_json(output/"run_config.json",dict(identity=new["identity"],protocol=new["protocol"]))
            io.write_json(output/"plan.json",new)
            if resolution: io.write_json(output/"output_resolution.json",resolution)
            counters=import_predictions(source,output,old,new)
            io.write_json(output/"recovery_import.json",dict(**report,**counters,source=str(source),source_identity=old["identity"],target_identity=new["identity"]))
            print(f"[recovery] cache import: {counters}",flush=True)
            (output/"logs").mkdir(exist_ok=True)
            return guarded.launch(new, legacy=queue)


if __name__=="__main__": raise SystemExit(main())
