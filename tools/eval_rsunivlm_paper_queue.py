"""Plan, run and merge the CVPR dense-prediction queue on independent GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import io as text_io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint,checkpoint_identity,source_revision
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_native_semantic as native
from tools.rsunivlm_batch_import import resolve_output_root


def source_identity():
    names=["tools/eval_rsunivlm_paper_queue.py","tools/rsunivlm_paper_protocol.py","tools/rsunivlm_paper_worker.py",
           "tools/rsunivlm_batch_runtime.py","tools/rsunivlm_native_semantic.py","baselines/rsunivlm/model.py","baselines/rsunivlm/protocols.py",
           "baselines/rsunivlm/evaluate.py","baselines/segearth_ov/protocols.py","baselines/flair_protocol.py",
           "baselines/binary_boundary_wfm.py","baselines/levircd_common.py","tools/rsunivlm_batch_import.py",
           "baselines/rskt_seg/xbd_label_utils.py",
           "assets/splits/second_bisrnet_dynamic_earth_val.txt"]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",default=",".join(protocol.DATASETS))
    parser.add_argument("--gpu_ids",default="0,1")
    parser.add_argument("--batch_size",type=int,default=16,help="Positive tile-sequence batch per GPU, no fixed upper bound")
    parser.add_argument("--tile_size",type=int,default=512,choices=(512,))
    parser.add_argument("--output_dir",default="/root/data/experiment/rsunivlm_native_classwise_tile512_bs16_2gpu_v1")
    parser.add_argument("--semantic_mode",choices=("native_classwise","legacy_categorical"),default="native_classwise",
        help="Paper per-class queries; legacy multiclass-RLE adapter remains explicit")
    parser.add_argument("--data_roots_json",default="",help="JSON object overriding dataset root paths")
    parser.add_argument("--source_root",default=os.environ.get("RSUNIVLM_ROOT",str(ROOT/"third_party/RSUniVLM")))
    parser.add_argument("--model_path",default=os.environ.get("RSUNIVLM_MODEL_PATH","/root/data/weight/llava-qwen-gmoe"))
    parser.add_argument("--vision_tower",default=os.environ.get("RSUNIVLM_VISION_TOWER","/root/data/weight/siglip-so400m-patch14-384"))
    parser.add_argument("--dtype",default="float16",choices=("float16","bfloat16"))
    parser.add_argument("--attention",default="eager",choices=("eager","sdpa","flash_attention_2"))
    parser.add_argument("--max_new_tokens",type=int,default=32000)
    parser.add_argument("--format_retries",type=int,default=0,choices=(0,1,2))
    parser.add_argument("--max_samples",type=int,default=0,help="First N per dataset, development only; 0 means full")
    parser.add_argument("--allow_nonstandard_population",action="store_true")
    parser.add_argument("--plan_only",action="store_true")
    parser.add_argument("--worker_plan",default="",help=argparse.SUPPRESS)
    parser.add_argument("--rank",type=int,default=0,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.batch_size<=0 or args.max_samples<0 or args.max_new_tokens<=0:
        parser.error("batch_size/max_new_tokens must be positive; max_samples must be nonnegative")
    return args


def upstream_files(source_root):
    names=("llava/model/builder.py","llava/model/llava_arch.py","llava/model/language_model/llava_qwen_gmoe.py",
           "llava/model/multimodal_encoder/siglip_encoder.py","llava/mm_utils.py","llava/conversation.py")
    return {name:hashlib.sha256((Path(source_root)/name).read_bytes()).hexdigest() for name in names}


def build_plan(args):
    semantic_mode=args.semantic_mode
    datasets=args.datasets.split(",")
    if not datasets or len(set(datasets))!=len(datasets) or any(name not in protocol.DATASETS for name in datasets):
        raise ValueError(f"Choose distinct datasets from {protocol.DATASETS}")
    gpus=[value.strip() for value in args.gpu_ids.split(",")]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+",value) for value in gpus) or len(set(gpus))!=len(gpus):
        raise ValueError("GPU IDs must be distinct nonempty identifiers")
    roots=dict(protocol.DEFAULT_ROOTS)
    if args.data_roots_json:
        overrides=json.loads(Path(args.data_roots_json).read_text())
        if not isinstance(overrides,dict) or set(overrides)-set(protocol.DATASETS): raise ValueError("Unknown dataset root override")
        roots.update(overrides)
    entries,errors=[],{}
    for dataset in datasets:
        try:
            samples,audit=protocol.discover(dataset,roots[dataset],strict=not args.allow_nonstandard_population)
            for sample in samples:
                if Path(sample.name).name!=sample.name or sample.name in (".","..") or "\\" in sample.name:
                    raise ValueError(f"Unsafe sample name: {sample.name}")
            entries.append(dict(dataset=dataset,root=roots[dataset],contract=native.contract(dataset) if native.enabled(semantic_mode,dataset) else protocol.spec(dataset),audit=audit,
                samples=[dict(sample=sample.serialize(),files=protocol.metadata(sample)) for sample in samples]))
            print(f"[preflight] {dataset}: {len(samples)} samples; {protocol.spec(dataset)['split']}",flush=True)
        except Exception as error:
            errors[dataset]=str(error)
    if errors: raise ValueError("Dataset preflight failed before loading weights:\n"+json.dumps(errors,indent=2))
    if not (Path(args.vision_tower)/"config.json").is_file(): raise FileNotFoundError(args.vision_tower)
    vision=[]
    for path in sorted(Path(args.vision_tower).iterdir()):
        if path.is_file() and path.suffix in (".json",".safetensors",".bin"):
            item=dict(path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            if path.suffix==".json": item["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            vision.append(item)
    settings=dict(source_root=str(Path(args.source_root).resolve()),model_path=str(Path(args.model_path).resolve()),
        vision_tower=str(Path(args.vision_tower).resolve()),dtype=args.dtype,attention=args.attention,max_new_tokens=args.max_new_tokens)
    value=dict(schema="rsunivlm_paper_queue_v2",datasets=datasets,world_size=len(gpus),batch_size=args.batch_size,semantic_mode=semantic_mode,
        format_failure_policy="record_failed_sample_and_continue_v1",
        tile_size=args.tile_size,format_retries=args.format_retries,strict_population=not args.allow_nonstandard_population,
        source_revision=source_revision(Path(args.source_root)),upstream_code_sha256=upstream_files(args.source_root),
        checkpoint=checkpoint_identity(Path(args.model_path).resolve()),
        vision_files=vision,model_settings=settings,code_sha256=source_identity(),
        populations={entry["dataset"]:fingerprint(entry["samples"]) for entry in entries},
        prediction_background={name:protocol.background_policy(name) for name in datasets
            if not native.enabled(semantic_mode,name) and protocol.background_policy(name)},
        native_semantic={name:native.settings(name,args.format_retries) for name in datasets if native.enabled(semantic_mode,name)},
        prompts={name:({key:[protocol.prompt(name,index,query_key=key) for index in range(args.format_retries+1)]
                for key,_,_ in protocol.SECOND_QUERIES} if name=="second" else
                {key:[native.prompt(name,key,index) for index in range(args.format_retries+1)] for key,_,_ in native.queries(name)}
                if native.enabled(semantic_mode,name) else
                [protocol.prompt(name,index) for index in range(args.format_retries+1)]) for name in datasets},
        second_query=(dict(policy="six_class_undirected_queries_v1",classes=[dict(key=key,id=index,name=name)
            for key,index,name in protocol.SECOND_QUERIES],input_order=["T1","T2"],reverse_query=False,
            prediction_union=False,queries_per_class_per_tile=1,overlap_between_classes=True,
            gt="class present in either change annotation",primary_metric="six-class miou_c") if "second" in datasets else None),
        levircd_query=(dict(policy="official_prompt_forward_binary_v1",image_order=["A","B"],
            temporal_directions=["forward"],prediction_kind="binary building change",reverse_query=False,
            direction_union=False,source_tiles="VISTAR adapter: native512 nonoverlap") if "levircd" in datasets else None),
        shard_policy="global selected sample index modulo world_size; no overlap",save_images="predicted masks only",
        verification="direct batch; no single-image equivalence claim; no automatic mismatch fallback",
        spatial="512 native nonoverlap, black right/bottom padding, official384 vision processor, nearest descriptor restoration",
        parameter_selection="fixed masks/classes; no GT-based output selection")
    return dict(identity=fingerprint(value),protocol=value,entries=entries,gpu_ids=gpus,max_samples=args.max_samples)


def worker(plan_path,rank):
    from tools.rsunivlm_paper_worker import Engine
    import importlib.metadata
    import torch
    plan=json.loads(Path(plan_path).read_text())
    if plan["identity"]!=fingerprint(plan["protocol"]): raise ValueError("Queue protocol identity is inconsistent")
    for entry in plan["entries"]:
        if fingerprint(entry["samples"])!=plan["protocol"]["populations"][entry["dataset"]]:
            raise ValueError("Dataset manifest changed since planning")
    if plan["protocol"]["code_sha256"]!=source_identity(): raise ValueError("Code changed since queue planning")
    if not 0<=rank<len(plan["gpu_ids"]): raise ValueError("Invalid rank")
    settings=plan["protocol"]["model_settings"]
    if upstream_files(settings["source_root"])!=plan["protocol"]["upstream_code_sha256"]:
        raise ValueError("Upstream model code changed since planning")
    if checkpoint_identity(Path(settings["model_path"]))!=plan["protocol"]["checkpoint"]:
        raise ValueError("Checkpoint identity changed since planning")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for the GPU worker")
    versions={}
    for package in ("torch","torchvision","transformers","accelerate","numpy","Pillow","rasterio","tifffile"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]="unavailable"
    storage.write_json(Path(plan["output_dir"])/f"environment_rank{rank:03d}.json",
        dict(rank=rank,physical_gpu=plan["gpu_ids"][rank],visible_device=torch.cuda.get_device_name(0),versions=versions))
    engine=Engine(plan,rank,len(plan["gpu_ids"]))
    failures={}
    for entry in plan["entries"]:
        dataset=entry["dataset"]
        try:
            print(f"[worker {rank}] start {dataset}",flush=True)
            result=engine.evaluate(entry,plan["output_dir"])
            if not result.get("complete",False):
                failures[dataset]=result.get("error") or f"{result.get('num_failed',0)} failed samples; dataset remains incomplete"
                print(f"[worker {rank}] {dataset}: remaining population processed; {failures[dataset]}; continuing queue",flush=True)
        except Exception as error:
            failures[dataset]=str(error)
            storage.write_json(Path(plan["output_dir"])/"errors"/f"rank{rank}_{dataset}.json",
                dict(error=str(error),traceback=traceback.format_exc()))
            print(f"[worker {rank}] {dataset} incomplete: {error}; proceeding to next dataset",flush=True)
    storage.write_json(Path(plan["output_dir"])/f"worker_{rank}.json",dict(failures=failures,complete=not failures))
    return 1 if failures else 0


def merge(plan):
    from tools.rsunivlm_paper_worker import aggregate
    root=Path(plan["output_dir"])
    results={}
    table=[]
    world=len(plan["gpu_ids"])
    for entry in plan["entries"]:
        dataset=entry["dataset"]
        all_names=[row["sample"]["name"] for row in entry["samples"]]
        selected=all_names[:plan["max_samples"]] if plan["max_samples"] else all_names
        rows=[]; failures=[]; failed_samples=[]; attempted=0; exhausted=True
        for rank in range(world):
            result=storage.read_json(root/dataset/"shards"/f"rank{rank:03d}"/"metrics.json")
            expected=selected[rank::world]
            if not result or result.get("identity")!=fingerprint([plan["identity"],dataset]):
                failures.append(f"Missing/incompatible shard {rank}"); exhausted=False; continue
            actual=[row["sample"] for row in result.get("rows",[])]
            shard_failed=result.get("failed_samples",[])
            if any(item.get("sample") not in expected for item in shard_failed):
                raise ValueError(f"Foreign failed samples in {dataset} shard {rank}")
            failed_samples.extend(dict(item,rank=rank) for item in shard_failed)
            attempted+=result.get("attempted",len(actual)+len(shard_failed))
            exhausted=exhausted and result.get("exhausted_population",result.get("complete",False))
            if not result.get("complete") or actual!=expected:
                failures.append(f"Shard {rank} incomplete: {result.get('error',actual)}")
            rows.extend(result.get("rows",[]))
        names=[row["sample"] for row in rows]
        if len(set(names))!=len(names) or set(names)-set(selected): raise ValueError(f"Duplicate/foreign samples in {dataset}")
        failed_names=[item["sample"] for item in failed_samples]
        if len(set(failed_names))!=len(failed_names) or set(failed_names)&set(names):
            raise ValueError(f"Duplicated or also-scored failed samples in {dataset}")
        complete=not failures and set(names)==set(selected)
        full_population=complete and len(selected)==len(all_names)==entry["contract"]["expected"] and plan["protocol"]["strict_population"]
        is_native=native.enabled(plan["protocol"].get("semantic_mode"),dataset)
        standard_available=entry["contract"].get("standard_exclusive_metrics_available",True)
        full=full_population and standard_available
        scores=aggregate(dataset,rows,semantic_mode=plan["protocol"].get("semantic_mode","legacy_categorical"))
        result=dict(dataset=dataset,identity=fingerprint([plan["identity"],dataset]),complete=complete,is_full_benchmark=full,
            is_full_query_population=full_population if is_native else None,standard_metrics_available=standard_available,
            expected_samples=entry["contract"]["expected"],discovered_samples=len(all_names),selected_samples=len(selected),
            processed_samples=len(rows),errors=failures,scores=scores,metric_scale="0..1",contract=entry["contract"],
            attempted_samples=attempted,num_failed=len(failed_samples),failed_samples=failed_samples,
            exhausted_population=exhausted,scores_scope="all selected samples" if complete else "successfully decoded complete samples only; partial results",
            note="VISTAR RSUniVLM adapter; includes trained-on LoveDA/UAVid; LEVIR overlap unverified; not an official zero-shot result")
        if dataset!="second" and standard_available:
            import numpy as np
            result["confusion_matrix"]=sum((np.asarray(row["confusion_matrix"],np.int64) for row in rows),
                                          protocol.empty_confusion(dataset)).tolist()
            if dataset=="flair" and not is_native:
                result["confusion_matrix_columns"]=entry["contract"]["classes"]+["background"]
        storage.write_json(root/dataset/"metrics.json",result)
        storage.write_json(root/dataset/"failed_samples.json",dict(num_failed=len(failed_samples),failed_samples=failed_samples))
        results[dataset]=result
        table.append(dict(dataset=dataset,processed=len(rows),attempted=attempted,failed=len(failed_samples),complete=complete,is_full_benchmark=full,
                          primary_metric=entry["contract"]["primary"],value=scores.get(entry["contract"]["primary"]),
                          wfm_3px=scores.get("wfm_3px")))
    summary=dict(complete=all(value["complete"] for value in results.values()),requested_datasets=list(results),
                 standard_metrics_available=all(value["standard_metrics_available"] for value in results.values()),
                 is_full_benchmark=set(results)==set(protocol.DATASETS) and
                     all(value["is_full_benchmark"] for value in results.values()),results=results)
    storage.write_json(root/"paper_summary.json",summary)
    buffer=text_io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    storage.atomic_text(root/"paper_summary.csv",buffer.getvalue())
    return summary


def launch(plan):
    root=Path(plan["output_dir"])
    plan_path=root/"plan.json"
    processes=[]
    def pump(process,gpu):
        with (root/"logs"/f"gpu_{gpu}.log").open("a",encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line); stream.flush()
                print(f"[GPU {gpu}] {line}",end="",flush=True)
    executor=ThreadPoolExecutor(max_workers=len(plan["gpu_ids"]))
    try:
        futures=[]
        for rank,gpu in enumerate(plan["gpu_ids"]):
            env=os.environ.copy()
            for name in ("RANK","WORLD_SIZE","LOCAL_RANK"): env.pop(name,None)
            env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="1")
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker_plan",str(plan_path),"--rank",str(rank)],
                cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace",bufsize=1,start_new_session=True)
            processes.append(process)
            futures.append(executor.submit(pump,process,gpu))
        storage.write_json(root/"runtime.json",dict(owned_pids=[process.pid for process in processes],gpu_ids=plan["gpu_ids"]))
        while any(process.poll() is None for process in processes):
            for future in futures:
                if future.done(): future.result()
            time.sleep(0.5)
        for future in futures: future.result()
    except BaseException:
        for process in processes:
            if process.poll() is None: os.killpg(process.pid,signal.SIGINT)
        for process in processes:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM); process.wait()
        raise
    finally:
        executor.shutdown(wait=True)
    summary=merge(plan)
    storage.write_json(root/"runtime_finished.json",dict(owned_pids=[process.pid for process in processes],
                                                       exit_codes=[process.returncode for process in processes]))
    print(f"[queue] summary={root/'paper_summary.json'}, complete={summary['complete']}",flush=True)
    return 0 if summary["complete"] and all(process.returncode==0 for process in processes) else 1


def main():
    args=parse_args()
    if args.worker_plan: return worker(args.worker_plan,args.rank)
    plan=build_plan(args)
    requested=Path(args.output_dir).expanduser().resolve()
    root,resolution=resolve_output_root(requested,plan["protocol"])
    root.mkdir(parents=True,exist_ok=True)
    with (root/".queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan["output_dir"]=str(root)
        storage.write_json(root/"run_config.json",dict(identity=plan["identity"],protocol=plan["protocol"]))
        storage.write_json(root/"plan.json",plan)
        if resolution: storage.write_json(root/"output_resolution.json",resolution)
        print(f"[queue] Actual OUTPUT_DIR={root}; GPUs={plan['gpu_ids']}; batch/GPU={args.batch_size}",flush=True)
        if args.plan_only:
            print("Preflight passed; no model loaded and no GPU process started.",flush=True)
            return 0
        (root/"logs").mkdir(exist_ok=True)
        return launch(plan)


if __name__=="__main__":
    raise SystemExit(main())
