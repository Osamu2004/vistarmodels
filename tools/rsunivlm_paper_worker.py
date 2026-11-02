"""One persistent RSUniVLM replica per GPU; stream arbitrary-sized tile batches."""
from __future__ import annotations

from collections import deque
import gc
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint, restore_grid
from baselines.segearth_ov import protocols as seg
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from tools import rsunivlm_paper_protocol as protocol
from tools.rsunivlm_batch_runtime import generate_requests


def outputs(dataset):
    return query_directions(dataset)


def query_directions(dataset):
    return ("forward","reverse") if dataset=="second" else ("forward",) if dataset=="levircd" else ("mask",)


def aggregate(dataset,rows):
    contract=protocol.spec(dataset)
    if dataset=="second":
        counts={str(index):{key:sum(row["class_counts"][str(index)][key] for row in rows)
                for key in ("tp","tn","fp","fn")} for index in range(1,7)}
        return protocol.second_metrics(counts)
    confusion=sum((np.asarray(row["confusion_matrix"],np.int64) for row in rows),
                  np.zeros((len(contract["classes"]),len(contract["classes"])),np.int64))
    scores=(seg.metrics_for_dataset(confusion,dataset) if dataset in seg.DATASET_SPECS else
            seg.metrics_from_confusion(confusion,contract["classes"]))
    if dataset=="levircd":
        scores.update(building_iou=scores["iou_building"],building_f1=scores["f1_building"])
    boundary=[row for row in rows if "wfm_3px_mean_present" in row]
    if boundary: scores.update(aggregate_binary_boundary_wfm(boundary))
    return scores


class Engine:
    def __init__(self,plan,rank,world_size):
        self.plan,self.rank,self.world_size=plan,rank,world_size
        self.model=None
        self.effective_batch=plan["protocol"]["batch_size"]
        if self.effective_batch<=0: raise ValueError("Batch size must be a positive integer")
        self.oom_splits=0
        self.actual_batches=set()

    def core(self):
        if self.model is None:
            from baselines.rsunivlm.model import RSUniVLM
            settings=self.plan["protocol"]["model_settings"]
            self.model=RSUniVLM(Path(settings["source_root"]),Path(settings["model_path"]),device="cuda:0",
                dtype=settings["dtype"],attention=settings["attention"],max_new_tokens=settings["max_new_tokens"],
                vision_tower=settings["vision_tower"])
        return self.model

    def forward(self,jobs,prompt):
        offset=0
        while offset<len(jobs):
            count=min(self.effective_batch,len(jobs)-offset)
            group=jobs[offset:offset+count]
            oom=False
            try:
                answers=generate_requests(self.core(),[job["images"] for job in group],prompt)
            except torch.cuda.OutOfMemoryError:
                oom=True
            if oom:
                if count==1: raise RuntimeError("GPU memory insufficient even at batch1")
                gc.collect(); torch.cuda.empty_cache()
                self.effective_batch=max(1,count//2); self.oom_splits+=1
                print(f"[paper queue] OOM: {count} -> {self.effective_batch}",flush=True)
                continue
            if len(answers)!=len(group): raise ValueError("Response/request population differs")
            members=[job["identity"] for job in group]
            for position,answer in enumerate(answers):
                answer["generation"].update(batch_members=members,batch_position=position)
            self.actual_batches.add(count)
            print(f"[paper queue] completed actual_batch={count}, images_per_sequence={len(group[0]['images'])}",flush=True)
            yield group,answers
            offset+=count

    def dispatch(self,jobs,dataset):
        budget=self.plan["protocol"]["format_retries"]
        try:
            for variant in range(budget+1):
                pending=[job for job in jobs if not job.get("done") and len(job["attempts"])==variant]
                prompt=protocol.prompt(dataset,variant)
                for group,answers in self.forward(pending,prompt):
                    for job,answer in zip(group,answers):
                        attempt=dict(prompt=prompt,prompt_variant=variant,**answer)
                        job["attempts"].append(attempt)
                        record=dict(identity=job["identity"],spatial=job["spatial"],direction=job["direction"],
                            dataset=dataset,attempts=job["attempts"],**attempt)
                        io.write_json(job["path"],record)
                        try:
                            grid=protocol.decode(record,dataset)
                        except (ValueError,KeyError) as error:
                            attempt["decode_error"]=str(error)
                            io.write_json(job["path"],dict(record,decode_error=str(error)))
                            print(f"[paper format] {job['path'].name}: {error}; attempt={variant+1}/{budget+1}",flush=True)
                        else:
                            io.write_json(job["path"],dict(record,grid_shape=list(grid.shape)))
                            self.apply(job,grid)
            for job in jobs:
                if not job.get("done"):
                    job["case"]["error"]=f"Format budget exhausted: {job['path']}"
        finally:
            for job in jobs:
                for image in job["images"]: image.close()

    @staticmethod
    def apply(job,grid):
        if job.get("done"): raise ValueError("Duplicate tile completion")
        x,y,w,h=job["box"]
        restored=restore_grid(grid,(job["tile_size"],job["tile_size"]))
        job["case"]["predictions"][job["direction"]][y:y+h,x:x+w]=restored[:h,:w]
        job["case"]["remaining"]-=1
        job["case"]["shapes"].append(list(grid.shape))
        job["done"]=True

    def evaluate(self,entry,output):
        dataset=entry["dataset"]
        contract=protocol.spec(dataset)
        size=self.plan["protocol"]["tile_size"]
        budget=self.plan["protocol"]["format_retries"]
        full=[protocol.Sample.restore(row["sample"]) for row in entry["samples"]]
        selected=full[:self.plan["max_samples"]] if self.plan["max_samples"] else full
        assigned=selected[self.rank::self.world_size]
        expected_meta={row["sample"]["name"]:row["files"] for row in entry["samples"]}
        root=Path(output)/dataset/"shards"/f"rank{self.rank:03d}"
        for folder in ("pred_mask","raw_text","receipts","errors"): (root/folder).mkdir(parents=True,exist_ok=True)
        dataset_id=fingerprint([self.plan["identity"],dataset])
        previous=io.read_json(root/"run_config.json")
        if previous and previous.get("identity")!=dataset_id: raise ValueError("Shard has another protocol")
        io.write_json(root/"run_config.json",dict(identity=dataset_id,rank=self.rank,world_size=self.world_size,
            assigned_samples=[sample.name for sample in assigned],contract=contract))
        states,pending,rows=deque(),[],[]
        progress=tqdm(total=len(assigned),desc=f"GPU{self.rank} {dataset}",unit="image")

        def finish(case):
            sample=case["sample"]
            if protocol.metadata(sample)!=expected_meta[sample.name]: raise ValueError(f"Inputs changed: {sample.name}")
            predictions=case["predictions"]
            checksums={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in predictions.items()}
            if not case["cached"]:
                for key,value in predictions.items(): io.save_prediction(root/"pred_mask"/f"{sample.name}_{key}.png",value)
                io.write_json(root/"receipts"/f"{sample.name}.json",dict(identity=case["identity"],sha256=checksums,
                    shape=list(case["shape"]),descriptor_shapes=case["shapes"]))
            # GT pixels are read only after every prediction for this sample exists.
            score_start=time.monotonic()
            targets=protocol.load_targets(sample,dataset,case["shape"])
            row=dict(sample=sample.name,reused_prediction=case["cached"],descriptor_shapes=case["shapes"])
            if dataset=="second":
                row["class_counts"]={str(index):protocol.counts(
                    (predictions["forward"]==index)|(predictions["reverse"]==index),
                    (targets[0]==index)|(targets[1]==index)) for index in range(1,7)}
                view=protocol.second_metrics(row["class_counts"])
            else:
                prediction=predictions["forward" if dataset=="levircd" else "mask"]
                confusion=seg.confusion_matrix(prediction,targets[0],num_classes=len(contract["classes"]),ignore_index=255)
                row["confusion_matrix"]=confusion.tolist()
                if dataset in seg.DATASET_SPECS and np.any(targets[0]!=255):
                    row.update(score_semantic_boundary_wfm(prediction,targets[0],num_classes=len(contract["classes"])))
                view=aggregate(dataset,[row])
            row["scoring_seconds"]=time.monotonic()-score_start
            rows.append(row)
            with (root/"progress.jsonl").open("a") as stream:
                import json
                stream.write(json.dumps(row,allow_nan=False)+"\n")
            progress.update(1)
            primary=view.get(contract["primary"])
            progress.set_postfix(batch=self.effective_batch,score="n/a" if primary is None else f"{primary:.4f}",cached=int(case["cached"]))
            io.write_json(root/"status.json",dict(processed=len(rows),assigned=len(assigned),last_sample=sample.name,
                                               effective_batch=self.effective_batch,complete=False))

        def drain():
            while states and (states[0].get("error") or states[0]["remaining"]==0):
                case=states.popleft()
                if case.get("error"): raise ValueError(case["error"])
                finish(case)

        try:
            for sample in assigned:
                if protocol.metadata(sample)!=expected_meta[sample.name]: raise ValueError(f"Inputs changed: {sample.name}")
                images=protocol.load_images(sample,dataset)
                try:
                    width,height=images[0].size
                    identity=fingerprint([dataset_id,sample.serialize(),expected_meta[sample.name]])
                    receipt=io.read_json(root/"receipts"/f"{sample.name}.json")
                    cached={}
                    if receipt and receipt.get("identity")==identity:
                        for key in outputs(dataset):
                            try:
                                with Image.open(root/"pred_mask"/f"{sample.name}_{key}.png") as opened:
                                    value=np.asarray(opened).copy()
                                if value.shape!=(height,width) or not np.all(value<len(contract["classes"])) or value.dtype!=np.uint8:
                                    raise ValueError("Invalid cached class mask")
                                if hashlib.sha256(value.tobytes()).hexdigest()!=receipt["sha256"][key]: raise ValueError("Mask hash changed")
                                cached[key]=value
                            except (OSError,KeyError,ValueError):
                                cached={}; break
                    directions=query_directions(dataset)
                    tiles=((height+size-1)//size)*((width+size-1)//size)
                    case=dict(sample=sample,identity=identity,shape=(height,width),cached=bool(cached),
                        remaining=0 if cached else tiles*len(directions),
                        shapes=receipt.get("descriptor_shapes",[]) if cached else [],
                        predictions=cached or {direction:np.empty((height,width),np.uint8) for direction in directions})
                    states.append(case)
                    if not cached:
                        for direction in directions:
                            source=images if direction!="reverse" else images[::-1]
                            for y in range(0,height,size):
                                for x in range(0,width,size):
                                    w,h=min(size,width-x),min(size,height-y)
                                    spatial=dict(source_box=[x,y,x+w,y+h],extent=[width,height],tile_size=size,padding="black_right_bottom")
                                    job=dict(case=case,direction=direction,box=(x,y,w,h),spatial=spatial,tile_size=size,
                                        path=root/"raw_text"/f"{sample.name}_{direction}_y{y}_x{x}.json",
                                        identity=fingerprint([identity,direction,spatial]),attempts=[])
                                    record=io.read_json(job["path"])
                                    if record and record.get("identity")==job["identity"]:
                                        job["attempts"]=record.get("attempts",[])
                                        for attempt in job["attempts"]:
                                            try: grid=protocol.decode(attempt,dataset)
                                            except (ValueError,KeyError): continue
                                            self.apply(job,grid); break
                                        if job.get("done"): continue
                                        if len(job["attempts"])>=budget+1:
                                            case["error"]=f"Stored format budget exhausted: {job['path']}"; continue
                                    job["images"]=[]
                                    for image in source:
                                        tile=Image.new("RGB",(size,size),(0,0,0))
                                        tile.paste(image.crop((x,y,x+w,y+h)),(0,0))
                                        job["images"].append(tile)
                                    pending.append(job)
                                    if len(pending)>=self.effective_batch:
                                        self.dispatch(pending,dataset); pending=[]; drain()
                    if (cached or case["remaining"]==0 or case.get("error")) and pending:
                        self.dispatch(pending,dataset); pending=[]
                    drain()
                finally:
                    for image in images: image.close()
            if pending: self.dispatch(pending,dataset); pending=[]
            drain()
            if states or len(rows)!=len(assigned): raise ValueError("Incomplete assigned population")
            result=dict(identity=dataset_id,dataset=dataset,rank=self.rank,world_size=self.world_size,
                complete=True,is_full_benchmark=False,scope="rank shard",processed=len(rows),
                assigned_samples=[sample.name for sample in assigned],scores=aggregate(dataset,rows),
                query_directions=list(query_directions(dataset)),
                effective_batch=self.effective_batch,actual_batch_sizes_this_process=sorted(self.actual_batches),
                oom_splits=self.oom_splits,rows=rows)
            io.write_json(root/"metrics.json",result)
            return result
        except Exception as error:
            io.write_json(root/"metrics.json",dict(identity=dataset_id,dataset=dataset,rank=self.rank,
                complete=False,is_full_benchmark=False,processed=len(rows),rows=rows,error=str(error),traceback=traceback.format_exc()))
            raise
        finally:
            progress.close()
            for job in pending:
                for image in job.get("images",[]): image.close()
