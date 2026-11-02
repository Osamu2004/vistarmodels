"""Run in WSL: synthetic CPU queue contracts; no pretrained weights or CUDA forward."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker
from tools import eval_rsunivlm_paper_queue as queue
from tools.rsunivlm_batch_runtime import generate_requests


def rle(name): return "\n".join([name+"*24"]*24)


def fixture(tmp_path,count=5,batch=3,world=1):
    samples=[]
    for index in range(count):
        image=tmp_path/f"image{index}.png"; label=tmp_path/f"label{index}.png"
        Image.new("RGB",(32,40),(index,0,0)).save(image)
        Image.fromarray(np.full((40,32),index%7+1,np.uint8)).save(label)
        samples.append(protocol.Sample(f"s{index}",(str(image),),(str(label),)))
    entry=dict(dataset="loveda",contract=protocol.spec("loveda"),
               samples=[dict(sample=sample.serialize(),files=protocol.metadata(sample)) for sample in samples])
    plan=dict(identity="fixture",protocol=dict(batch_size=batch,tile_size=512,format_retries=2,strict_population=False),
              max_samples=0,gpu_ids=[str(index) for index in range(world)],entries=[entry],output_dir=str(tmp_path/"output"))
    return plan,entry,samples


def install_fake(monkeypatch,calls):
    def forward(core,requests,prompt,**kwargs):
        colors=[request[0].getpixel((0,0))[0] for request in requests]
        calls.append(colors)
        return [dict(response=rle(protocol.spec("loveda")["classes"][value%7]),
            generation=dict(inference_batch_size=len(requests),ended_with_eos=True,hit_token_limit=False)) for value in colors]
    monkeypatch.setattr(worker,"generate_requests",forward)
    monkeypatch.setattr(worker.Engine,"core",lambda self:None)
    monkeypatch.setattr(worker,"score_semantic_boundary_wfm",lambda *args,**kwargs:{})


def test_arbitrary_prime_batch_streams_across_images_and_resumes_without_forward(tmp_path,monkeypatch):
    plan,entry,samples=fixture(tmp_path)
    calls=[]; install_fake(monkeypatch,calls)
    result=worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert calls==[[0,1,2],[3,4]] and result["complete"] and result["scores"]["miou"]==1.
    for index,sample in enumerate(samples):
        path=Path(plan["output_dir"])/"loveda/shards/rank000/pred_mask"/f"{sample.name}_mask.png"
        prediction=np.asarray(Image.open(path))
        assert prediction.shape==(40,32) and np.all(prediction==index%7)
    calls.clear()
    resumed=worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert not calls and resumed["scores"]==result["scores"]
    assert all(row["reused_prediction"] for row in resumed["rows"])


def test_two_shards_are_disjoint_cover_every_selected_sample_and_merge_counts(tmp_path,monkeypatch):
    plan,entry,samples=fixture(tmp_path,world=2)
    calls=[]; install_fake(monkeypatch,calls)
    first=worker.Engine(plan,0,2).evaluate(entry,plan["output_dir"])
    second=worker.Engine(plan,1,2).evaluate(entry,plan["output_dir"])
    assert first["assigned_samples"]==["s0","s2","s4"]
    assert second["assigned_samples"]==["s1","s3"]
    assert not set(first["assigned_samples"])&set(second["assigned_samples"])
    merged=queue.merge(plan)
    assert merged["complete"] and not merged["is_full_benchmark"]
    assert merged["results"]["loveda"]["processed_samples"]==5
    assert merged["results"]["loveda"]["scores"]["miou"]==1.


def test_merge_uses_global_confusion_instead_of_averaging_shard_iou(tmp_path):
    names=["a","b"]
    plan=dict(identity="fixture",gpu_ids=["0","1"],max_samples=0,protocol=dict(strict_population=False),
              output_dir=str(tmp_path),entries=[dict(dataset="xbd_pre",contract=protocol.spec("xbd_pre"),
              samples=[dict(sample=dict(name=name)) for name in names])])
    for rank,matrix in enumerate(([[900,0],[0,100]],[[0,1000],[0,0]])):
        storage.write_json(tmp_path/f"xbd_pre/shards/rank{rank:03d}/metrics.json",dict(
            identity=fingerprint(["fixture","xbd_pre"]),complete=True,
            rows=[dict(sample=names[rank],confusion_matrix=matrix)]))
    merged=queue.merge(plan)
    assert merged["results"]["xbd_pre"]["scores"]["building_iou"]==pytest.approx(100/1100)


def test_second_six_class_union_keeps_both_changed_landcover_categories():
    a=np.asarray([[1]],np.uint8); b=np.asarray([[5]],np.uint8)
    counts={str(index):protocol.counts((a==index)|(b==index),(a==index)|(b==index)) for index in range(1,7)}
    scores=protocol.second_metrics(counts)
    assert scores["per_class"]["water"]["tp"]==1
    assert scores["per_class"]["building"]["tp"]==1
    assert scores["miou_c"]==pytest.approx(2/6,abs=1e-6)
    with pytest.raises(ValueError): protocol.decode(dict(response=rle("invented class")),"second",query_key="water")


def test_second_split_is_593_unique_and_unknown_gt_colors_are_rejected(tmp_path):
    names=[line.split("#",1)[0].strip() for line in protocol.SPLIT_FILE.read_text().splitlines()
           if line.split("#",1)[0].strip()]
    assert len(names)==len(set(names))==593
    target=tmp_path/"label.png"
    Image.new("RGB",(1,1),(0,0,255)).save(target)
    assert int(protocol.load_second_label(target)[0,0])==1
    Image.new("RGB",(1,1),(11,12,13)).save(target)
    with pytest.raises(ValueError,match="Unknown SECOND"): protocol.load_second_label(target)


@pytest.mark.parametrize("batch",[3,5,17])
@pytest.mark.parametrize("n_images",[1,2])
def test_any_positive_batch_keeps_one_or_two_images_per_sequence(batch,n_images):
    class Template:
        roles=("user","assistant")
        def append_message(self,*args): pass
        def get_prompt(self): return "same prompt"
    def forward(tokens,**kwargs):
        assert tokens.shape[0]==batch and len(kwargs["modalities"])==batch
        assert len(kwargs["images"])==batch*n_images
        assert [int(value[0,0,0]) for value in kwargs["images"]]==list(range(batch*n_images))
        return torch.tensor([[7,99]]*batch)
    core=SimpleNamespace(device=torch.device("cpu"),dtype=torch.float32,context_len=8192,max_new_tokens=20,
        template=Template(),processor=None,image_token_index=-200,
        tokenizer=SimpleNamespace(eos_token_id=99,batch_decode=lambda rows,**kwargs:["answer"]),
        tokenizer_image_token=lambda *args,**kwargs:torch.tensor([2,*([-200]*n_images),3]),
        process_images=lambda images,*args:torch.stack([torch.full((3,4,4),image.getpixel((0,0))[0]) for image in images]),
        model=SimpleNamespace(config=SimpleNamespace(image_aspect_ratio="square"),generation_config=SimpleNamespace(eos_token_id=99),generate=forward))
    requests=[[Image.new("RGB",(512,512),(index*n_images+image,0,0)) for image in range(n_images)] for index in range(batch)]
    answers=generate_requests(core,requests,protocol.prompt("second",query_key="water") if n_images==2 else protocol.prompt("loveda"))
    assert len(answers)==batch and all(row["generation"]["inference_batch_size"]==batch for row in answers)
    assert all(row["generation"]["images_per_sequence"]==n_images for row in answers)


def test_odd_batch_oom_split_preserves_all_requests(tmp_path,monkeypatch):
    plan,_,_=fixture(tmp_path,batch=5)
    engine=worker.Engine(plan,0,1)
    engine.model=object()
    calls=[]
    def limited(core,requests,prompt):
        if len(requests)>2: raise torch.cuda.OutOfMemoryError("synthetic")
        calls.extend(request[0].getpixel((0,0))[0] for request in requests)
        return [dict(response=rle("water"),generation=dict(inference_batch_size=len(requests))) for _ in requests]
    monkeypatch.setattr(worker,"generate_requests",limited)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    jobs=[dict(identity=str(index),images=[Image.new("RGB",(512,512),(index,0,0))]) for index in range(5)]
    groups=list(engine.forward(jobs,protocol.prompt("loveda")))
    assert calls==list(range(5)) and [len(group) for group,_ in groups]==[2,2,1]
    assert engine.effective_batch==2 and engine.oom_splits==1


def test_model_is_loaded_once_and_reused(tmp_path,monkeypatch):
    import baselines.rsunivlm.model as model_module
    plan,_,_=fixture(tmp_path)
    plan["protocol"]["model_settings"]=dict(source_root="source",model_path="model",dtype="float16",
        attention="eager",max_new_tokens=8192,vision_tower="vision")
    calls=[]; sentinel=object()
    def factory(*args,**kwargs): calls.append(kwargs); return sentinel
    monkeypatch.setattr(model_module,"RSUniVLM",factory)
    engine=worker.Engine(plan,0,1)
    assert engine.core() is sentinel and engine.core() is sentinel and len(calls)==1


def test_dataset_failure_is_recorded_and_the_worker_continues_queue(tmp_path,monkeypatch):
    entries=[dict(dataset=name,samples=[]) for name in ("loveda","levircd")]
    value=dict(batch_size=16,code_sha256={},upstream_code_sha256={},checkpoint={"fixture":True},
        populations={entry["dataset"]:fingerprint([]) for entry in entries},
        model_settings=dict(source_root="source",model_path="model"))
    plan=dict(identity=fingerprint(value),protocol=value,entries=entries,gpu_ids=["0"],max_samples=0,output_dir=str(tmp_path/"out"))
    path=tmp_path/"plan.json"; path.write_text(json.dumps(plan))
    monkeypatch.setattr(queue,"source_identity",lambda:{})
    monkeypatch.setattr(queue,"upstream_files",lambda root:{})
    monkeypatch.setattr(queue,"checkpoint_identity",lambda root:{"fixture":True})
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda index:"synthetic CPU test")
    calls=[]
    def evaluate(self,entry,output):
        calls.append(entry["dataset"])
        if entry["dataset"]=="loveda": raise ValueError("synthetic bad format")
        return dict(complete=True)
    monkeypatch.setattr(worker.Engine,"evaluate",evaluate)
    assert queue.worker(str(path),0)==1
    assert calls==["loveda","levircd"]
    record=storage.read_json(tmp_path/"out/worker_0.json")
    assert not record["complete"] and record["failures"]=={"loveda":"synthetic bad format"}
