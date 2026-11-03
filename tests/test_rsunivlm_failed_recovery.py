"""WSL-only recovery checks using fake answers; never load pretrained weights."""
import copy
import errno
import fcntl
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from tools import eval_rsunivlm_paper_queue as queue
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def completed_source(tmp_path, monkeypatch, dataset="flair", world=2):
    source=tmp_path/"original"; source.mkdir()
    samples=[]
    for index in range(4):
        image=tmp_path/f"image{index}.png"; label=tmp_path/f"label{index}.png"
        Image.new("RGB",(32,40),(index,0,0)).save(image)
        Image.fromarray(np.zeros((40,32),np.uint8)).save(label)
        samples.append(protocol.Sample(f"s{index}",(str(image),)*(2 if dataset=="second" else 1),
                                       (str(label),)*(2 if dataset=="second" else 1)))
    entry=dict(dataset=dataset,contract=native.contract(dataset) if dataset=="flair" else protocol.spec(dataset),
        samples=[dict(sample=s.serialize(),files=protocol.metadata(s)) for s in samples])
    entry["contract"]["expected"]=4
    value=dict(batch_size=16,tile_size=512,format_retries=0,strict_population=False,semantic_mode=native.MODE,
        world_size=world,datasets=[dataset],code_sha256=queue.source_identity(),populations={dataset:fingerprint(entry["samples"])},
        prediction_background={},native_semantic={dataset:native.settings(dataset,0)} if dataset=="flair" else {},
        prompts={},second_query={} if dataset=="second" else None,levircd_query=None)
    plan=dict(protocol=value,entries=[entry],gpu_ids=[str(i) for i in range(world)],max_samples=0,output_dir=str(source))
    value["prompts"][dataset]={key:[recovery.instruction(plan,dataset,key,0)] for key in recovery.keys(plan,dataset)}
    plan["identity"]=fingerprint(value)
    calls=[]; active_root=[source]; failures=[True]
    monkeypatch.setattr(worker.Engine,"core",lambda self:None)
    def images(sample,dataset):
        result=[]
        for path in sample.images:
            with Image.open(path) as opened: result.append(opened.convert("RGB"))
        return result
    monkeypatch.setattr(protocol,"load_images",images)
    bad_key=recovery.keys(plan,dataset)[0]
    def generate(core,requests,prompt):
        key,variant=next((key,v) for key in recovery.keys(plan,dataset) for v in range(3)
                         if recovery.instruction(plan,dataset,key,v)==prompt)
        ids=[request[0].getpixel((0,0))[0] for request in requests]
        calls.append((key,variant,ids))
        label=("building" if dataset=="flair" else "water") if key==bad_key else "others"
        return [dict(response="building:1" if failures[0] and key==bad_key and index==0 else "\n".join([label+"*24"]*23),
                     generation=dict(ended_with_eos=True,hit_token_limit=False)) for index in ids]
    monkeypatch.setattr(worker,"generate_requests",generate)
    def targets(sample,dataset,shape):
        folder=active_root[0]/dataset/"shards"/f"rank{int(sample.name[1:])%world:03d}"/"pred_mask"
        assert all((folder/f"{sample.name}_{key}.png").is_file() for key in recovery.keys(plan,dataset))
        return [np.zeros(shape,np.uint8)]*(2 if dataset=="second" else 1)
    monkeypatch.setattr(protocol,"load_targets",targets)
    for rank in range(world): worker.Engine(plan,rank,world).evaluate(entry,source)
    assert not queue.merge(plan)["complete"]
    io.write_json(source/"plan.json",plan)
    io.write_json(source/"run_config.json",dict(identity=plan["identity"],protocol=value))
    io.write_json(source/"runtime_finished.json",dict(exit_codes=[1]+[0]*(world-1)))
    calls.clear()
    return source,plan,entry,calls,active_root,failures


@pytest.mark.parametrize("dataset",["flair","second"])
def test_recover_only_bad_query_reuse_successes_combine_counts_and_resume(tmp_path,monkeypatch,dataset):
    source,plan,entry,calls,active_root,failures=completed_source(tmp_path,monkeypatch,dataset)
    original={str(path.relative_to(source)):path.read_bytes() for path in source.rglob("*") if path.is_file()}
    old,new,report=recovery.prepare(source,[],"1,0")
    assert new["identity"]!=old["identity"] and new["protocol"]["format_retries"]==2
    assert new["gpu_ids"]==["1","0"] and report["source_failed_counts"]=={dataset:1}
    assert new["protocol"]["code_sha256"]==old["protocol"]["code_sha256"]
    output=tmp_path/"recovered"; new["output_dir"]=str(output); active_root[0]=output
    counts=recovery.import_predictions(source,output,old,new)
    assert counts["reused_complete_samples"]==3 and counts["imported_failed_query_histories"]==1
    assert counts["reused_valid_query_answers"]==len(recovery.keys(new,dataset))-1
    assert counts["missing_queries"]==0 and counts["hardlinked_masks"]==3*len(recovery.keys(new,dataset))
    failures[0]=False
    for rank in range(2): worker.Engine(new,rank,2).evaluate(entry,output)
    assert calls==[(recovery.keys(new,dataset)[0],1,[0])]
    result=queue.merge(new)
    assert result["complete"] and result["results"][dataset]["processed_samples"]==4
    rows=[row for rank in range(2) for row in io.read_json(output/dataset/"shards"/f"rank{rank:03d}"/"metrics.json")["rows"]]
    assert result["results"][dataset]["scores"]==worker.aggregate(dataset,rows,semantic_mode=native.MODE)
    path=output/dataset/"shards/rank000/raw_text"/f"s0_{recovery.keys(new,dataset)[0]}_y0_x0.json"
    attempts=io.read_json(path)["attempts"]
    assert [a["prompt_variant"] for a in attempts]==[0,1]
    assert "reused_from" in attempts[0] and "reused_from" not in attempts[1]
    assert {str(path.relative_to(source)):path.read_bytes() for path in source.rglob("*") if path.is_file()}==original
    calls.clear()
    recovery.import_predictions(source,output,old,new)
    for rank in range(2): worker.Engine(new,rank,2).evaluate(entry,output)
    assert not calls and queue.merge(new)["complete"]


def test_remaining_templates_exhaust_once_and_leave_failed_samples_partial(tmp_path,monkeypatch):
    source,plan,entry,calls,active_root,_=completed_source(tmp_path,monkeypatch)
    old,new,_=recovery.prepare(source,[])
    output=tmp_path/"recovered"; new["output_dir"]=str(output); active_root[0]=output
    recovery.import_predictions(source,output,old,new)
    for rank in range(2): worker.Engine(new,rank,2).evaluate(entry,output)
    key=recovery.keys(new,"flair")[0]
    assert calls==[(key,1,[0]),(key,2,[0])]
    result=queue.merge(new)["results"]["flair"]
    assert result["num_failed"]==1 and result["processed_samples"]==3 and not result["complete"]
    assert not list((output/"flair/shards/rank000/pred_mask").glob("s0_*.png"))
    calls.clear(); recovery.import_predictions(source,output,old,new)
    for rank in range(2): worker.Engine(new,rank,2).evaluate(entry,output)
    assert not calls


def test_preflight_protects_unfinished_source_and_numerical_identity(tmp_path,monkeypatch):
    source,plan,*_=completed_source(tmp_path,monkeypatch)
    finished=(source/"runtime_finished.json").read_bytes()
    (source/"runtime_finished.json").unlink()
    with pytest.raises(ValueError,match="whole queue"): recovery.prepare(source,[])
    (source/"runtime_finished.json").write_bytes(finished)
    monkeypatch.setattr(queue,"source_identity",lambda:{"changed":"code"})
    with pytest.raises(ValueError,match="Numerical source changed"): recovery.prepare(source,[])


def test_import_rejects_changed_inputs_and_raw_answer_identity(tmp_path,monkeypatch):
    source,plan,*_=completed_source(tmp_path,monkeypatch)
    old,new,_=recovery.prepare(source,[])
    item=new["entries"][0]["samples"][0]
    item["files"][0]["size"]+=1
    with pytest.raises(ValueError,match="Source inputs changed"):
        recovery.import_predictions(source,tmp_path/"changed",old,new)
    item["files"][0]["size"]-=1
    key=recovery.keys(new,"flair")[0]
    path=source/"flair/shards/rank000/raw_text"/f"s0_{key}_y0_x0.json"
    raw=io.read_json(path); raw["identity"]="tampered"; io.write_json(path,raw)
    with pytest.raises(ValueError,match="query identity differs"):
        recovery.import_predictions(source,tmp_path/"bad_raw",old,new)


def test_active_source_lock_stops_before_any_plan_or_model(tmp_path,monkeypatch):
    source=tmp_path/"active"; source.mkdir()
    monkeypatch.setattr(sys,"argv",["retry","--source_run",str(source)])
    with (source/".queue.lock").open("a") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit,match="still running"): recovery.main()


def test_plan_only_does_not_import_or_launch_and_incomplete_shards_are_not_recovered(tmp_path,monkeypatch):
    source,plan,*_=completed_source(tmp_path,monkeypatch)
    output=tmp_path/"plan_only"
    monkeypatch.setattr(sys,"argv",["retry","--source_run",str(source),"--output_dir",str(output),"--plan_only"])
    def forbidden(*args,**kwargs): raise AssertionError("Plan only must not import/launch")
    monkeypatch.setattr(recovery,"import_predictions",forbidden); monkeypatch.setattr(queue,"launch",forbidden)
    monkeypatch.setattr(recovery,"verify_runtime_assets",lambda plan:None)
    assert recovery.main()==0 and not output.exists()
    path=source/"flair/shards/rank000/metrics.json"
    result=io.read_json(path); result["exhausted_population"]=False; io.write_json(path,result)
    with pytest.raises(ValueError,match="non-format error"): recovery.prepare(source,["flair"])


def test_mask_copy_fallback_preserves_bytes_on_different_filesystem(tmp_path,monkeypatch):
    source=tmp_path/"source.png"; destination=tmp_path/"copy.png"
    Image.new("L",(3,4),1).save(source)
    def exdev(*args): raise OSError(errno.EXDEV,"Different filesystem")
    monkeypatch.setattr(recovery.os,"link",exdev)
    assert recovery.link_or_copy(source,destination)=="copy"
    assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(destination.read_bytes()).digest()


def test_shard_accounting_and_original_retry_budget_are_validated(tmp_path,monkeypatch):
    source,plan,*_=completed_source(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="increase"): recovery.prepare(source,[],retries=0)
    with pytest.raises(ValueError,match="GPU IDs"): recovery.prepare(source,[],gpu_ids="0")
    path=source/"flair/shards/rank000/metrics.json"
    result=io.read_json(path); result["rows"].append(copy.deepcopy(result["rows"][0])); io.write_json(path,result)
    with pytest.raises(ValueError,match="accounting differs"): recovery.prepare(source,[])


def test_runtime_assets_check_vision_identity_without_loading_weights(tmp_path,monkeypatch):
    vision=tmp_path/"vision"; vision.mkdir()
    config=vision/"config.json"; config.write_text('{"hidden_size": 8}\n')
    item=dict(path=str(config.resolve()),size=config.stat().st_size,mtime_ns=config.stat().st_mtime_ns,
              sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    plan=dict(protocol=dict(model_settings=dict(source_root="source",model_path="checkpoint",vision_tower=str(vision)),
        upstream_code_sha256={"source":"same"},checkpoint={"checkpoint":"same"},vision_files=[item]))
    monkeypatch.setattr(queue,"upstream_files",lambda path:{"source":"same"})
    monkeypatch.setattr(recovery,"checkpoint_identity",lambda path:{"checkpoint":"same"})
    recovery.verify_runtime_assets(plan)
    config.write_text('{"hidden_size": 16}\n')
    with pytest.raises(ValueError,match="Vision checkpoint files changed"):
        recovery.verify_runtime_assets(plan)
