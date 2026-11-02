"""Run in WSL only: paper-query contracts and official-demo decoder equivalence."""
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from tools import eval_rsunivlm_paper_queue as queue
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def fixture(tmp_path, dataset="flair", count=4, world=2, batch=3):
    tmp_path.mkdir(parents=True,exist_ok=True)
    samples=[]
    for index in range(count):
        image=tmp_path/f"image{index}.png"; label=tmp_path/f"label{index}.png"
        Image.new("RGB",(32,40),(index,0,0)).save(image)
        target=np.full((40,32),index%2,np.uint8); target[0,0]=255
        Image.fromarray(target).save(label)
        samples.append(protocol.Sample(f"s{index}",(str(image),),(str(label),)))
    entry=dict(dataset=dataset,contract=native.contract(dataset),samples=[dict(sample=s.serialize(),files=protocol.metadata(s)) for s in samples])
    value=dict(batch_size=batch,tile_size=512,format_retries=0,strict_population=False,semantic_mode=native.MODE)
    plan=dict(identity=fingerprint(value),protocol=value,entries=[entry],gpu_ids=[str(i) for i in range(world)],max_samples=0,output_dir=str(tmp_path/"out"))
    return plan,entry,samples


def fake_runtime(monkeypatch,calls):
    monkeypatch.setattr(worker.Engine,"core",lambda self:None)
    def load_images(sample,dataset):
        with Image.open(sample.images[0]) as opened: return [opened.convert("RGB")]
    monkeypatch.setattr(protocol,"load_images",load_images)
    def generate(core,requests,prompt):
        calls.append((prompt,[request[0].getpixel((0,0))[0] for request in requests]))
        # Two independently queried foreground masks overlap everywhere.
        foreground=any(f"the {name} in" in prompt for name in ("building","pervious surface"))
        label="unexpected native foreground name" if foreground else "others"
        return [dict(response="\n".join([label+"*24"]*23),generation=dict(ended_with_eos=True,hit_token_limit=False)) for _ in requests]
    monkeypatch.setattr(worker,"generate_requests",generate)


def test_native_class_queries_and_prompts_are_fixed_without_gt_selection():
    assert [len(native.queries(name)) for name in ("loveda","flair","uavid","xbd_pre","chn6_cug")]==[7,12,8,1,1]
    key,index,name=native.queries("flair")[0]
    assert (index,name)==(0,"building")
    assert native.prompt("flair",key)=="[SEG] Can you segment the building in this image?"
    assert native.prompt("flair",key,1)=="[SEG] Please segment the building in this image."
    assert native.contract("flair")["primary"]=="miou"
    assert not native.contract("flair")["standard_exclusive_metrics_available"]
    assert native.contract("xbd_pre")["standard_exclusive_metrics_available"]


def test_official_binary_rule_and_format_errors_remain_distinct():
    grid,labels=native.decode_binary(dict(response="others*1, unknown label*1, background*1\nwater*1, others*2"))
    assert grid.tolist()==[[0,1,1],[1,0,0]]
    assert labels["unknown label"]==1
    with pytest.raises(ValueError,match="Invalid RLE"):
        native.decode_binary(dict(response="water"))
    with pytest.raises(ValueError,match="Ragged"):
        native.decode_binary(dict(response="others*24\nwater*23"))
    with pytest.raises(ValueError,match="truncated"):
        native.decode_binary(dict(response="others*24",generation=dict(hit_token_limit=True,ended_with_eos=False)))


def test_decoder_and_restoration_match_actual_upstream_demo_when_available():
    roots=[Path(__file__).resolve().parents[1]/"third_party"/name for name in ("RSUniVLM_clean_66dce1d","RSUniVLM")]
    path=next((root/"app.py" for root in roots if (root/"app.py").is_file()),None)
    if path is None: pytest.skip("Official source is needed for source-backed decoder equivalence")
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="parse_output_string")
    namespace=dict(np=np,re=re,cv2=cv2)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),namespace)
    for height,width,size in ((23,24,512),(24,24,512),(3,7,511)):
        response="\n".join([f"others*1, queried category*{width-1}"]*height)
        expected=namespace["parse_output_string"](response,(size,size),20)
        assert expected is not None
        grid,_=native.decode_binary(dict(response=response))
        np.testing.assert_array_equal(native.restore(grid,size),expected[:,:,0]//255)


def test_two_rank_queries_keep_overlap_save_before_gt_and_resume_without_forward(tmp_path,monkeypatch):
    plan,entry,samples=fixture(tmp_path)
    entry["contract"]["expected"]=4; plan["protocol"]["strict_population"]=True
    plan["identity"]=fingerprint(plan["protocol"])
    calls=[]; fake_runtime(monkeypatch,calls)
    def load_targets(sample,dataset,shape):
        rank=int(sample.name[1:])%2
        folder=Path(plan["output_dir"])/dataset/"shards"/f"rank{rank:03d}"/"pred_mask"
        assert all((folder/f"{sample.name}_{key}.png").is_file() for key,_,_ in native.queries(dataset))
        with Image.open(sample.labels[0]) as opened: return [np.asarray(opened).copy()]
    monkeypatch.setattr(protocol,"load_targets",load_targets)
    first=worker.Engine(plan,0,2).evaluate(entry,plan["output_dir"])
    second=worker.Engine(plan,1,2).evaluate(entry,plan["output_dir"])
    assert first["assigned_samples"]==["s0","s2"] and second["assigned_samples"]==["s1","s3"]
    assert len(calls)==24 and all(len(ids)==2 for _,ids in calls)
    assert all(row["overlapping_query_pixels"]==1279 for row in first["rows"]+second["rows"])
    merged=queue.merge(plan)
    result=merged["results"]["flair"]
    assert merged["complete"] and result["processed_samples"]==4
    assert result["is_full_query_population"]
    assert not result["is_full_benchmark"] and not result["standard_metrics_available"]
    assert result["scores"]["miou"]==result["scores"]["query_miou"]
    assert not result["scores"]["table7_protocol_verified"]
    assert "confusion_matrix" not in result and "wfm_3px" not in result["scores"]
    values=result["scores"]["per_query"]
    assert values[native.queries("flair")[0][0]]["iou"]==0.5
    assert values[native.queries("flair")[1][0]]["iou"]==0.5
    assert result["scores"]["query_valid_pixels"]==4*1279
    calls.clear()
    resumed=worker.Engine(plan,0,2).evaluate(entry,plan["output_dir"])
    assert not calls and all(row["reused_prediction"] for row in resumed["rows"])
    assert resumed["scores"]==first["scores"]


def test_arbitrary_batch_across_images_and_valid_raw_text_recovers_lost_png(tmp_path,monkeypatch):
    plan,entry,_=fixture(tmp_path,count=5,world=1,batch=3)
    calls=[]; fake_runtime(monkeypatch,calls)
    monkeypatch.setattr(protocol,"load_targets",lambda sample,*args:[np.zeros((40,32),np.uint8)])
    worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    groups=[ids for prompt,ids in calls if "the building in" in prompt]
    assert groups==[[0,1,2],[3,4]]
    folder=Path(plan["output_dir"])/"flair/shards/rank000"
    (folder/"pred_mask"/f"s0_{native.queries('flair')[0][0]}.png").unlink()
    calls.clear()
    worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert not calls  # Reconstruct only from this protocol's saved class queries.


def test_binary_benchmarks_retain_standard_metric_and_whole_population_flags(tmp_path,monkeypatch):
    plan,entry,_=fixture(tmp_path,dataset="xbd_pre",count=1,world=1)
    entry["contract"]["expected"]=1; plan["protocol"]["strict_population"]=True
    calls=[]; fake_runtime(monkeypatch,calls)
    monkeypatch.setattr(protocol,"load_targets",lambda *args:[np.ones((40,32),np.uint8)])
    worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    result=queue.merge(plan)["results"]["xbd_pre"]
    assert result["complete"] and result["is_full_benchmark"] and result["is_full_query_population"]
    assert result["scores"]["building_iou"]==result["scores"]["query_miou"]==1.
    assert np.asarray(result["confusion_matrix"]).shape==(2,2)
    assert len(calls)==1


def test_class_mean_pools_counts_instead_of_averaging_image_or_rank_scores():
    rows=[]
    for size,correct in ((900,True),(100,False)):
        counts={key:dict(tp=0,tn=size,fp=0,fn=0) for key,_,_ in native.queries("flair")}
        first=native.queries("flair")[0][0]
        counts[first]=dict(tp=size if correct else 0,tn=0,fp=0,fn=0 if correct else size)
        rows.append(dict(native_semantic_query=True,query_counts=counts,query_valid_pixels=size,
            zero_positive_query_pixels=0 if correct else size,overlapping_query_pixels=0))
    result=worker.aggregate("flair",rows,semantic_mode=native.MODE)
    assert result["miou"]==0.9 and result["query_miou"]==0.9
    assert result["query_valid_pixels"]==1000


def test_cli_defaults_select_paper_queries_and_zero_format_retries(monkeypatch):
    monkeypatch.setattr("sys.argv",["queue"])
    args=queue.parse_args()
    assert args.semantic_mode==native.MODE and args.format_retries==0 and args.max_new_tokens==32000
    assert "tools/rsunivlm_native_semantic.py" in queue.source_identity()
