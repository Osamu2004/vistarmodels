"""WSL-only contracts for the user-requested FLAIR background interpretation."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from baselines.binary_boundary_wfm import score_semantic_boundary_wfm
from baselines.segearth_ov import protocols as seg
from tools import eval_rsunivlm_paper_queue as queue
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def test_flair_others_is_background_and_building_keeps_its_id():
    response="others *1, BUILDING *1, Background *1, other *1\nwater*1, others*3"
    result=protocol.decode(dict(response=response),"flair")
    assert result.tolist()==[[12,0,12,12],[4,12,12,12]]
    assert len(protocol.spec("flair")["classes"])==12
    assert protocol.spec("flair")["classes"][0]=="building"
    with pytest.raises(ValueError,match="Unknown flair"):
        protocol.decode(dict(response="invented label*24"),"flair")


def test_background_false_negatives_remain_in_the_denominator():
    target=np.asarray([[0,0,4,4,255]],np.uint8)
    prediction=np.asarray([[0,12,4,12,0]],np.uint8)
    matrix=seg.confusion_matrix(prediction,target,num_classes=12,ignore_index=255,prediction_ignore_index=12)
    scores=worker.aggregate("flair",[dict(confusion_matrix=matrix.tolist())])
    assert matrix.shape==(12,13) and matrix.sum()==4
    assert matrix[0,12]==matrix[4,12]==1
    assert scores["iou_building"]==scores["iou_water"]==0.5
    assert scores["miou"]==scores["pixel_accuracy"]==0.5
    assert scores["valid_pixels"]==4
    assert scores["background_pixels_on_valid_gt"]==2
    assert scores["background_rate_on_valid_gt"]==0.5
    assert "iou_background" not in scores and "rejected_pixels" not in scores


def test_background_participates_in_boundaries_instead_of_becoming_ignore():
    target=np.zeros((8,8),np.uint8)
    prediction=target.copy(); prediction[:,4:]=12
    scores=score_semantic_boundary_wfm(prediction,target,num_classes=13,ignore_margin=0)
    assert scores["wfm_3px_valid_pixels"]==64
    assert scores["wfm_3px_gt_ignore_pixels"]==0
    assert scores["wfm_3px_pred_rejected_pixels"]==0
    assert scores["wfm_3px_pred_edge_pixels"]>0


def test_worker_saves_background_resumes_and_merges_twelve_by_thirteen(tmp_path,monkeypatch):
    from baselines.rsunivlm.protocols import fingerprint

    image=tmp_path/"image.png"; label=tmp_path/"label.png"
    Image.new("RGB",(32,40)).save(image)
    target=np.zeros((40,32),np.uint8); target[0,0]=255
    Image.fromarray(target).save(label)
    sample=protocol.Sample("flair_fixture",(str(image),),(str(label),))
    entry=dict(dataset="flair",contract=protocol.spec("flair"),samples=[dict(sample=sample.serialize(),files=protocol.metadata(sample))])
    value=dict(batch_size=16,tile_size=512,format_retries=2,strict_population=False)
    plan=dict(identity=fingerprint(value),protocol=value,max_samples=0,gpu_ids=["0"],entries=[entry],output_dir=str(tmp_path/"out"))
    monkeypatch.setattr(protocol,"load_images",lambda *args:[Image.new("RGB",(32,40))])
    monkeypatch.setattr(protocol,"load_targets",lambda *args:[target.copy()])
    monkeypatch.setattr(worker.Engine,"core",lambda self:None)
    calls=[]
    def generate(core,requests,prompt):
        calls.append(len(requests))
        return [dict(response="\n".join(["others*24"]*24),generation=dict(ended_with_eos=True,hit_token_limit=False)) for _ in requests]
    monkeypatch.setattr(worker,"generate_requests",generate)
    result=worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert result["complete"] and calls==[1]
    assert result["scores"]["valid_pixels"]==1279
    assert result["scores"]["background_pixels_on_valid_gt"]==1279
    assert result["scores"]["miou"]==0.0
    path=Path(plan["output_dir"])/"flair/shards/rank000/pred_mask/flair_fixture_mask.png"
    with Image.open(path) as opened:
        assert np.all(np.asarray(opened)==12)
    calls.clear()
    resumed=worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert not calls and resumed["rows"][0]["reused_prediction"]
    assert resumed["scores"]==result["scores"]
    merged=queue.merge(plan)["results"]["flair"]
    assert merged["complete"] and merged["scores"]==result["scores"]
    assert np.asarray(merged["confusion_matrix"]).shape==(12,13)
    assert merged["confusion_matrix_columns"][-1]=="background"


def test_existing_dataset_background_mappings_are_unchanged():
    assert int(protocol.decode(dict(response="others*24"),"loveda")[0,0])==0
    assert int(protocol.decode(dict(response="others*24"),"levircd")[0,0])==0
    assert int(protocol.decode(dict(response="others*24"),"second",query_key="water")[0,0])==0
    with pytest.raises(ValueError,match="truncated"):
        protocol.decode(dict(response="others*24",generation=dict(hit_token_limit=True,ended_with_eos=False)),"flair")
