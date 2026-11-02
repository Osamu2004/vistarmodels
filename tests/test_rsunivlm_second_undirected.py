"""WSL synthetic checks; never run pretrained models or local CUDA here."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools import eval_rsunivlm_paper_queue as queue
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def fixture(tmp_path, count=3, world=1):
    samples = []
    for index in range(count):
        paths = [tmp_path / f"{prefix}_{index}.png" for prefix in ("T1", "T2", "label1", "label2")]
        Image.new("RGB", (512, 512), (17+index, 0, 0)).save(paths[0])
        Image.new("RGB", (512, 512), (117+index, 0, 0)).save(paths[1])
        # Water -> building. Both categories must remain positive in the
        # directionless class-wise target; neither may overwrite the other.
        Image.fromarray(np.full((512, 512), 1, np.uint8)).save(paths[2])
        Image.fromarray(np.full((512, 512), 5, np.uint8)).save(paths[3])
        samples.append(protocol.Sample(f"case_{index}", tuple(map(str, paths[:2])), tuple(map(str, paths[2:]))))
    entry = dict(dataset="second", contract=protocol.spec("second"),
        samples=[dict(sample=s.serialize(), files=protocol.metadata(s)) for s in samples])
    plan = dict(identity="synthetic-second-undirected-six", max_samples=0,
        gpu_ids=[str(rank) for rank in range(world)], entries=[entry], output_dir=str(tmp_path / "output"),
        protocol=dict(batch_size=3, tile_size=512, format_retries=0, strict_population=False))
    return plan, entry, samples


def fake_model(monkeypatch, calls):
    def generate(core, requests, prompt):
        key, index, name = next(row for row in protocol.SECOND_QUERIES if prompt == protocol.prompt("second", query_key=row[0]))
        pairs = [(r[0].getpixel((0, 0))[0], r[1].getpixel((0, 0))[0]) for r in requests]
        assert all(b-a == 100 for a, b in pairs), "Input order was reversed"
        calls.append((key, len(requests), pairs))
        label = name if index in (1, 5) else "others"
        return [dict(response="\n".join([label+"*24"]*23),
            generation=dict(inference_batch_size=len(requests), images_per_sequence=2,
                            ended_with_eos=True, hit_token_limit=False)) for _ in requests]
    monkeypatch.setattr(worker, "generate_requests", generate)
    monkeypatch.setattr(worker.Engine, "core", lambda self: None)


def test_six_semantic_queries_preserve_classes_without_temporal_direction():
    spec = protocol.spec("second")
    assert spec["classes"] == [name for _, _, name in protocol.SECOND_QUERIES]
    assert len(spec["classes"]) == 6 and spec["primary"] == "miou_c"
    assert worker.query_directions("second") == ("single_pair",)
    assert worker.outputs("second") == tuple(key for key, _, _ in protocol.SECOND_QUERIES)
    for key, _, name in protocol.SECOND_QUERIES:
        text = protocol.prompt("second", query_key=key)
        assert name in text and "between these two images" in text
        assert "second image" not in text and "first image" not in text
    with pytest.raises(ValueError, match="explicit category"):
        protocol.prompt("second")


def test_decoder_is_category_specific_and_refuses_generic_words_and_truncation():
    grid = protocol.decode(dict(response="others*23, water*1"), "second", query_key="water")
    np.testing.assert_array_equal(grid, [[0]*23+[1]])
    assert protocol.decode(dict(response="ground*24"), "second", query_key="bare_land").all()
    for word in ("building", "land-cover", "land Cover", "land_cover", "class"):
        with pytest.raises(ValueError, match="Unknown second category"):
            protocol.decode(dict(response=word+"*24"), "second", query_key="water")
    with pytest.raises(ValueError, match="truncated"):
        protocol.decode(dict(response="water*24", generation=dict(hit_token_limit=True, ended_with_eos=False)),
                        "second", query_key="water")


def test_all_six_masks_exist_before_gt_and_overlapping_classes_keep_their_scores(tmp_path, monkeypatch):
    plan, entry, samples = fixture(tmp_path)
    calls = []
    fake_model(monkeypatch, calls)
    root = Path(plan["output_dir"]) / "second/shards/rank000"
    original = protocol.load_targets
    def targets_after_all_masks(sample, dataset, shape):
        assert all((root / "pred_mask" / f"{sample.name}_{key}.png").is_file() for key, _, _ in protocol.SECOND_QUERIES)
        return original(sample, dataset, shape)
    monkeypatch.setattr(protocol, "load_targets", targets_after_all_masks)
    result = worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    assert result["complete"] and sum(size for _, size, _ in calls) == 3*6
    assert result["scores"]["miou_c"] == pytest.approx(2/6, abs=1e-6)
    assert result["scores"]["per_class"]["water"]["iou_change"] == pytest.approx(1.)
    assert result["scores"]["per_class"]["building"]["iou_change"] == pytest.approx(1.)
    assert len(list((root / "pred_mask").glob("*.png"))) == 3*6
    assert not list((root / "pred_mask").glob("*reverse*"))
    for sample in samples:
        for key in ("water", "building"):
            with Image.open(root / "pred_mask" / f"{sample.name}_{key}.png") as image:
                assert np.asarray(image).all()
    raw = json.loads(next((root / "raw_text").glob("*.json")).read_text())
    assert raw["temporal_mode"] == "undirected" and raw["input_order"] == ["T1", "T2"]
    assert "direction" not in raw and raw["query_class_id"] in range(1, 7)
    calls.clear()
    resumed = worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    assert not calls and all(row["reused_prediction"] for row in resumed["rows"])


def test_each_category_fills_arbitrary_batches_across_images(tmp_path, monkeypatch):
    plan, entry, _ = fixture(tmp_path, count=5)
    calls = []
    fake_model(monkeypatch, calls)
    worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    for key, _, _ in protocol.SECOND_QUERIES:
        assert [size for name, size, _ in calls if name == key] == [3, 2]


def test_two_shards_merge_class_counts_not_binary_or_average_scores(tmp_path, monkeypatch):
    plan, entry, _ = fixture(tmp_path, count=4, world=2)
    calls = []
    fake_model(monkeypatch, calls)
    a = worker.Engine(plan, 0, 2).evaluate(entry, plan["output_dir"])
    b = worker.Engine(plan, 1, 2).evaluate(entry, plan["output_dir"])
    assert set(a["assigned_samples"]).isdisjoint(b["assigned_samples"])
    merged = queue.merge(plan)
    assert merged["complete"] and not merged["is_full_benchmark"]
    result = merged["results"]["second"]
    assert result["processed_samples"] == 4 and result["scores"]["miou_c"] == pytest.approx(2/6, abs=1e-6)
    assert sum(size for _, size, _ in calls) == 4*6


def test_failed_category_keeps_other_samples_running_without_scoring_partial_masks(tmp_path,monkeypatch):
    plan,entry,_=fixture(tmp_path,count=4)
    calls=[]; fake_model(monkeypatch,calls)
    original=worker.generate_requests
    def generate(core,requests,prompt):
        answers=original(core,requests,prompt)
        if prompt==protocol.prompt("second",query_key="water"):
            for request,answer in zip(requests,answers):
                if request[0].getpixel((0,0))[0]==17: answer["response"]="conj"
        return answers
    monkeypatch.setattr(worker,"generate_requests",generate)
    targets=protocol.load_targets
    def score_only_complete(sample,*args):
        assert sample.name!="case_0"
        return targets(sample,*args)
    monkeypatch.setattr(protocol,"load_targets",score_only_complete)
    result=worker.Engine(plan,0,1).evaluate(entry,plan["output_dir"])
    assert result["attempted"]==4 and result["processed"]==3 and result["num_failed"]==1
    assert result["exhausted_population"] and not result["complete"]
    assert sum(size for _,size,_ in calls)==4*6
    root=Path(plan["output_dir"])/"second/shards/rank000"
    assert not list((root/"pred_mask").glob("case_0_*.png"))
    merged=queue.merge(plan)["results"]["second"]
    assert not merged["complete"] and merged["num_failed"]==1 and merged["exhausted_population"]
    assert merged["scores"]["per_class"]["water"]["tp"]==3*512*512
