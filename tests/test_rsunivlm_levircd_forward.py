"""WSL-only synthetic regressions; no pretrained weights or CUDA are needed."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools import eval_rsunivlm_paper_queue as queue
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def fixture(tmp_path, count=4, world=1):
    samples = []
    for index in range(count):
        a, b, label = (tmp_path / f"{prefix}_{index}.png" for prefix in ("A", "B", "label"))
        Image.new("RGB", (512, 512), (17+index, 0, 0)).save(a)
        Image.new("RGB", (512, 512), (117+index, 0, 0)).save(b)
        target = np.zeros((512, 512), np.uint8)
        target[:, 384:] = 255
        Image.fromarray(target).save(label)
        samples.append(protocol.Sample(f"test_{index:03d}", (str(a), str(b)), (str(label),)))
    entry = dict(dataset="levircd", contract=protocol.spec("levircd"),
        samples=[dict(sample=sample.serialize(), files=protocol.metadata(sample)) for sample in samples])
    plan = dict(identity="synthetic-levir-forward", max_samples=0, entries=[entry],
        gpu_ids=[str(rank) for rank in range(world)], output_dir=str(tmp_path / "output"),
        protocol=dict(batch_size=3, tile_size=512, format_retries=0, strict_population=False))
    return plan, entry, samples


def fake_model(monkeypatch, calls):
    def generate(core, requests, prompt):
        assert prompt == protocol.original.CD_PROMPT
        pairs = [(request[0].getpixel((0, 0))[0], request[1].getpixel((0, 0))[0]) for request in requests]
        # A reverse query would violate this invariant and fail the test.
        assert all(b-a == 100 for a, b in pairs)
        calls.extend(pairs)
        return [dict(response="\n".join(["others*18, building*6"]*23),
            generation=dict(ended_with_eos=True, hit_token_limit=False,
                            inference_batch_size=len(requests), images_per_sequence=2)) for _ in requests]
    monkeypatch.setattr(worker, "generate_requests", generate)
    monkeypatch.setattr(worker.Engine, "core", lambda self: None)


def test_official_prompt_and_forward_only_contract():
    assert protocol.prompt("levircd", 0) == "[SEG] Please segment the building area that have changed in the second image."
    assert worker.query_directions("levircd") == ("forward",)
    assert worker.outputs("levircd") == ("forward",)
    assert "forward query only" in protocol.spec("levircd")["label_protocol"]
    for attempt in (1, 2):
        assert protocol.prompt("levircd", attempt) == protocol.original.format_retry_prompt("levircd", attempt)


def test_one_query_per_tile_no_reverse_or_union_and_native_binary_score(tmp_path, monkeypatch):
    plan, entry, samples = fixture(tmp_path)
    calls = []
    fake_model(monkeypatch, calls)
    original_load_targets = protocol.load_targets
    def load_after_prediction(sample, dataset, shape):
        assert (Path(plan["output_dir"]) / "levircd/shards/rank000/pred_mask" / f"{sample.name}_forward.png").is_file()
        return original_load_targets(sample, dataset, shape)
    monkeypatch.setattr(protocol, "load_targets", load_after_prediction)
    result = worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    assert len(calls) == 4 and result["complete"]
    assert result["query_directions"] == ["forward"]
    assert result["scores"]["building_iou"] == pytest.approx(1.)
    masks = Path(plan["output_dir"]) / "levircd/shards/rank000/pred_mask"
    assert len(list(masks.glob("*.png"))) == 4
    assert not list(masks.glob("*union*")) and not list(masks.glob("*reverse*"))
    for sample in samples:
        with Image.open(masks / f"{sample.name}_forward.png") as image:
            value = np.asarray(image)
        assert value.shape == (512, 512) and set(np.unique(value)) == {0, 1}
        assert not value[:, :384].any() and value[:, 384:].all()
    calls.clear()
    resumed = worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    assert not calls and all(row["reused_prediction"] for row in resumed["rows"])


def test_legacy_union_artifact_cannot_substitute_for_forward_prediction(tmp_path, monkeypatch):
    plan, entry, samples = fixture(tmp_path, count=1)
    calls = []
    fake_model(monkeypatch, calls)
    worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    root = Path(plan["output_dir"]) / "levircd/shards/rank000"
    forward = root / "pred_mask" / f"{samples[0].name}_forward.png"
    old_union = root / "pred_mask" / f"{samples[0].name}_union.png"
    forward.rename(old_union)
    receipt_path = root / "receipts" / f"{samples[0].name}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sha256"] = {"union": receipt["sha256"]["forward"]}
    receipt_path.write_text(json.dumps(receipt))
    for path in (root / "raw_text").glob("*.json"):
        path.unlink()
    calls.clear()
    worker.Engine(plan, 0, 1).evaluate(entry, plan["output_dir"])
    assert len(calls) == 1 and forward.is_file() and old_union.is_file()


def test_two_gpu_shards_keep_ordered_pairs_and_merge_global_counts(tmp_path, monkeypatch):
    plan, entry, _ = fixture(tmp_path, world=2)
    calls = []
    fake_model(monkeypatch, calls)
    a = worker.Engine(plan, 0, 2).evaluate(entry, plan["output_dir"])
    b = worker.Engine(plan, 1, 2).evaluate(entry, plan["output_dir"])
    assert set(a["assigned_samples"]).isdisjoint(b["assigned_samples"])
    result = queue.merge(plan)
    assert len(calls) == 4 and result["complete"]
    assert result["results"]["levircd"]["processed_samples"] == 4
    assert result["results"]["levircd"]["scores"]["building_iou"] == pytest.approx(1.)
    assert not result["is_full_benchmark"]


def test_binary_decoder_refuses_unknown_class_and_truncated_descriptor():
    with pytest.raises(ValueError, match="Unknown levircd category"):
        protocol.decode(dict(response="road*24"), "levircd")
    with pytest.raises(ValueError, match="truncated"):
        protocol.decode(dict(response="building*24", generation=dict(hit_token_limit=True, ended_with_eos=False)), "levircd")
