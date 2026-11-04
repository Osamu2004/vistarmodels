"""Run in WSL: synthetic CPU fusion, scorer and restart/provenance checks."""
import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from baselines.segearth_ov import protocols as seg
from tools import rsunivlm_fixed_order_fusion as fusion
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol
from tools import score_rsunivlm_fused as tool


def test_fixed_order_includes_background_but_later_foreground_wins():
    masks = [np.ones((2, 3), np.uint8), np.array([[0, 1, 0], [0, 1, 0]], np.uint8),
        np.array([[0, 1, 1], [0, 0, 0]], np.uint8)]
    pred, audit = fusion.fuse(enumerate(masks), num_classes=3, uncovered_id=0)
    assert pred.tolist() == [[0, 2, 2], [0, 1, 0]]
    assert audit["overlapping_pixels"] == 3
    assert audit["uncovered_pixels"] == 0


def test_flair_empty_predictions_are_background_not_building_or_gt_ignore():
    masks = [np.zeros((16, 16), np.uint8) for _ in range(12)]
    prediction, audit = fusion.fuse(enumerate(masks), num_classes=12, uncovered_id=12)
    assert np.all(prediction == 12) and audit["uncovered_pixels"] == 256
    target = np.zeros((16, 16), np.uint8)
    row = tool.score_prediction(prediction, target, "flair")
    matrix = np.asarray(row["confusion_matrix"])
    assert matrix.shape == (12, 13) and matrix[0, 12] == 256
    scores = seg.metrics_for_dataset(matrix, "flair")
    assert scores["valid_pixels"] == 256
    assert scores["miou"] == scores["iou_building"] == 0
    assert scores["rejected_pixels"] == 256


@pytest.mark.parametrize("pairs", [[], [(1, np.zeros((2, 2), np.uint8))],
    [(0, np.zeros((2, 2), np.uint8))],
    [(0, np.zeros((2, 2), np.uint8)), (1, np.zeros((3, 2), np.uint8))],
    [(0, np.full((2, 2), 255, np.uint8)), (1, np.zeros((2, 2), np.uint8))]])
def test_missing_wrong_order_ragged_or_nonbinary_queries_never_fill(pairs):
    with pytest.raises(ValueError):
        fusion.fuse(pairs, num_classes=2, uncovered_id=0)


def test_semantic_boundaries_include_foreground_class_transitions_and_gt_ignore():
    target = np.ones((40, 40), np.uint8)
    target[:, 20:] = 2
    target[:4] = 255
    prediction = target.copy()
    prediction[:4] = 0
    row = tool.score_prediction(prediction, target, "loveda")
    direct = score_semantic_boundary_wfm(prediction, target, num_classes=7, ignore_margin=2)
    assert row["wfm_3px_gt_edge_pixels"] > 0
    for key, value in direct.items():
        assert row[key] == value
    assert np.asarray(row["confusion_matrix"]).sum() == 36 * 40
    assert aggregate_binary_boundary_wfm([row])["wfm_3px"] == pytest.approx(1)


def test_all_gt_ignored_is_not_a_missing_prediction_or_zero_boundary():
    row = tool.score_prediction(np.zeros((16, 16), np.uint8), np.full((16, 16), 255, np.uint8), "flair")
    assert np.asarray(row["confusion_matrix"]).sum() == 0
    assert row["wfm_3px_edge"] is row["wfm_3px_nonedge"] is None


def source_run(tmp_path, dataset="loveda", count=3):
    source = tmp_path / "saved"
    source.mkdir()
    (source / ".queue.lock").touch()
    items = []
    classes = protocol.spec(dataset)["classes"]
    for index in range(count):
        rgb = tmp_path / f"rgb{index}.png"
        label = tmp_path / f"gt{index}.png"
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(rgb)
        target = np.ones((32, 32), np.uint8)
        target[:, 16:] = 2
        # LoveDA original IDs 1..7 become dense IDs 0..6.
        Image.fromarray(target + 1).save(label)
        sample = protocol.Sample(f"s{index}", (str(rgb),), (str(label),))
        items.append(dict(sample=sample.serialize(), files=protocol.metadata(sample)))
    settings = dict(semantic_mode=native.MODE, code_sha256=tool.code_identity(),
        populations={dataset:fingerprint(items)},
        native_semantic={dataset:native.settings(dataset, 0)}, strict_population=False)
    plan = dict(identity=fingerprint(settings), protocol=settings, gpu_ids=["0", "1"],
        max_samples=0, entries=[dict(dataset=dataset, samples=items, contract=native.contract(dataset))])
    storage.write_json(source / "plan.json", plan)
    storage.write_json(source / "run_config.json", dict(identity=plan["identity"], protocol=settings))
    result = dict(identity=fingerprint([plan["identity"], dataset]), complete=True, processed_samples=count,
        selected_samples=count, num_failed=0, scores={"miou":0.33}, contract=protocol.spec(dataset))
    storage.write_json(source / "paper_summary.json", dict(complete=True, results={dataset:result}))
    for index, item in enumerate(items):
        folder = source / dataset / "shards" / f"rank{index%2:03d}"
        for name in ("pred_mask", "receipts"):
            (folder / name).mkdir(parents=True, exist_ok=True)
        hashes = {}
        for key, class_id, _ in native.queries(dataset):
            array = np.zeros((32, 32), np.uint8)
            if class_id == 1:
                array[:, :16] = 1
            if class_id == 2:
                array[:, 16:] = 1
            storage.save_prediction(folder / "pred_mask" / f"s{index}_{key}.png", array)
            hashes[key] = hashlib.sha256(array.tobytes()).hexdigest()
        storage.write_json(folder / "receipts" / f"s{index}.json", dict(identity=tool.sample_id(plan, dataset, item),
            shape=[32, 32], sha256=hashes))
    return source, plan


def test_whole_pipeline_reuses_predictions_aggregates_counts_and_resumes(tmp_path, monkeypatch):
    source, plan = source_run(tmp_path)
    output = tmp_path / "fused"
    old_files = {str(path.relative_to(source)):path.read_bytes() for path in source.rglob("*") if path.is_file()}
    original_loader = protocol.load_targets
    calls = []

    def load(sample, dataset, shape):
        # Fixed labels are already committed before any GT pixel is read.
        assert (output / dataset / "pred_mask" / f"{sample.name}.png").is_file()
        calls.append(sample.name)
        return original_loader(sample, dataset, shape)

    monkeypatch.setattr(protocol, "load_targets", load)
    args = ["--source_run", str(source), "--output_dir", str(output), "--datasets", "loveda", "--workers", "0"]
    assert tool.main(args) == 0
    summary = storage.read_json(output / "paper_summary.json")
    result = summary["results"]["loveda"]
    assert result["complete"] and result["processed_samples"] == 3
    assert result["scores"]["miou"] == 1
    assert result["scores"]["wfm_3px"] == pytest.approx(1)
    assert sum(map(sum, result["confusion_matrix"])) == 3 * 32 * 32
    assert not result["is_full_benchmark"]
    assert len(calls) == 3
    assert tool.main(args) == 0
    assert len(calls) == 3  # No GT reload/WFm recomputation on a valid resume.
    assert storage.read_json(output / "loveda/metrics.json")["cached_samples_this_invocation"] == 3
    assert {str(path.relative_to(source)):path.read_bytes() for path in source.rglob("*") if path.is_file()} == old_files
    row = storage.read_json(output / "loveda/rows/s0.json")
    row["confusion_matrix"][0][0] = 999999  # An accidental edited row must be rescored.
    storage.write_json(output / "loveda/rows/s0.json", row)
    assert tool.main(args) == 0 and len(calls) == 4


@pytest.mark.parametrize("damage", ["missing_mask", "changed_mask", "changed_receipt", "changed_gt"])
def test_bad_source_stays_partial_and_never_fills_background(tmp_path, damage):
    source, plan = source_run(tmp_path)
    folder = source / "loveda/shards/rank000"
    key = native.queries("loveda")[0][0]
    path = folder / "pred_mask" / f"s0_{key}.png"
    if damage == "missing_mask":
        path.unlink()
    elif damage == "changed_mask":
        storage.save_prediction(path, np.ones((32, 32), np.uint8))
    elif damage == "changed_receipt":
        storage.write_json(folder / "receipts/s0.json", dict(identity="other", shape=[32, 32], sha256={}))
    else:
        Image.fromarray(np.zeros((16, 16), np.uint8)).save(Path(plan["entries"][0]["samples"][0]["sample"]["labels"][0]))
    output = tmp_path / "partial"
    assert tool.main(["--source_run", str(source), "--output_dir", str(output), "--datasets", "loveda", "--workers", "0"]) == 1
    result = storage.read_json(output / "loveda/metrics.json")
    assert not result["complete"] and result["num_failed"] == 1 and result["processed_samples"] == 2
    assert not (output / "loveda/pred_mask/s0.png").exists()


def test_changed_protocol_or_scorer_refuses_old_cache(tmp_path):
    source, plan = source_run(tmp_path)
    plan["protocol"]["code_sha256"]["baselines/binary_boundary_wfm.py"] = "other"
    plan["identity"] = fingerprint(plan["protocol"])
    storage.write_json(source / "plan.json", plan)
    storage.write_json(source / "run_config.json", dict(identity=plan["identity"], protocol=plan["protocol"]))
    with pytest.raises(ValueError, match="scorer/query code differs"):
        tool.prepare(source, ["loveda"])


def test_inherits_binary_and_second_without_relabeling_query_metrics(tmp_path):
    source, plan = source_run(tmp_path)
    _, original, entries, config = tool.prepare(source, ["loveda"])
    inherited = dict(dataset="second", complete=True, processed_samples=593, selected_samples=593,
        num_failed=0, standard_metrics_available=True, is_full_benchmark=True,
        contract={"primary":"miou_c"}, scores={"miou_c":0.0949})
    original["results"]["second"] = copy.deepcopy(inherited)
    output = tmp_path / "summary"
    output.mkdir()
    updated = copy.deepcopy(original["results"]["loveda"])
    updated.update(standard_metrics_available=True, is_full_benchmark=False)
    summary = tool.write_summary(output, original, {"loveda":updated}, config)
    assert summary["results"]["second"]["scores"] == inherited["scores"]
    assert "wfm_3px" not in summary["results"]["second"]["scores"]
    assert "inherited_from_source_run" not in original["results"]["second"]


def test_fusion_gt_independence_and_exclusion_of_multilabel_second():
    assert fusion.settings("loveda", protocol.spec("loveda")["classes"])["gt_used_for_fusion"] is False
    with pytest.raises(ValueError, match="only defined"):
        fusion.settings("second", ["water", "tree"])


def test_two_cpu_workers_equal_serial_sufficient_statistics(tmp_path):
    source, _ = source_run(tmp_path)
    common = ["--source_run", str(source), "--datasets", "loveda"]
    for workers in (0, 2):
        output = tmp_path / f"cpu{workers}"
        assert tool.main(common + ["--workers", str(workers), "--output_dir", str(output)]) == 0
    serial = storage.read_json(tmp_path / "cpu0/loveda/metrics.json")
    parallel = storage.read_json(tmp_path / "cpu2/loveda/metrics.json")
    assert serial["confusion_matrix"] == parallel["confusion_matrix"]
    assert serial["scores"] == parallel["scores"]


def test_active_source_queue_and_nested_output_refused(tmp_path):
    import fcntl
    source, _ = source_run(tmp_path)
    with (source / ".queue.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            tool.main(["--source_run", str(source), "--datasets", "loveda", "--workers", "0"])
    with pytest.raises(ValueError, match="separate sibling"):
        tool.main(["--source_run", str(source), "--datasets", "loveda", "--output_dir", str(source / "new")])
