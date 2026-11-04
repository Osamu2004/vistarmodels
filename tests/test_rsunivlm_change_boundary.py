"""WSL-only saved-mask boundary regression; no weights/model/CUDA."""
import hashlib

import numpy as np
from PIL import Image
import pytest

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from baselines.segearth_ov import protocols as seg
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_levir_labels as levir
from tools import score_rsunivlm_fused as fused
from tools import score_rsunivlm_change_boundary as tool


def example_second():
    first = np.zeros((40, 40), np.uint8)
    last = np.zeros_like(first)
    first[4:30, 4:20] = 1
    last[12:36, 12:36] = 4
    masks = {key:((first == index) | (last == index)).astype(np.uint8)
        for key, index, _ in protocol.SECOND_QUERIES}
    return masks, [first, last]


def test_second_dense_priority_keeps_overlap_and_class_interfaces_for_boundary():
    masks, targets = example_second()
    prediction, target, audit = tool.second_labels(masks, targets)
    assert prediction[15, 15] == target[15, 15] == 4
    assert prediction[5, 5] == target[5, 5] == 1
    assert prediction[0, 0] == target[0, 0] == 0
    assert audit["prediction"]["overlapping_pixels"] > 0
    row = tool.score_masks("second", masks, targets)
    assert row["class_counts"]["1"]["tp"] == int(masks["water"].sum())
    assert row["class_counts"]["4"]["tp"] == int(masks["tree"].sum())
    direct = score_semantic_boundary_wfm(prediction, target, num_classes=7, ignore_margin=2)
    assert all(row[key] == value for key, value in direct.items())
    # A binary change union loses the water/tree interface.
    binary = score_semantic_boundary_wfm((prediction > 0).astype(np.uint8), (target > 0).astype(np.uint8), num_classes=2)
    assert row["wfm_3px_gt_edge_pixels"] > binary["wfm_3px_gt_edge_pixels"]
    _, boundary = tool.aggregate("second", [row])
    assert boundary["wfm_3px"] == pytest.approx(1)
    assert boundary["wfm_3px"] == aggregate_binary_boundary_wfm([row])["wfm_3px"]
    assert "mwfm_c_3px" not in boundary


def test_second_wrong_internal_boundary_penalized_despite_identical_binary_union():
    masks, targets = example_second()
    truth, _, _ = tool.second_labels(masks, targets)
    # Force the full changed region into tree, erasing the internal boundary.
    for array in masks.values():
        array[:] = 0
    masks["tree"][:] = truth > 0
    row = tool.score_masks("second", masks, targets)
    _, boundary = tool.aggregate("second", [row])
    assert boundary["wfm_3px"] < 1 - 1e-6


def saved_runs(tmp_path, count=2):
    source, summary_root = tmp_path / "saved", tmp_path / "fused"
    source.mkdir(); summary_root.mkdir()
    (source / ".queue.lock").touch(); (summary_root / ".fusion.lock").touch()
    entries, saved = [], {}
    for dataset in tool.DATASETS:
        items, outputs, targets_all = [], [], []
        for i in range(count):
            image_paths = []
            for t in range(2):
                path = tmp_path / f"{dataset}_{i}_image{t}.png"
                Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(path)
                image_paths.append(str(path))
            if dataset == "second":
                masks, targets = example_second()
                raw = targets
            else:
                target = np.zeros((40, 40), np.uint8)
                target[5:30, 8:32] = 1
                masks, targets = {"forward":target.copy()}, [target]
                # This image requires the saved explicit vistar_positive policy.
                raw = [target * np.uint8(156)]
            label_paths = []
            for t, array in enumerate(raw):
                path = tmp_path / f"{dataset}_{i}_gt{t}.png"
                Image.fromarray(array).save(path)
                label_paths.append(str(path))
            sample = protocol.Sample(f"s{i}", tuple(image_paths), tuple(label_paths))
            items.append(dict(sample=sample.serialize(), files=protocol.metadata(sample)))
            outputs.append(masks); targets_all.append(targets)
        entries.append(dict(dataset=dataset, samples=items, contract=protocol.spec(dataset)))
        saved[dataset] = outputs, targets_all
    old_settings = dict(code_sha256=fused.code_identity(), populations={e["dataset"]:fingerprint(e["samples"]) for e in entries},
        levircd_gt_reader=levir.settings("vistar_positive"), strict_population=False)
    plan = dict(identity=fingerprint(old_settings), protocol=old_settings, gpu_ids=["0", "1"], entries=entries, max_samples=0)
    storage.write_json(source / "plan.json", plan)
    storage.write_json(source / "run_config.json", dict(identity=plan["identity"], protocol=old_settings))
    before = dict(source_identity=plan["identity"], source_run=str(source), source_plan_sha256=fused.sha(source / "plan.json"),
        code_sha256=fused.code_identity(), versions=tool.metric_versions())
    summary = dict(identity=fingerprint(before), complete=True, is_full_benchmark=False, results={})
    storage.write_json(summary_root / "run_config.json", dict(identity=summary["identity"], protocol=before))
    for entry in entries:
        dataset = entry["dataset"]
        outputs, targets_all = saved[dataset]
        count_rows = []
        for i, (item, masks, targets) in enumerate(zip(entry["samples"], outputs, targets_all)):
            folder = source / dataset / "shards" / f"rank{i%2:03d}"
            for name in ("pred_mask", "receipts"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            hashes = {}
            for key, mask in masks.items():
                storage.save_prediction(folder / "pred_mask" / f"s{i}_{key}.png", mask)
                hashes[key] = hashlib.sha256(mask.tobytes()).hexdigest()
            storage.write_json(folder / "receipts" / f"s{i}.json", dict(identity=fused.sample_id(plan, dataset, item), shape=[40, 40], sha256=hashes))
            if dataset == "levircd":
                count_rows.append(seg.confusion_matrix(masks["forward"], targets[0], num_classes=2, ignore_index=255))
            else:
                count_rows.append({str(index):protocol.counts(masks[key].astype(bool), (targets[0] == index) | (targets[1] == index))
                    for key, index, _ in protocol.SECOND_QUERIES})
        result = dict(identity=fingerprint([plan["identity"], dataset]), complete=True, num_failed=0,
            processed_samples=count, selected_samples=count, contract=entry["contract"], is_full_benchmark=False)
        if dataset == "levircd":
            matrix = sum(count_rows, np.zeros((2, 2), np.int64))
            result.update(confusion_matrix=matrix.tolist(), scores=seg.metrics_from_confusion(matrix, ["unchanged", "building"]))
            result["scores"]["building_iou"] = result["scores"]["iou_building"]
        else:
            totals = {str(index):{key:sum(row[str(index)][key] for row in count_rows)
                for key in ("tp", "tn", "fp", "fn")} for index in range(1, 7)}
            result["scores"] = protocol.second_metrics(totals)
        summary["results"][dataset] = result
    for name in ("loveda", "flair", "uavid", "xbd_pre", "chn6_cug"):
        summary["results"][name] = dict(complete=True, processed_samples=1, selected_samples=1,
            contract={"primary":"miou"}, scores={"miou":0.42, "wfm_3px":0.12})
    storage.write_json(summary_root / "paper_summary.json", summary)
    return source, summary_root


def invoke(source, summary_root, output, workers=0):
    return tool.main(["--source_run", str(source), "--summary_run", str(summary_root),
        "--output_dir", str(output), "--workers", str(workers)])


def test_cpu_adds_boundary_preserves_original_iou_and_other_five_resumes(tmp_path, monkeypatch):
    source, before = saved_runs(tmp_path)
    snapshot = {p:p.read_bytes() for root in (source, before) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "boundary"
    assert invoke(source, before, output) == 0
    old = storage.read_json(before / "paper_summary.json")
    new = storage.read_json(output / "paper_summary.json")
    for name in tool.DATASETS:
        assert new["results"][name]["scores"]["wfm_3px"] == pytest.approx(1)
        metric = old["results"][name]["contract"]["primary"]
        assert old["results"][name]["scores"][metric] == new["results"][name]["scores"][metric]
    for name in set(protocol.DATASETS) - set(tool.DATASETS):
        assert old["results"][name]["scores"] == new["results"][name]["scores"]
    assert not list(output.rglob("*.png"))
    def forbidden(*args, **kwargs):
        raise AssertionError("A valid cached row must not rerun boundary scoring")
    monkeypatch.setattr(tool, "score_masks", forbidden)
    assert invoke(source, before, output) == 0
    assert snapshot == {p:p.read_bytes() for root in (source, before) for p in root.rglob("*") if p.is_file()}


def test_workers_match_serial_and_no_predicted_image_copies(tmp_path):
    source, before = saved_runs(tmp_path)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    assert invoke(source, before, serial) == 0
    assert invoke(source, before, parallel, workers=2) == 0
    a = storage.read_json(serial / "paper_summary.json")
    b = storage.read_json(parallel / "paper_summary.json")
    for name in tool.DATASETS:
        assert a["results"][name]["scores"] == b["results"][name]["scores"]
    assert not list(parallel.rglob("*.png"))


def test_missing_prediction_or_changed_primary_iou_never_publishes_boundary(tmp_path):
    source, before = saved_runs(tmp_path)
    (source / "second/shards/rank000/pred_mask/s0_water.png").unlink()
    summary = storage.read_json(before / "paper_summary.json")
    summary["results"]["levircd"]["scores"]["building_iou"] = 0.1
    storage.write_json(before / "paper_summary.json", summary)
    output = tmp_path / "failed"
    assert invoke(source, before, output) == 1
    result = storage.read_json(output / "paper_summary.json")
    assert not result["complete"]
    for name in tool.DATASETS:
        assert not result["results"][name]["boundary_complete"]
        assert "wfm_3px" not in result["results"][name]["scores"]


def test_saved_gt_policy_and_separate_outputs_required(tmp_path):
    source, before = saved_runs(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        invoke(source, before, before)
    config = storage.read_json(before / "run_config.json")
    config["protocol"]["versions"]["numpy"] = "other"
    config["identity"] = fingerprint(config["protocol"])
    storage.write_json(before / "run_config.json", config)
    summary = storage.read_json(before / "paper_summary.json")
    summary["identity"] = config["identity"]
    storage.write_json(before / "paper_summary.json", summary)
    with pytest.raises(ValueError, match="versions changed"):
        tool.prepare(source, before)
