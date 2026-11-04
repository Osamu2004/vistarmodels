"""CPU-only boundary addition to complete RSUniVLM LEVIR-CD/SECOND results."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import multiprocessing
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from baselines.segearth_ov import protocols as seg
from baselines.binary_boundary_wfm import score_semantic_boundary_wfm, aggregate_binary_boundary_wfm
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_levir_labels as levir
from tools import score_rsunivlm_fused as previous
from tools import rsunivlm_fixed_order_fusion as fusion

DATASETS = ("levircd", "second")
POLICY = "native_change_semantic_boundary_3px_fixed_order_v1"


def metric_versions():
    versions = {}
    for name in ("numpy", "scipy", "Pillow", "opencv-python", "opencv-python-headless", "tifffile", "rasterio"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    versions["cv2"] = cv2.__version__
    return versions


def prepare(source, summary_root):
    plan = storage.read_json(source / "plan.json")
    original = storage.read_json(source / "run_config.json")
    summary = storage.read_json(summary_root / "paper_summary.json")
    old_config = storage.read_json(summary_root / "run_config.json")
    if not plan or not original or plan.get("identity") != fingerprint(plan.get("protocol")) or original.get("identity") != plan["identity"] or original.get("protocol") != plan["protocol"]:
        raise ValueError("Source plan/config identity missing or inconsistent")
    if not summary or not old_config or summary.get("identity") != old_config.get("identity") or old_config.get("identity") != fingerprint(old_config.get("protocol")):
        raise ValueError("Require the completed exclusive-fusion summary/config")
    before = old_config["protocol"]
    if before.get("versions") != metric_versions():
        raise ValueError("Metric-library versions changed since semantic fusion; use the original scoring environment")
    if before.get("source_identity") != plan["identity"] or before.get("source_run") != str(source) or before.get("source_plan_sha256") != previous.sha(source / "plan.json"):
        raise ValueError("Fused summary refers to another prediction source")
    if summary.get("complete") is not True or set(summary.get("results", {})) != set(protocol.DATASETS):
        raise ValueError("Require a complete seven-dataset summary")
    # Preserve the proven inference/GT/scoring helpers and old fusion identity.
    for name, digest in previous.code_identity().items():
        if before.get("code_sha256", {}).get(name) != digest:
            raise ValueError(f"Fusion numerical dependency changed: {name}")
        if name not in ("tools/score_rsunivlm_fused.py", "tools/rsunivlm_fixed_order_fusion.py") and plan["protocol"].get("code_sha256", {}).get(name) != digest:
            raise ValueError(f"Original numerical dependency changed: {name}")
    reader = plan["protocol"].get("levircd_gt_reader")
    if reader and reader != levir.settings(reader.get("gt_mode")):
        raise ValueError("Saved LEVIR GT reader policy/source changed")
    entries = {entry["dataset"]: entry for entry in plan["entries"]}
    selected = []
    for dataset in DATASETS:
        entry = entries[dataset]
        if fingerprint(entry["samples"]) != plan["protocol"]["populations"][dataset]:
            raise ValueError(f"Source population differs: {dataset}")
        population = entry["samples"][:plan["max_samples"]] if plan["max_samples"] else entry["samples"]
        result = summary["results"][dataset]
        if not population or not plan.get("gpu_ids") or result.get("identity") != fingerprint([plan["identity"], dataset]) or result.get("complete") is not True or result.get("num_failed", 0) or result.get("processed_samples") != len(population) or result.get("selected_samples") != len(population):
            raise ValueError(f"Missing/incompatible complete change scores: {dataset}")
        names = [item["sample"]["name"] for item in population]
        if len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
            raise ValueError("Unsafe/duplicate sample names")
        selected.append(dict(dataset=dataset, samples=population))
    value = dict(schema="rsunivlm_change_boundary_v1", policy=POLICY, source_run=str(source),
        source_identity=plan["identity"], source_plan_sha256=previous.sha(source / "plan.json"),
        summary_run=str(summary_root), summary_identity=summary["identity"],
        summary_sha256=previous.sha(summary_root / "paper_summary.json"),
        levircd_gt_reader=reader or levir.settings("exact_binary"),
        code_sha256={**previous.code_identity(), "tools/rsunivlm_levir_labels.py": previous.sha(Path(levir.__file__)),
            "tools/score_rsunivlm_change_boundary.py": previous.sha(Path(__file__))},
        versions=metric_versions(),
        levircd_boundary="LoveDA semantic 3px WFm applied to native-resolution binary building-change labels",
        second_boundary="merge six undirected queries into labels 0..6 by ascending ID, last positive wins; same merger for GT class memberships; then LoveDA whole-semantic-label boundary WFm",
        second_gt_membership="class present in either original annotation; fixed single input order; no reverse prediction",
        second_overlap_priority=[1, 2, 3, 4, 5, 6], second_uncovered_id=0,
        boundary_aggregation="existing IDGBR min positive edge/nonedge image-averaged scores; no six-class WFm macro",
        iou_check="rebuild original sufficient statistics; compare counts and primary metric with saved source",
        model_forward_calls=0, gpu_used=False, saves_prediction_images=False)
    return plan, summary, selected, dict(identity=fingerprint(value), protocol=value)


def predictions(job):
    item, plan, dataset = job["item"], job["plan"], job["dataset"]
    name = item["sample"]["name"]
    root = Path(job["source"]) / dataset / "shards" / f"rank{job['position'] % len(plan['gpu_ids']):03d}"
    receipt_path = root / "receipts" / f"{name}.json"
    receipt = storage.read_json(receipt_path)
    keys = ("forward",) if dataset == "levircd" else tuple(key for key, _, _ in protocol.SECOND_QUERIES)
    if not receipt or receipt.get("identity") != previous.sample_id(plan, dataset, item) or set(receipt.get("sha256", {})) != set(keys):
        raise ValueError(f"Missing/incompatible source receipt: {receipt_path}")
    arrays = {}
    for key in keys:
        path = root / "pred_mask" / f"{name}_{key}.png"
        if path.is_symlink():
            raise ValueError(f"Unexpected prediction symlink: {path}")
        with Image.open(path) as opened:
            array = np.asarray(opened).copy()
        if array.dtype != np.uint8 or array.ndim != 2 or list(array.shape) != receipt.get("shape") or not all(array.shape) or not np.all(array < 2) or hashlib.sha256(array.tobytes()).hexdigest() != receipt["sha256"][key]:
            raise ValueError(f"Saved binary prediction differs: {path}")
        arrays[key] = array
    return arrays, previous.sha(receipt_path)


def second_labels(masks, targets):
    """Deterministic dense views for boundary only; preserve multilabel IoU.

    GT has up to two changed categories at a pixel. Use the same fixed
    last-positive class priority for prediction and GT, without selecting
    predictions according to GT. Class0 is the uncovered/unchanged default.
    """
    shape = masks[protocol.SECOND_QUERIES[0][0]].shape
    def pairs():
        yield 0, np.zeros(shape, np.uint8)
        for key, index, _ in protocol.SECOND_QUERIES:
            yield index, masks[key]
    prediction, pred_audit = fusion.fuse(pairs(), num_classes=7, uncovered_id=0)
    def gt_pairs():
        yield 0, np.zeros(shape, np.uint8)
        for _, index, _ in protocol.SECOND_QUERIES:
            yield index, ((targets[0] == index) | (targets[1] == index)).astype(np.uint8)
    target, gt_audit = fusion.fuse(gt_pairs(), num_classes=7, uncovered_id=0)
    return prediction, target, dict(prediction=pred_audit, ground_truth=gt_audit,
        scope="fixed dense views for semantic boundary only; original multilabel IoU unchanged")


def score_masks(dataset, masks, targets):
    if dataset == "levircd":
        prediction, target = masks["forward"], targets[0]
        return dict(confusion_matrix=seg.confusion_matrix(prediction, target, num_classes=2, ignore_index=255).tolist(),
            **score_semantic_boundary_wfm(prediction, target, num_classes=2, ignore_margin=2))
    class_counts = {}
    for key, index, _ in protocol.SECOND_QUERIES:
        target = ((targets[0] == index) | (targets[1] == index)).astype(np.uint8)
        class_counts[str(index)] = protocol.counts(masks[key].astype(bool), target.astype(bool))
    prediction, target, audit = second_labels(masks, targets)
    return dict(class_counts=class_counts, boundary_fusion=audit,
        **score_semantic_boundary_wfm(prediction, target, num_classes=7, ignore_margin=2))


def process(job):
    cv2.setNumThreads(1)
    sample = protocol.Sample.restore(job["item"]["sample"])
    try:
        if protocol.metadata(sample) != job["item"]["files"]:
            raise ValueError("Input files changed since original inference")
        masks, source_receipt = predictions(job)
        shape = next(iter(masks.values())).shape
        identity = fingerprint([job["identity"], job["dataset"], previous.sample_id(job["plan"], job["dataset"], job["item"]), source_receipt])
        path = Path(job["output"]) / job["dataset"] / "rows" / f"{sample.name}.json"
        if path.is_symlink():
            raise ValueError("Unexpected cached row symlink")
        cached = storage.read_json(path)
        if previous.cached_row_valid(cached) and cached.get("identity") == identity:
            return dict(ok=True, cached=True, row=cached)
        if job["dataset"] == "levircd":
            targets = [levir.read_binary(sample.labels[0], shape, gt_mode=job["gt_mode"])[0]]
        else:
            targets = protocol.load_targets(sample, "second", shape)
        row = dict(identity=identity, sample=sample.name, source_receipt_sha256=source_receipt,
            **score_masks(job["dataset"], masks, targets))
        if protocol.metadata(sample) != job["item"]["files"]:
            raise ValueError("Input files changed during boundary scoring")
        row["row_payload_sha256"] = fingerprint(row)
        storage.write_json(path, row)
        return dict(ok=True, cached=False, row=row)
    except Exception as error:
        return dict(ok=False, sample=sample.name, error=str(error), traceback=traceback.format_exc())


def aggregate(dataset, rows):
    if not rows:
        return {}, {}
    if dataset == "levircd":
        matrix = sum((np.asarray(row["confusion_matrix"], np.int64) for row in rows), np.zeros((2, 2), np.int64))
        scores = seg.metrics_from_confusion(matrix, ["unchanged", "building"])
        scores["building_iou"] = scores["iou_building"]
        return scores, dict(**aggregate_binary_boundary_wfm(rows), boundary_metric="wfm_3px")
    counts = {str(index): {field:sum(row["class_counts"][str(index)][field] for row in rows)
        for field in ("tp", "tn", "fp", "fn")} for _, index, _ in protocol.SECOND_QUERIES}
    return protocol.second_metrics(counts), dict(**aggregate_binary_boundary_wfm(rows),
        boundary_metric="wfm_3px", boundary_protocol=dict(policy=POLICY,
            classes="0 unchanged plus six changed labels in a fixed dense boundary view",
            fusion_policy=fusion.POLICY,
            aggregation="same whole-semantic-boundary IDGBR image aggregation as LoveDA",
            original_iou_stays_multilabel=True))


def check_iou(dataset, old, rebuilt, rows):
    metric = "building_iou" if dataset == "levircd" else "miou_c"
    if abs(old["scores"][metric] - rebuilt[metric]) > 1e-12:
        raise ValueError(f"Rebuilt {metric} differs from saved scores; refusing boundary publication")
    if dataset == "levircd" and "confusion_matrix" in old:
        matrix = sum((np.asarray(row["confusion_matrix"], np.int64) for row in rows), np.zeros((2, 2), np.int64))
        if matrix.tolist() != old["confusion_matrix"]:
            raise ValueError("Rebuilt LEVIR confusion matrix differs")
    if dataset == "second":
        for _, _, name in protocol.SECOND_QUERIES:
            for field in ("tp", "tn", "fp", "fn"):
                if old["scores"]["per_class"][name][field] != rebuilt["per_class"][name][field]:
                    raise ValueError(f"Rebuilt SECOND {name}/{field} differs")


def jobs_result(pool, jobs, workers):
    if pool is None:
        yield from map(process, jobs)
        return
    pending, iterator = deque(), iter(jobs)
    for _ in range(2 * workers):
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(process, job))
    while pending:
        yield pending.popleft().result()
        job = next(iterator, None)
        if job is not None:
            pending.append(pool.submit(process, job))


def evaluate(source, output, plan, config, entry, old, pool, workers):
    dataset = entry["dataset"]
    (output / dataset / "rows").mkdir(parents=True, exist_ok=True)
    minimal = {"identity":plan["identity"], "gpu_ids":plan["gpu_ids"]}
    jobs = (dict(source=str(source), output=str(output), dataset=dataset, item=item, position=i,
        plan=minimal, identity=config["identity"], gt_mode=config["protocol"]["levircd_gt_reader"]["gt_mode"])
        for i, item in enumerate(entry["samples"]))
    rows, failed, cached = [], [], 0
    bar = tqdm(jobs_result(pool, jobs, workers), total=len(entry["samples"]), desc=f"Boundary {dataset}", unit="image")
    for result in bar:
        if result["ok"]:
            rows.append(result["row"])
            cached += int(result["cached"])
        else:
            failed.append(result)
            print(f"[boundary failed] {dataset}/{result['sample']}: {result['error']}", flush=True)
        bar.set_postfix(scored=len(rows), failed=len(failed), cached=cached)
    rebuilt, boundary = aggregate(dataset, rows)
    complete = len(rows) == len(entry["samples"]) and not failed
    if complete:
        try:
            check_iou(dataset, old, rebuilt, rows)
        except (ValueError, KeyError) as error:
            failed.append(dict(error=str(error), stage="original_iou_consistency"))
            complete = False
    report = dict(identity=fingerprint([config["identity"], dataset]), complete=complete,
        selected_samples=len(entry["samples"]), processed_samples=len(rows), num_failed=len(failed),
        failures=failed, cached_samples_this_invocation=cached, iou_consistency_passed=complete,
        rebuilt_iou=rebuilt, candidate_boundary=boundary)
    storage.write_json(output / dataset / "boundary_report.json", report)
    result = copy.deepcopy(old)
    result.update(identity=report["identity"], source_prediction_identity=old["identity"],
        complete=complete, boundary_complete=complete, boundary_report=str(output / dataset / "boundary_report.json"),
        boundary_processed_samples=len(rows), boundary_num_failed=len(failed),
        is_full_benchmark=bool(old.get("is_full_benchmark")) and complete)
    # An incomplete/mismatched boundary is never added to the main scores.
    if complete:
        result["scores"].update(boundary)
        result["boundary_provenance"] = config["protocol"]
    else:
        result["boundary_failures"] = failed
    storage.write_json(output / dataset / "metrics.json", result)
    return result


def write_summary(output, old, changes, config):
    summary = copy.deepcopy(old)
    summary.update(identity=config["identity"], previous_summary_identity=old["identity"],
        boundary_attempted_datasets=list(changes),
        boundary_updated_datasets=[name for name, value in changes.items() if value["boundary_complete"]],
        change_boundary_protocol=config["protocol"])
    summary["results"].update(changes)
    annotations = storage.read_json(ROOT / "baselines/rsunivlm/receipts/wsl_fused_all7_scores_20261104.json") or {}
    summary["training_annotation"] = annotations.get("training_annotation", {})
    for name, result in summary["results"].items():
        metadata = annotations.get("results", {}).get(name, {}).get("training_exposure")
        if metadata:
            result["training_exposure"] = metadata
    summary["complete"] = all(value.get("complete") is True for value in summary["results"].values())
    summary["is_full_benchmark"] = bool(old.get("is_full_benchmark")) and summary["complete"]
    storage.write_json(output / "paper_summary.json", summary)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["dataset", "dataset_display", "primary_metric", "value", "value_percent",
        "boundary_metric", "wfm_3px", "wfm_percent", "processed", "selected", "complete",
        "boundary_processed", "boundary_failed"])
    writer.writeheader()
    for name, value in summary["results"].items():
        scores = value["scores"]
        key = value["contract"]["primary"]
        boundary = scores.get("wfm_3px")
        marker = value.get("training_exposure", {}).get("marker", "")
        writer.writerow(dict(dataset=name, dataset_display=name+marker, primary_metric=key, value=scores[key],
            value_percent=f"{100*scores[key]:.2f}", boundary_metric=scores.get("boundary_metric", "wfm_3px" if boundary is not None else ""),
            wfm_3px=boundary, wfm_percent="" if boundary is None else f"{100*boundary:.2f}",
            processed=value["processed_samples"], selected=value["selected_samples"], complete=value["complete"],
            boundary_processed=value.get("boundary_processed_samples", ""), boundary_failed=value.get("boundary_num_failed", "")))
    storage.atomic_text(output / "paper_summary.csv", stream.getvalue())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", required=True, help="Original complete recovery run containing native binary masks/plan")
    parser.add_argument("--summary_run", default="", help="Completed fixedorder_fusion_v1 run containing the latest seven-set summary")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("workers must be nonnegative; 0 uses the main process")
    source = Path(args.source_run).expanduser().resolve()
    summary_root = Path(args.summary_run or (str(source)+"_fixedorder_fusion_v1")).expanduser().resolve()
    output = Path(args.output_dir or (str(summary_root)+"_cd_boundary_v1")).expanduser().resolve()
    if any(output == item or output in item.parents or item in output.parents for item in (source, summary_root)):
        raise ValueError("New boundary output must be separate from both original runs")
    with ExitStack() as stack:
        for root, name in ((source, ".queue.lock"), (summary_root, ".fusion.lock")):
            lock = stack.enter_context((root / name).open("rb"))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, original, entries, config = prepare(source, summary_root)
        output.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((output / ".boundary.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_config = storage.read_json(output / "run_config.json")
        if old_config and old_config != config:
            raise ValueError("Boundary output identity differs; choose a new output directory")
        if not old_config and any(p.name != ".boundary.lock" for p in output.iterdir()):
            raise ValueError("Refusing to adopt unidentified existing outputs")
        storage.write_json(output / "run_config.json", config)
        print(f"[change boundary] CPU workers={args.workers}; no model/Qwen/GPU/new images; OUTPUT_DIR={output}", flush=True)
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"))) if args.workers else None
        changes = {}
        for entry in entries:
            name = entry["dataset"]
            changes[name] = evaluate(source, output, plan, config, entry, original["results"][name], pool, args.workers)
        summary = write_summary(output, original, changes, config)
        storage.write_json(output / "runtime_finished.json", dict(identity=config["identity"], complete=summary["complete"],
            model_forward_calls=0, gpu_used=False, new_prediction_images=0))
        print((output / "paper_summary.csv").read_text(), flush=True)
        print(f"[change boundary] complete={summary['complete']}; summary={output/'paper_summary.json'}", flush=True)
        return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
