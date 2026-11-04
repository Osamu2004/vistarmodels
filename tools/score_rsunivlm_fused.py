"""CPU-only, restartable exclusive fusion and mIoU/3px-WFm rescoring."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import io as text_io
import multiprocessing
from pathlib import Path
import sys
import time
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
from tools import rsunivlm_fixed_order_fusion as fusion
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol

DEPENDENCIES = (
    "tools/score_rsunivlm_fused.py", "tools/rsunivlm_fixed_order_fusion.py",
    "tools/rsunivlm_native_semantic.py", "tools/rsunivlm_paper_protocol.py",
    "baselines/rsunivlm/protocols.py", "baselines/rsunivlm/evaluate.py",
    "baselines/segearth_ov/protocols.py", "baselines/flair_protocol.py",
    "baselines/binary_boundary_wfm.py", "baselines/levircd_common.py",
    "baselines/rskt_seg/xbd_label_utils.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    return {name: sha(ROOT / name) for name in DEPENDENCIES}


def sample_id(plan, dataset, item):
    return fingerprint([fingerprint([plan["identity"], dataset]), item["sample"], item["files"]])


def prepare(source, datasets, max_samples=0):
    plan = storage.read_json(source / "plan.json")
    config = storage.read_json(source / "run_config.json")
    summary = storage.read_json(source / "paper_summary.json")
    if not plan or not config or plan.get("identity") != fingerprint(plan.get("protocol")) or config.get("identity") != plan["identity"] or config.get("protocol") != plan["protocol"]:
        raise ValueError("Source plan/run_config identity differs or is missing")
    if not summary or not plan.get("gpu_ids"):
        raise ValueError("Missing source summary or shard assignments")
    if plan["protocol"].get("semantic_mode") != native.MODE:
        raise ValueError("Require saved native_classwise predictions")
    # Compare the old scorer/GT/native helper hashes, without loading models.
    for name, digest in code_identity().items():
        if name.startswith(("tools/score_rsunivlm_fused", "tools/rsunivlm_fixed_order_fusion")):
            continue
        if plan["protocol"].get("code_sha256", {}).get(name) != digest:
            raise ValueError(f"Source GT/scorer/query code differs: {name}")
    entries = {entry["dataset"]: entry for entry in plan["entries"]}
    if len(entries) != len(plan["entries"]) or set(datasets) - set(entries):
        raise ValueError("Duplicate or missing source dataset entries")
    chosen = []
    for dataset in datasets:
        entry = entries[dataset]
        if fingerprint(entry["samples"]) != plan["protocol"]["populations"].get(dataset):
            raise ValueError(f"Source sample manifest differs: {dataset}")
        population = entry["samples"][:plan["max_samples"]] if plan["max_samples"] else entry["samples"]
        result = summary.get("results", {}).get(dataset, {})
        if result.get("identity") != fingerprint([plan["identity"], dataset]) or result.get("complete") is not True or result.get("processed_samples") != len(population) or result.get("selected_samples") != len(population) or result.get("num_failed", 0):
            raise ValueError(f"Need a complete, identity-matched source dataset: {dataset}")
        classes = protocol.spec(dataset)["classes"]
        recorded = plan["protocol"].get("native_semantic", {}).get(dataset, {}).get("queries")
        expected = [dict(key=key, id=i, name=name) for key, i, name in native.queries(dataset)]
        if not recorded or [dict(key=q["key"], id=q["id"], name=q["name"]) for q in recorded] != expected:
            raise ValueError(f"Fixed query vocabulary/order differs: {dataset}")
        names = [row["sample"]["name"] for row in population]
        if len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
            raise ValueError("Invalid/duplicate sample output names")
        selected = population[:max_samples] if max_samples else population
        chosen.append(dict(dataset=dataset, samples=selected, source_population=len(population),
            source_query_miou=result["scores"].get("query_miou", result["scores"].get("miou")),
            discovered_samples=len(entry["samples"]), fusion=fusion.settings(dataset, classes)))
    versions = {}
    for package in ("numpy", "scipy", "Pillow", "opencv-python", "opencv-python-headless", "tifffile", "rasterio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    versions["cv2"] = cv2.__version__
    settings = dict(schema="rsunivlm_exclusive_fusion_v1", source_run=str(source),
        source_identity=plan["identity"], source_plan_sha256=sha(source / "plan.json"),
        source_summary_sha256=sha(source / "paper_summary.json"), code_sha256=code_identity(), versions=versions,
        datasets=datasets, max_samples=max_samples,
        populations={entry["dataset"]: fingerprint(entry["samples"]) for entry in chosen},
        fusion={entry["dataset"]: entry["fusion"] for entry in chosen},
        boundary="existing semantic IDGBR Sobel3+dilation3, ignore margin2, Margolin beta1/sigma5, min positive class-mean WFm",
        model_forward_calls=0, gpu_used=False, original_official_protocol=False)
    return plan, summary, chosen, dict(identity=fingerprint(settings), protocol=settings)


def source_masks(source, plan, dataset, item, position):
    name = item["sample"]["name"]
    rank = position % len(plan["gpu_ids"])
    folder = source / dataset / "shards" / f"rank{rank:03d}"
    path = folder / "receipts" / f"{name}.json"
    receipt = storage.read_json(path)
    if not receipt or receipt.get("identity") != sample_id(plan, dataset, item):
        raise ValueError(f"Missing/incompatible source prediction receipt: {path}")
    expected = native.queries(dataset)
    if set(receipt.get("sha256", {})) != {key for key, _, _ in expected}:
        raise ValueError("Missing/foreign class query hashes in receipt")
    shape = receipt.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("Invalid source native shape")

    def arrays():
        for key, index, _ in expected:
            mask_path = folder / "pred_mask" / f"{name}_{key}.png"
            if mask_path.is_symlink():
                raise ValueError(f"Unexpected prediction symlink: {mask_path}")
            with Image.open(mask_path) as opened:
                array = np.asarray(opened).copy()
            if array.dtype != np.uint8 or list(array.shape) != shape or not np.all(array < 2):
                raise ValueError(f"Invalid native binary source mask: {mask_path}")
            if hashlib.sha256(array.tobytes()).hexdigest() != receipt["sha256"][key]:
                raise ValueError(f"Source prediction content changed: {mask_path}")
            yield index, array
    return receipt, sha(path), arrays()


def score_prediction(prediction, target, dataset):
    classes = protocol.spec(dataset)["classes"]
    limit = len(classes) + int(dataset == "flair")
    if prediction.shape != target.shape or not np.all((prediction >= 0) & (prediction < limit)):
        raise ValueError("Invalid exclusive prediction")
    if not np.all(((target >= 0) & (target < len(classes))) | (target == 255)):
        raise ValueError("GT contains a class outside the recorded evaluation taxonomy")
    matrix = seg.confusion_matrix(prediction, target, num_classes=len(classes), ignore_index=255,
        prediction_ignore_index=len(classes) if dataset == "flair" else None)
    # Class12 is a real prediction-only background for FLAIR, never GT ignore.
    try:
        boundary = score_semantic_boundary_wfm(prediction, target, ignore_index=255,
            ignore_margin=2, num_classes=limit)
    except ValueError as error:
        # Fully GT-ignored images/support are not model failures. They add no
        # IoU counts and no positive/present boundary class to the mean.
        if str(error) not in ("Semantic boundary WFm requires at least one valid GT pixel",
                "No valid evaluation pixels remain after semantic ignore masking"):
            raise
        boundary = dict(wfm_3px_nonedge=None, wfm_3px_edge=None,
            wfm_3px_mean_present=None, wfm_3px_min_present=None,
            wfm_3px_valid_pixels=0, boundary_unavailable_reason=str(error))
    return dict(confusion_matrix=matrix.tolist(), **boundary)


def cached_row_valid(row):
    if not row or "row_payload_sha256" not in row:
        return False
    return row["row_payload_sha256"] == fingerprint({key: value for key, value in row.items() if key != "row_payload_sha256"})


def process(job):
    cv2.setNumThreads(1)
    started = time.monotonic()
    source, output = Path(job["source"]), Path(job["output"])
    dataset, item = job["dataset"], job["item"]
    sample = protocol.Sample.restore(item["sample"])
    destination = output / dataset
    row_path = destination / "rows" / f"{sample.name}.json"
    mask_path = destination / "pred_mask" / f"{sample.name}.png"
    try:
        if protocol.metadata(sample) != item["files"]:
            raise ValueError(f"Input files changed since original inference: {sample.name}")
        _, receipt_digest, arrays = source_masks(source, job["plan"], dataset, item, job["position"])
        identity = fingerprint([job["identity"], dataset, sample_id(job["plan"], dataset, item), receipt_digest])
        classes = protocol.spec(dataset)["classes"]
        fused, audit = fusion.fuse(arrays, num_classes=len(classes),
            uncovered_id=len(classes) if dataset == "flair" else 0)
        digest = hashlib.sha256(fused.tobytes()).hexdigest()
        saved = storage.read_json(row_path)
        if mask_path.is_symlink() or row_path.is_symlink():
            raise ValueError("Unexpected fused-output symlink")
        if cached_row_valid(saved) and saved.get("identity") == identity and saved.get("prediction_sha256") == digest and mask_path.is_file():
            with Image.open(mask_path) as opened:
                old = np.asarray(opened)
            if old.dtype == np.uint8 and old.shape == fused.shape and hashlib.sha256(old.tobytes()).hexdigest() == digest:
                return dict(ok=True, cached=True, row=saved)
        # Prediction fusion/persistence precede the first GT pixel read.
        storage.save_prediction(mask_path, fused)
        target = protocol.load_targets(sample, dataset, fused.shape)[0]
        scores = score_prediction(fused, target, dataset)
        if protocol.metadata(sample) != item["files"]:
            raise ValueError("Input files changed during scoring")
        row = dict(identity=identity, sample=sample.name, source_receipt_sha256=receipt_digest,
            prediction_sha256=digest, shape=list(fused.shape), fusion=audit,
            source_mask_encoding="uint8 0/1 per query", prediction_encoding="uint8 exclusive class ID",
            scoring_seconds=time.monotonic() - started, **scores)
        row["row_payload_sha256"] = fingerprint(row)
        storage.write_json(row_path, row)
        return dict(ok=True, cached=False, row=row)
    except Exception as error:
        return dict(ok=False, sample=sample.name, error=str(error), traceback=traceback.format_exc())


def bounded_results(executor, jobs, workers):
    """Only 2*workers jobs in flight; never decode a whole dataset into RAM."""
    if executor is None:
        yield from map(process, jobs)
        return
    from collections import deque
    pending, iterator = deque(), iter(jobs)
    for _ in range(2 * workers):
        job = next(iterator, None)
        if job is not None:
            pending.append(executor.submit(process, job))
    while pending:
        yield pending.popleft().result()
        job = next(iterator, None)
        if job is not None:
            pending.append(executor.submit(process, job))


def evaluate_entry(source, output, plan, config, entry, executor, workers):
    dataset = entry["dataset"]
    folder = output / dataset
    for name in ("rows", "pred_mask"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    # Do not send the full 20k-image source manifest to each worker/job.
    minimal_plan = {"identity": plan["identity"], "gpu_ids": plan["gpu_ids"]}
    jobs = (dict(source=str(source), output=str(output), plan=minimal_plan, identity=config["identity"],
        dataset=dataset, item=item, position=i) for i, item in enumerate(entry["samples"]))
    rows, failures, cached = [], [], 0
    progress = tqdm(bounded_results(executor, jobs, workers), total=len(entry["samples"]),
        desc=f"Fuse/score {dataset}", unit="image")
    for result in progress:
        if result["ok"]:
            rows.append(result["row"])
            cached += int(result["cached"])
        else:
            failures.append(result)
            print(f"[fusion failed] {dataset}/{result['sample']}: {result['error']}", flush=True)
        progress.set_postfix(scored=len(rows), cached=cached, failed=len(failures))
    matrix = sum((np.asarray(row["confusion_matrix"], np.int64) for row in rows), protocol.empty_confusion(dataset))
    scores = seg.metrics_for_dataset(matrix, dataset)
    if rows:
        scores.update(aggregate_binary_boundary_wfm(rows))
    scores.update(metric_scope="exclusive_multiclass", metric_protocol=fusion.POLICY)
    contract = protocol.spec(dataset)
    complete = len(rows) == len(entry["samples"]) and not failures
    full = complete and len(rows) == entry["source_population"] == entry["discovered_samples"] == contract["expected"] and plan["protocol"].get("strict_population", False)
    result = dict(dataset=dataset, identity=fingerprint([config["identity"], dataset]), complete=complete,
        is_full_benchmark=full, standard_metrics_available=True, official_upstream_benchmark_result=False,
        original_official_protocol=False, metric_scale="0..1; percent fields explicitly suffixed",
        selected_samples=len(entry["samples"]), processed_samples=len(rows), expected_samples=contract["expected"],
        num_failed=len(failures), failed_samples=failures, exhausted_population=True,
        scores_scope="all selected samples" if complete else "partial successful samples only",
        confusion_matrix=matrix.tolist(), scores=scores, contract=contract, fusion=entry["fusion"],
        source_run=str(source), source_identity=plan["identity"], source_query_miou=entry["source_query_miou"],
        cached_samples_this_invocation=cached,
        fusion_audit={field:sum(row["fusion"][field] for row in rows)
            for field in ("native_pixels", "uncovered_pixels", "overlapping_pixels")})
    if dataset == "flair":
        result["confusion_matrix_columns"] = contract["classes"] + ["prediction-only background"]
    storage.write_json(folder / "metrics.json", result)
    return result


def write_summary(output, original, updated, config):
    summary = copy.deepcopy(original)
    summary.update(schema="rsunivlm_fused_paper_summary_v1", identity=config["identity"],
        fusion_protocol=config["protocol"], updated_datasets=list(updated),
        inherited_datasets=[key for key in summary["results"] if key not in updated],
        note="Three semantic datasets use VISTAR fixed-order fusion; other results are copied with original metric scopes; not official Text2Seg/RSUniVLM results")
    for name, result in summary["results"].items():
        if name not in updated:
            result["inherited_from_source_run"] = config["protocol"]["source_run"]
    summary["results"].update(updated)
    summary["complete"] = all(value.get("complete") is True for value in summary["results"].values())
    summary["standard_metrics_available"] = all(value.get("standard_metrics_available") is True for value in summary["results"].values())
    summary["is_full_benchmark"] = set(summary["results"]) == set(protocol.DATASETS) and all(value.get("is_full_benchmark") is True for value in summary["results"].values())
    storage.write_json(output / "paper_summary.json", summary)
    buffer = text_io.StringIO()
    fields = ["dataset", "primary_metric", "value", "value_percent", "wfm_3px", "wfm_percent", "processed", "selected", "failed", "complete", "scope", "updated"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for name, result in summary["results"].items():
        scores = result["scores"]
        metric = result["contract"]["primary"]
        scope = scores.get("metric_scope", "original source protocol")
        label = "query_miou" if result.get("standard_metrics_available") is False and metric == "miou" else metric
        value, boundary = scores.get(metric), scores.get("wfm_3px")
        writer.writerow(dict(dataset=name, primary_metric=label, value=value,
            value_percent="" if value is None else f"{100*value:.2f}", wfm_3px=boundary,
            wfm_percent="" if boundary is None else f"{100*boundary:.2f}",
            processed=result["processed_samples"], selected=result["selected_samples"], failed=result.get("num_failed"),
            complete=result["complete"], scope=scope, updated=name in updated))
    storage.atomic_text(output / "paper_summary.csv", buffer.getvalue())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--datasets", default=",".join(fusion.DATASETS))
    parser.add_argument("--workers", type=int, default=2, help="CPU processes; 0 uses main process; GPU is never used")
    parser.add_argument("--max_samples", type=int, default=0, help="Development prefix per dataset; 0 means full source population")
    options = parser.parse_args(argv)
    datasets = options.datasets.split(",")
    if options.workers < 0 or options.max_samples < 0 or not datasets or len(datasets) != len(set(datasets)) or set(datasets) - set(fusion.DATASETS):
        parser.error("Choose distinct loveda/flair/uavid datasets and nonnegative workers/max_samples")
    source = Path(options.source_run).expanduser().resolve()
    output = Path(options.output_dir or (str(source) + "_fixedorder_fusion_v1")).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a separate sibling of the saved source run")
    with ExitStack() as stack:
        # Shared lock refuses an active original queue and never edits it.
        source_lock = stack.enter_context((source / ".queue.lock").open("rb"))
        fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan, original, entries, config = prepare(source, datasets, options.max_samples)
        output.mkdir(parents=True, exist_ok=True)
        target_lock = stack.enter_context((output / ".fusion.lock").open("a"))
        fcntl.flock(target_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = storage.read_json(output / "run_config.json")
        if previous and previous != config:
            raise ValueError("Fusion output belongs to another source/rule/code/environment; choose a new output directory")
        if not previous and any(path.name != ".fusion.lock" for path in output.iterdir()):
            raise ValueError("Refusing to adopt a nonempty output without a matching run_config")
        storage.write_json(output / "run_config.json", config)
        print(f"[fusion] CPU workers={options.workers}; no model/Qwen/GPU; OUTPUT_DIR={output}", flush=True)
        executor = None
        if options.workers:
            executor = stack.enter_context(ProcessPoolExecutor(max_workers=options.workers,
                mp_context=multiprocessing.get_context("spawn")))
        updated = {}
        for entry in entries:
            updated[entry["dataset"]] = evaluate_entry(source, output, plan, config, entry, executor, options.workers)
            # A checkpoint summary after each dataset, without treating pending fusion as finished.
            storage.write_json(output / "status.json", dict(identity=config["identity"], finished_datasets=list(updated),
                remaining_datasets=[name for name in datasets if name not in updated], complete=False))
        summary = write_summary(output, original, updated, config)
        storage.write_json(output / "runtime_finished.json", dict(identity=config["identity"], complete=summary["complete"],
            model_forward_calls=0, gpu_used=False, workers=options.workers))
        storage.write_json(output / "status.json", dict(identity=config["identity"], finished_datasets=list(updated),
            remaining_datasets=[], complete=summary["complete"]))
        print((output / "paper_summary.csv").read_text(), flush=True)
        print(f"[fusion] complete={summary['complete']}; summary={output/'paper_summary.json'}", flush=True)
        return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
