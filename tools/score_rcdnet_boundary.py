"""Score native-resolution RCDNet LEVIR-CD masks with the shared 3px WFm.

Reads an existing RCDNet adapter output. No model or Torch is imported;
predictions, ground truth, and original metrics are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.levircd_common import binary_counts, binary_metrics, resolve_path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_mask(path: Path, expected_size: int) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.shape != (expected_size, expected_size):
        raise ValueError(f"Expected a {expected_size}x{expected_size} grayscale mask: {path}; got {raw.shape}")
    if not np.all((raw == 0) | (raw == 1) | (raw == 255)):
        raise ValueError(f"Mask must contain saved binary labels 0/1 or 0/255: {path}")
    return (raw > 0).astype(np.uint8)


def score_saved_rcdnet(eval_dir: Path, expected_samples: int = 128,
                       expected_size: int = 1024) -> tuple[dict, list[dict]]:
    from baselines.binary_boundary_wfm import (
        aggregate_binary_boundary_wfm,
        score_binary_boundary_wfm,
    )
    if expected_samples <= 0 or expected_size <= 0:
        raise ValueError("Expected sample count and native size must be positive")
    eval_dir = resolve_path(eval_dir)
    source_metrics = eval_dir / 'metrics.json'
    source_rows = eval_dir / 'per_image_metrics.jsonl'
    original = json.loads(source_metrics.read_text())
    if (original.get('method') != 'RCDNet' or original.get('dataset') != 'LEVIR-CD'
            or original.get('split') != 'test'):
        raise ValueError("Require RCDNet adapter metrics for LEVIR-CD test")
    if any(original.get(key) != expected_samples for key in
           ('num_dataset_pairs', 'num_evaluated_pairs')):
        raise ValueError("Original evaluation population is incomplete or differs from expected_samples")

    saved_rows = [json.loads(line) for line in source_rows.read_text().splitlines() if line.strip()]
    names = [row['name'] for row in saved_rows]
    if len(names) != expected_samples or len(set(names)) != expected_samples:
        raise ValueError("Original per-image results have missing or duplicate samples")
    predictions = {path.name.removesuffix('_pred_mask.png'): path
                   for path in (eval_dir / 'pred_mask').glob('*_pred_mask.png')}
    targets = {path.stem: path for path in (eval_dir / 'gt_mask').glob('*.png')}
    if set(predictions) != set(names) or set(targets) != set(names):
        raise ValueError("pred_mask, gt_mask, and original per-image sample names must match exactly")

    sources = [source_metrics, source_rows] + list(predictions.values()) + list(targets.values())
    fingerprints = {str(path): sha(path) for path in sources}
    expected_rows = {row['name']: row for row in saved_rows}
    totals = dict(tp=0, tn=0, fp=0, fn=0)
    rows = []
    for index, name in enumerate(sorted(names), 1):
        prediction = read_mask(predictions[name], expected_size)
        target = read_mask(targets[name], expected_size)
        counts = binary_counts(prediction, target)
        if any(counts[key] != expected_rows[name].get(key) for key in totals):
            raise ValueError(f"Saved predictions/GT disagree with original per-image counts: {name}")
        for key, value in counts.items():
            totals[key] += value
        rows.append(dict(name=name, **counts,
                         **score_binary_boundary_wfm(prediction, target)))
        if index % 16 == 0 or index == expected_samples:
            print(f"RCDNet boundary WFm: {index}/{expected_samples}", file=sys.stderr)

    if any(totals[key] != original.get(key) for key in totals):
        raise ValueError("Recomputed global counts differ from original RCDNet metrics")
    recomputed = binary_metrics(totals)
    for key in ('iou_change', 'f1_score_change'):
        if not math.isclose(recomputed[key], float(original[key]), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Recomputed {key} differs from original RCDNet metrics")
    if any(sha(Path(path)) != digest for path, digest in fingerprints.items()):
        raise ValueError("An input prediction, GT, or result file changed during scoring")

    scores = aggregate_binary_boundary_wfm(rows)
    summary = dict(
        complete=True, method='RCDNet', dataset='LEVIR-CD', split='test',
        source_run=str(eval_dir), processed_samples=len(rows), num_failed=0,
        is_full_benchmark=(expected_samples == 128 and expected_size == 1024),
        native_size=expected_size, checkpoint=original.get('checkpoint'),
        checkpoint_release=original.get('checkpoint_release'),
        source_protocol=original.get('protocol'), architecture=original.get('architecture'),
        scores=scores, iou_verification=dict(matches_source=True, counts=totals,
            iou_change=recomputed['iou_change'], f1_score_change=recomputed['f1_score_change']),
        source_sha256=fingerprints,
        scorer_sha256=sha(Path(__file__)),
        shared_metric_sha256=sha(ROOT / 'baselines/binary_boundary_wfm.py'),
        shared_binary_io_sha256=sha(ROOT / 'baselines/levircd_common.py'),
    )
    return summary, rows


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_dir', required=True, help='Existing RCDNet output with pred_mask, gt_mask, and original metrics')
    parser.add_argument('--output_dir', help='Defaults to <eval_dir>/boundary_wfm_3px')
    parser.add_argument('--expected_samples', type=int, default=128)
    parser.add_argument('--expected_size', type=int, default=1024)
    args = parser.parse_args()
    source = resolve_path(args.eval_dir)
    output = resolve_path(args.output_dir) if args.output_dir else source / 'boundary_wfm_3px'
    summary, rows = score_saved_rcdnet(source, args.expected_samples, args.expected_size)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'per_image_boundary.json', rows)
    write_json(output / 'paper_summary.json', summary)
    print(json.dumps(dict(complete=summary['complete'], processed_samples=len(rows),
        iou_change_percent=100 * summary['iou_verification']['iou_change'],
        wfm_3px_percent=summary['scores']['wfm_3px_percent'],
        paper_summary=str(output / 'paper_summary.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
