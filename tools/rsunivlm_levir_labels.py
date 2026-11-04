"""Lossless binary LEVIR label representations, with a CPU-only audit CLI.

Previously accepted 2-D IDs keep the original interpretation. Additional
representations require equal RGB channels, explicit binary values and opaque
alpha; arbitrary colors/intermediate grays are never thresholded or resized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

POLICY = 'preserve_2d_ids_and_lossless_binary_color_v1'


def settings():
    return dict(policy=POLICY, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy='2-D IDs in {0,1,255}: identical original >0 interpretation, including palette indices',
        additional='equal-channel RGB, opaque RGBA/LA, or binary rendered palette when legacy IDs were invalid',
        values=[0, 1, 255], alpha='255 only', spatial='native extent; no resize',
        gt_files_modified=False, changes_model_inputs=False)


def read_binary(path, shape=None):
    """Return exact 0/1 labels and diagnostics; reject ambiguous encodings."""
    path = Path(path)
    before = path.stat()
    with Image.open(path) as opened:
        raw = np.array(opened)
        mode = opened.mode
        values = np.unique(raw)
        details = dict(path=str(path), mode=mode, raw_shape=list(raw.shape), dtype=str(raw.dtype),
            distinct_values=int(len(values)), values_preview=values[:24].tolist())
        legacy = raw.ndim == 2 and bool(np.all(np.isin(raw, (0, 1, 255))))
        details['legacy_accepted'] = legacy
        if legacy:
            gray, encoding = raw, 'legacy_2d_ids'
        else:
            if mode == 'P':
                color = np.array(opened.convert('RGBA'))
                encoding = 'rendered_binary_palette'
            elif mode in ('RGB', 'RGBA', 'LA'):
                color = raw
                encoding = 'binary_'+mode.lower()
            else:
                raise ValueError('Unsupported LEVIR GT encoding: '+json.dumps(details, ensure_ascii=False))
            if mode == 'LA':
                opaque = color.ndim == 3 and color.shape[2] == 2 and bool(np.all(color[..., 1] == 255))
                gray = color[..., 0] if opaque else None
            else:
                opaque = (color.ndim == 3 and color.shape[2] in (3, 4) and
                          (color.shape[2] == 3 or bool(np.all(color[..., 3] == 255))))
                equal = opaque and bool(np.all(color[..., 0] == color[..., 1]) and
                                        np.all(color[..., 0] == color[..., 2]))
                gray = color[..., 0] if equal else None
            if gray is None or not np.all(np.isin(gray, (0, 1, 255))):
                raise ValueError('LEVIR GT is not an opaque exact binary representation: '+json.dumps(details, ensure_ascii=False))
        target = (gray > 0).astype(np.uint8)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'LEVIR GT changed during read: {path}')
    if shape is not None and target.shape != tuple(shape):
        raise ValueError(f'Native target extent mismatch: {path}: {target.shape} != {tuple(shape)}')
    details.update(encoding=encoding, target_shape=list(target.shape), positive_pixels=int(target.sum()),
                   mask_sha256=hashlib.sha256(target.tobytes()).hexdigest())
    return target, details


class _TargetProxy:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def __getattr__(self, name):
        return getattr(self.original, name)

    def load_targets(self, sample, dataset, shape):
        if dataset != 'levircd':
            return self.original.load_targets(sample, dataset, shape)
        target, details = read_binary(sample.labels[0], shape)
        self.stream.write(json.dumps(dict(sample=sample.name, **details), ensure_ascii=False)+'\n')
        self.stream.flush()
        return [target]


@contextmanager
def bind_targets(plan, rank, output):
    configured = plan['protocol'].get('levircd_gt_reader')
    if not configured:
        yield
        return
    if configured != settings():
        raise ValueError('LEVIR GT reader policy/source differs from the saved plan')
    from tools import rsunivlm_paper_worker as worker
    path = Path(output)/'levircd'/'shards'/f'rank{rank:03d}'/'gt_reading.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    original = worker.protocol
    with path.open('w', encoding='utf-8') as stream:
        worker.protocol = _TargetProxy(original, stream)
        try:
            yield
        finally:
            worker.protocol = original


def install_reader(engine_class, plan):
    if not plan['protocol'].get('levircd_gt_reader'):
        return
    original = engine_class.evaluate

    def evaluate(self, entry, output):
        if entry['dataset'] != 'levircd':
            return original(self, entry, output)
        with bind_targets(self.plan, self.rank, output):
            return original(self, entry, output)

    engine_class.evaluate = evaluate


def audit_run(source, output=None):
    source = Path(source).expanduser().resolve()
    plan = json.loads((source/'plan.json').read_text())
    entry = next((entry for entry in plan['entries'] if entry['dataset'] == 'levircd'), None)
    if entry is None:
        raise ValueError('The source plan does not contain LEVIR-CD')
    selected = entry['samples'][:plan['max_samples']] if plan['max_samples'] else entry['samples']
    records, failures = [], []
    for row in selected:
        sample = row['sample']
        try:
            for expected in row['files']:
                path = Path(expected['path'])
                if (path.stat().st_size, path.stat().st_mtime_ns) != (expected['size'], expected['mtime_ns']):
                    raise ValueError(f'Source input metadata changed: {path}')
            with Image.open(sample['images'][0]) as image:
                shape = (image.height, image.width)
            _, details = read_binary(sample['labels'][0], shape)
            records.append(dict(sample=sample['name'], **details))
        except (OSError, ValueError) as error:
            failures.append(dict(sample=sample['name'], path=sample['labels'][0], error=str(error)))
    report = dict(source_run=str(source), source_identity=plan['identity'], policy=settings(),
        selected=len(selected), legacy_accepted=sum(row['legacy_accepted'] for row in records),
        additional_lossless_accepted=sum(not row['legacy_accepted'] for row in records),
        rejected=len(failures), complete=not failures, records=records, failures=failures,
        model_or_cuda_called=False, gt_files_modified=False)
    output = Path(output).expanduser().resolve() if output else source.with_name(source.name+'_levir_label_audit.json')
    if output == source or source in output.parents:
        raise ValueError('Audit output must be outside the original run directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('selected', 'legacy_accepted', 'additional_lossless_accepted', 'rejected', 'complete')}, ensure_ascii=False), flush=True)
    for failure in failures[:8]:
        print(json.dumps(failure, ensure_ascii=False), flush=True)
    print(f'Label audit: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', required=True)
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    return 0 if audit_run(args.source_run, args.output or None)['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
