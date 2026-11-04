"""Opt-in RLE fallback for the last retry and exhausted legacy query caches."""
from __future__ import annotations

import copy
import gc
from pathlib import Path

from PIL import Image

from tools import rsunivlm_constrained_rle as rle


def prefill_exhausted(engine, entry, output):
    from baselines.rsunivlm import evaluate as io
    from baselines.rsunivlm.protocols import fingerprint
    from tools import retry_rsunivlm_failed as recovery
    from tools import rsunivlm_paper_protocol as protocol
    dataset = entry['dataset']
    root = Path(output)/dataset/'shards'/f'rank{engine.rank:03d}'
    metrics = io.read_json(root/'metrics.json')
    if not metrics or metrics.get('identity') != fingerprint([engine.plan['identity'], dataset]):
        raise ValueError('Constrained fallback requires matching CPU-prepass shard metrics')
    failed = {item['sample'] for item in metrics.get('failed_samples', [])}
    rows = entry['samples'][:engine.plan['max_samples']] if engine.plan['max_samples'] else entry['samples']
    assigned = rows[engine.rank::engine.world_size]
    size = engine.plan['protocol']['tile_size']
    budget = engine.plan['protocol']['format_retries']
    pending = {key:[] for key in recovery.keys(engine.plan, dataset)}

    def flush(key):
        jobs, pending[key] = pending[key], []
        try:
            for group, answers in engine.forward(jobs, recovery.instruction(engine.plan, dataset, key, budget)):
                for job, answer in zip(group, answers):
                    record = copy.deepcopy(job['record'])
                    attempt = dict(prompt_variant=len(record['attempts']),
                        prompt=recovery.instruction(engine.plan, dataset, key, budget),
                        exhausted_history_extra=True, **answer)
                    record['attempts'].append(attempt)
                    record.update(attempt)
                    io.write_json(job['path'], record)
                    try:
                        grid = recovery.decoded(engine.plan, dataset, key, attempt)
                    except (ValueError, KeyError) as error:
                        attempt['decode_error'] = str(error)
                        record['decode_error'] = str(error)
                    else:
                        record.pop('decode_error', None)
                        record['grid_shape'] = list(grid.shape)
                    io.write_json(job['path'], record)
                    engine._rle_counts[dataset]['extra_legacy_answers'] += 1
        finally:
            for job in jobs:
                for image in job['images']:
                    image.close()

    try:
        for item in assigned:
            sample = protocol.Sample.restore(item['sample'])
            if sample.name not in failed:
                continue
            if protocol.metadata(sample) != item['files']:
                raise ValueError(f'Inputs changed before constrained fallback: {sample.name}')
            images = protocol.load_images(sample, dataset)
            try:
                width, height = images[0].size
                sample_id = recovery.sample_id(engine.plan, dataset, item)
                for key in pending:
                    for y in range(0, height, size):
                        for x in range(0, width, size):
                            path = root/'raw_text'/f'{sample.name}_{key}_y{y}_x{x}.json'
                            record = io.read_json(path)
                            if not record:
                                continue
                            attempts = record.get('attempts', [])
                            if len(attempts) < budget+1 or any(attempt.get('generation', {}).get('rle_constraints') for attempt in attempts):
                                continue
                            if len(attempts) != budget+1:
                                raise ValueError(f'Unexpected unconstrained retry history length: {path}')
                            w, h = min(size, width-x), min(size, height-y)
                            spatial = dict(source_box=[x, y, x+w, y+h], extent=[width, height], tile_size=size, padding='black_right_bottom')
                            identity = fingerprint([sample_id, key, spatial])
                            if record.get('identity') != identity or record.get('spatial') != spatial:
                                raise ValueError(f'Constrained query cache identity differs: {path}')
                            recovery_attempt_valid = False
                            for attempt in attempts:
                                try:
                                    recovery.decoded(engine.plan, dataset, key, attempt)
                                except (ValueError, KeyError):
                                    pass
                                else:
                                    recovery_attempt_valid = True
                                    break
                            if recovery_attempt_valid:
                                continue
                            recovery.instruction(engine.plan, dataset, key, budget)  # Validate the fixed query key.
                            tiles = []
                            for image in images:
                                tile = Image.new('RGB', (size, size), (0, 0, 0))
                                crop = image.crop((x, y, x+w, y+h))
                                try:
                                    tile.paste(crop, (0, 0))
                                finally:
                                    crop.close()
                                tiles.append(tile)
                            pending[key].append(dict(path=path, identity=identity, key=key,
                                images=tiles, record=record, force_constrained=True))
                            if len(pending[key]) >= engine.effective_batch:
                                flush(key)
            finally:
                for image in images:
                    image.close()
        for key in pending:
            flush(key)
    finally:
        for jobs in pending.values():
            for job in jobs:
                for image in job['images']:
                    image.close()


def install(engine_class, plan, rank):
    configured = plan['protocol'].get('constrained_rle')
    if not configured:
        return
    if configured != rle.settings(configured['rows'], configured['cols']):
        raise ValueError('Constrained runtime differs from its recorded source/policy')
    import torch
    from baselines.rsunivlm import evaluate as io
    original_evaluate, original_forward = engine_class.evaluate, engine_class.forward

    def evaluate(self, entry, output):
        previous = getattr(self, '_rle_dataset', None)
        self._rle_dataset = dataset = entry['dataset']
        if not hasattr(self, '_rle_counts'):
            self._rle_counts = {}
        self._rle_counts.setdefault(dataset, dict(generated=0, extra_legacy_answers=0))
        try:
            with rle.bind_decoders(plan):
                prefill_exhausted(self, entry, output)
                result = original_evaluate(self, entry, output)
                rle.annotate_scored_samples(plan, dataset, rank, output, result)
                return result
        finally:
            self._rle_dataset = previous
            io.write_json(Path(output)/f'constrained_rle_rank{rank:03d}.json',
                dict(identity=plan['identity'], rank=rank, policy=configured, datasets=self._rle_counts))

    def forward(self, jobs, prompt):
        if not jobs:
            return
        budget = plan['protocol']['format_retries']
        active = [bool(job.get('force_constrained')) or len(job.get('attempts', [])) == budget for job in jobs]
        if not any(active):
            yield from original_forward(self, jobs, prompt)
            return
        if not all(active):
            raise ValueError('Do not mix free and constrained retries in one dispatch')
        dataset = self._rle_dataset
        keys = {job.get('key', job.get('direction')) for job in jobs}
        if len(keys) != 1:
            raise ValueError('Constrained batches require one fixed category prompt')
        key = next(iter(keys))
        offset = 0
        while offset < len(jobs):
            count = min(self.effective_batch, len(jobs)-offset)
            group, oom = jobs[offset:offset+count], False
            try:
                answers = rle.generate_answers(self.core(), [job['images'] for job in group], prompt, plan, dataset, key)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                if count == 1:
                    raise RuntimeError('GPU memory insufficient even at constrained batch1')
                gc.collect(); torch.cuda.empty_cache()
                self.effective_batch = max(1, count//2); self.oom_splits += 1
                continue
            if len(answers) != len(group):
                raise ValueError('Constrained generator returned a different batch size')
            self.actual_batches.add(count)
            self._rle_counts[dataset]['generated'] += count
            print(f'[RLE constrained] {dataset}/{key}: actual_batch={count}; grid={configured["rows"]}x{configured["cols"]}', flush=True)
            yield group, answers
            offset += count

    engine_class.evaluate, engine_class.forward = evaluate, forward
