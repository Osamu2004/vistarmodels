"""Run in WSL: synthetic CPU decoding, spawn, ordering and resume contracts.

No pretrained model or CUDA forward is used. These tests are not run on the Mac.
"""
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
import tifffile

from baselines.rsunivlm import evaluate as storage
from baselines.rsunivlm.protocols import fingerprint
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_data_loading as loading


def fixture(tmp_path, dataset='loveda', count=5, world=1):
    folder = tmp_path/'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        paths = []
        for temporal in range(2 if dataset in ('second', 'levircd') else 1):
            values = np.zeros((32, 530, 3), np.uint8)
            values[..., 0] = index*11+temporal
            values[..., 1] = np.arange(530, dtype=np.uint16)[None, :] % 256
            values[..., 2] = np.arange(32, dtype=np.uint8)[:, None]
            path = folder/f'image_{index}_{temporal}.png'
            if dataset == 'flair':
                path = path.with_suffix('.tif')
                extra = np.full((32, 530, 2), 99, np.uint8)
                tifffile.imwrite(path, np.concatenate((values, extra), axis=-1), photometric='minisblack')
            else:
                Image.fromarray(values).save(path)
            paths.append(str(path))
        labels = []
        for temporal in range(2 if dataset == 'second' else 1):
            path = folder/f'label_{index}_{temporal}.png'
            Image.fromarray(np.full((32, 530), (index+temporal) % 2, np.uint8)).save(path)
            labels.append(str(path))
        sample = protocol.Sample(f's{index}', tuple(paths), tuple(labels))
        rows.append(dict(sample=sample.serialize(), files=protocol.metadata(sample)))
    config = dict(batch_size=3, tile_size=512, format_retries=0, strict_population=False,
                  semantic_mode='native_classwise')
    entry = dict(dataset=dataset, samples=rows,
                 contract=native.contract(dataset) if native.enabled('native_classwise', dataset) else protocol.spec(dataset))
    plan = dict(identity=fingerprint(config), protocol=config, entries=[entry], max_samples=0,
                gpu_ids=[str(rank) for rank in range(world)], output_dir=str(tmp_path/'serial'))
    return plan, entry


def close_images(images):
    for image in images:
        image.close()


@pytest.mark.parametrize('dataset', ['loveda', 'flair', 'levircd', 'second'])
@pytest.mark.parametrize('num_workers', [0, 2])
def test_original_pixels_temporal_order_and_parent_rng_are_retained(tmp_path, dataset, num_workers):
    plan, entry = fixture(tmp_path, dataset, count=3)
    before = torch.get_rng_state().clone()
    reader = loading.OrderedRGBReader(entry, plan, 0, 1, loading.LoaderOptions(num_workers, 2, 60))
    try:
        assert reader.loader.batch_size is None and reader.loader.drop_last is False
        for row in entry['samples']:
            sample = protocol.Sample.restore(row['sample'])
            expected = protocol.load_images(sample, dataset)
            actual = reader.read(sample, dataset)
            try:
                assert len(actual) == len(expected)
                for first, second in zip(expected, actual):
                    assert first.mode == second.mode == 'RGB' and first.size == second.size
                    assert first.tobytes() == second.tobytes()
            finally:
                close_images(expected); close_images(actual)
    finally:
        reader.close()
    assert torch.equal(before, torch.get_rng_state())
    assert reader.statistics()['consumed_sources'] == 3
    assert reader.statistics()['cleanup_error'] is None


def test_two_shards_keep_original_membership_without_padding_or_duplicates(tmp_path):
    plan, entry = fixture(tmp_path, count=5, world=2)
    names = []
    for rank in range(2):
        reader = loading.OrderedRGBReader(entry, plan, rank, 2, loading.LoaderOptions(0))
        shard = []
        try:
            for row in entry['samples'][rank::2]:
                sample = protocol.Sample.restore(row['sample'])
                close_images(reader.read(sample, 'loveda'))
                shard.append(sample.name)
        finally:
            reader.close()
        assert shard == (['s0', 's2', 's4'] if rank == 0 else ['s1', 's3'])
        names.extend(shard)
    assert len(set(names)) == len(names) == 5


def test_prefetch_does_not_open_gt_and_detects_changed_inputs(tmp_path, monkeypatch):
    plan, entry = fixture(tmp_path, count=2)
    labels = {Path(row['sample']['labels'][0]) for row in entry['samples']}
    original_open = Image.open
    def rgb_only(path, *args, **kwargs):
        assert Path(path) not in labels, 'DataLoader opened GT pixels'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', rgb_only)
    item = loading.RGBSourceDataset('loveda', entry['samples'])[0]
    assert all(value.device.type == 'cpu' and value.dtype == torch.uint8 for value in item['rgb'])
    image = Path(entry['samples'][1]['sample']['images'][0])
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Inputs changed'):
        loading.RGBSourceDataset('loveda', entry['samples'])[1]


def install_fake_runtime(monkeypatch, calls, active_output, entry):
    monkeypatch.setattr(worker.Engine, 'core', lambda self: None)
    monkeypatch.setattr(worker, 'score_semantic_boundary_wfm', lambda *a, **k: {})
    monkeypatch.setattr(native, 'score_semantic_boundary_wfm', lambda *a, **k: {})
    def generate(core, requests, prompt):
        calls.append((prompt, [[hashlib.sha256(image.tobytes()).hexdigest() for image in request]
                              for request in requests]))
        return [dict(response='\n'.join(['others*24']*23),
                     generation=dict(ended_with_eos=True, hit_token_limit=False)) for _ in requests]
    monkeypatch.setattr(worker, 'generate_requests', generate)
    def target(sample, dataset, shape):
        keys = [key for key, _, _ in native.queries(dataset)] if native.enabled('native_classwise', dataset) else worker.outputs(dataset)
        folder = active_output['path']/dataset/'shards'/'rank000'/'pred_mask'
        assert all((folder/f'{sample.name}_{key}.png').is_file() for key in keys), 'GT was read before complete predictions'
        values = []
        for path in sample.labels:
            with Image.open(path) as opened:
                values.append(np.asarray(opened).copy())
        return values
    monkeypatch.setattr(protocol, 'load_targets', target)


@pytest.mark.parametrize('dataset', protocol.DATASETS)
def test_all_seven_datasets_keep_query_batches_masks_scores_and_cached_resume(tmp_path, monkeypatch, dataset):
    plan, entry = fixture(tmp_path, dataset, count=4)
    calls, active = [], dict(path=Path(plan['output_dir']))
    install_fake_runtime(monkeypatch, calls, active, entry)
    expected = worker.Engine(plan, 0, 1).evaluate(entry, str(active['path']))
    original_calls = list(calls)
    serial_masks = {str(path.relative_to(active['path'])): path.read_bytes()
                    for path in active['path'].rglob('pred_mask/*.png')}
    calls.clear()
    parallel = tmp_path/'parallel'
    active['path'] = parallel
    engine = worker.Engine(plan, 0, 1)
    with loading.bind_ordered_reader(engine, entry, str(parallel), loading.LoaderOptions(2, 2, 60)):
        actual = engine.evaluate(entry, str(parallel))
    assert calls == original_calls
    assert actual['scores'] == expected['scores'] and actual['complete'] == expected['complete']
    assert {str(path.relative_to(parallel)): path.read_bytes() for path in parallel.rglob('pred_mask/*.png')} == serial_masks
    report = storage.read_json(parallel/dataset/'shards/rank000/data_loading.json')
    assert report['identity'] == plan['identity'] and report['consumed_sources'] == 4
    calls.clear()
    with loading.bind_ordered_reader(engine, entry, str(parallel), loading.LoaderOptions(0)):
        resumed = engine.evaluate(entry, str(parallel))
    assert calls == [] and resumed['scores'] == expected['scores']
    assert all(row['reused_prediction'] for row in resumed['rows'])


def test_partial_raw_resume_and_exhausted_format_failures_keep_original_decisions(tmp_path, monkeypatch):
    plan, entry = fixture(tmp_path, 'loveda', count=3)
    serial = Path(plan['output_dir'])
    calls, active = [], dict(path=serial)
    install_fake_runtime(monkeypatch, calls, active, entry)
    worker.Engine(plan, 0, 1).evaluate(entry, str(serial))
    shard = serial/'loveda/shards/rank000'
    for name in ('s1', 's2'):
        (shard/'receipts'/f'{name}.json').unlink()
        for path in (shard/'pred_mask').glob(f'{name}_*.png'):
            path.unlink()
    invalid = next((shard/'raw_text').glob('s1_*.json'))
    record = storage.read_json(invalid)
    record['attempts'][0]['response'] = record['response'] = 'road**2'
    record['decode_error'] = 'synthetic exhausted format failure'
    storage.write_json(invalid, record)
    next((shard/'raw_text').glob('s2_*.json')).unlink()
    parallel = tmp_path/'resumed_parallel'
    shutil.copytree(serial, parallel)
    calls.clear()
    expected = worker.Engine(plan, 0, 1).evaluate(entry, str(serial))
    expected_calls = list(calls)
    calls.clear(); active['path'] = parallel
    engine = worker.Engine(plan, 0, 1)
    with loading.bind_ordered_reader(engine, entry, str(parallel), loading.LoaderOptions(2, 2, 60)):
        actual = engine.evaluate(entry, str(parallel))
    assert calls == expected_calls and len(calls) == 1
    assert actual['scores'] == expected['scores']
    assert [row['sample'] for row in actual['failed_samples']] == ['s1']
    assert [row['sample'] for row in actual['rows']] == ['s0', 's2']
    assert not list((parallel/'loveda/shards/rank000/pred_mask').glob('s1_*.png'))


def test_order_error_and_exception_restore_protocol_bindings_and_close_workers(tmp_path):
    plan, entry = fixture(tmp_path, count=2)
    engine = worker.Engine(plan, 0, 1)
    old_worker, old_native = worker.protocol, native.protocol
    with pytest.raises(ValueError, match='order'):
        with loading.bind_ordered_reader(engine, entry, plan['output_dir'], loading.LoaderOptions(0)) as reader:
            reader.read(protocol.Sample.restore(entry['samples'][1]['sample']), 'loveda')
    assert worker.protocol is old_worker and native.protocol is old_native
    assert reader.loader is None and reader.iterator is None


def test_partial_pass_closes_spawned_workers_and_restores_modules(tmp_path):
    plan, entry = fixture(tmp_path, count=3)
    engine = worker.Engine(plan, 0, 1)
    old_worker, old_native = worker.protocol, native.protocol
    processes = []
    with pytest.raises(RuntimeError, match='stop partial pass'):
        with loading.bind_ordered_reader(engine, entry, plan['output_dir'], loading.LoaderOptions(2, 2, 60)) as reader:
            sample = protocol.Sample.restore(entry['samples'][0]['sample'])
            close_images(reader.read(sample, 'loveda'))
            processes = list(reader.iterator._workers)
            raise RuntimeError('stop partial pass')
    assert processes and all(not process.is_alive() for process in processes)
    assert reader.cleanup_error is None
    assert worker.protocol is old_worker and native.protocol is old_native


def test_bad_prefetched_future_image_does_not_reorder_or_drop_an_earlier_image(tmp_path):
    plan, entry = fixture(tmp_path, count=3)
    broken = Path(entry['samples'][1]['sample']['images'][0])
    broken.write_bytes(b'not an image')
    sample = protocol.Sample.restore(entry['samples'][1]['sample'])
    entry['samples'][1]['files'] = protocol.metadata(sample)
    reader = loading.OrderedRGBReader(entry, plan, 0, 1, loading.LoaderOptions(2, 2, 60))
    try:
        first = protocol.Sample.restore(entry['samples'][0]['sample'])
        close_images(reader.read(first, 'loveda'))
        assert reader.position == 1
        with pytest.raises(Exception, match='image'):
            reader.read(sample, 'loveda')
        assert reader.position == 1
    finally:
        reader.close()


def test_zero_source_shard_and_invalid_options(tmp_path):
    plan, entry = fixture(tmp_path, count=1, world=2)
    reader = loading.OrderedRGBReader(entry, plan, 1, 2, loading.LoaderOptions(2))
    reader.close()
    assert reader.position == 0 and reader.statistics()['assigned_sources'] == 0
    for options in (dict(num_workers=-1), dict(prefetch_factor=0), dict(timeout=-1), dict(timeout=float('inf'))):
        with pytest.raises(ValueError):
            loading.LoaderOptions(**options)


def test_installed_entrypoint_hook_and_env_options(tmp_path, monkeypatch):
    plan, entry = fixture(tmp_path, count=2)
    calls, active = [], dict(path=Path(plan['output_dir']))
    install_fake_runtime(monkeypatch, calls, active, entry)
    monkeypatch.setenv('RSUNIVLM_NUM_WORKERS', '0')
    monkeypatch.setenv('RSUNIVLM_PREFETCH_FACTOR', '3')
    monkeypatch.setenv('RSUNIVLM_LOADER_TIMEOUT', '45')
    options = loading.LoaderOptions.from_env()
    assert (options.num_workers, options.prefetch_factor, options.timeout) == (0, 3, 45)
    class HookedEngine(worker.Engine):
        pass
    loading.install_data_loading(HookedEngine, options)
    result = HookedEngine(plan, 0, 1).evaluate(entry, plan['output_dir'])
    assert result['complete'] and calls
    report = storage.read_json(active['path']/'loveda/shards/rank000/data_loading.json')
    assert report['options']['num_workers'] == 0 and report['consumed_sources'] == 2
