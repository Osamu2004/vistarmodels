"""WSL-only exact GT representation and saved-mask rescoring regressions."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from tools import eval_rsunivlm_guarded as guarded
from tools import eval_rsunivlm_paper_queue as queue
from tools import repair_rsunivlm_format as repair
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_levir_labels as labels
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


def binary():
    return np.asarray([[0, 255, 1], [255, 0, 0]], dtype=np.uint8)


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA', 'LA', '1'])
def test_exact_binary_representations_keep_pixels_and_native_extent(tmp_path, mode):
    gray = binary()
    if mode == 'RGB':
        raw = np.repeat(gray[..., None], 3, axis=2)
    elif mode == 'RGBA':
        raw = np.concatenate((np.repeat(gray[..., None], 3, axis=2), np.full((*gray.shape, 1), 255, np.uint8)), axis=2)
    elif mode == 'LA':
        raw = np.stack((gray, np.full_like(gray, 255)), axis=2)
    else:
        raw = gray
    image = Image.fromarray(raw)
    if mode == '1':
        image = Image.fromarray((gray > 0).astype(np.uint8)*255).convert('1')
    path = tmp_path/'label.png'; image.save(path)
    before = path.read_bytes()
    result, details = labels.read_binary(path, gray.shape)
    np.testing.assert_array_equal(result, (gray > 0).astype(np.uint8))
    assert details['legacy_accepted'] == (mode in ('L', '1'))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='extent mismatch'):
        labels.read_binary(path, (3, 2))


def test_palette_fallback_and_already_valid_index_semantics_are_separate(tmp_path):
    raw = np.asarray([[2, 3], [3, 2]], np.uint8)
    palette = [0]*768; palette[9:12] = [255]*3
    image = Image.fromarray(raw).convert('P'); image.putpalette(palette)
    path = tmp_path/'palette.png'; image.save(path)
    result, details = labels.read_binary(path)
    assert details['encoding'] == 'rendered_binary_palette'
    np.testing.assert_array_equal(result, (raw == 3).astype(np.uint8))
    # When old IDs were already accepted, keep the old index interpretation.
    raw = np.asarray([[0, 1], [1, 0]], np.uint8)
    image = Image.fromarray(raw).convert('P'); image.putpalette([255]*3+[0]*765)
    image.save(path)
    result, details = labels.read_binary(path)
    assert details['legacy_accepted'] and details['encoding'] == 'legacy_2d_ids'
    np.testing.assert_array_equal(result, raw)


@pytest.mark.parametrize('kind', ['gray127', 'rgb127', 'colored', 'rgba_transparent', 'la_transparent'])
def test_ambiguous_values_colors_or_alpha_are_not_thresholded(tmp_path, kind):
    gray = binary()
    if kind == 'gray127':
        raw = gray.copy(); raw[0, 0] = 127
    elif kind == 'rgb127':
        raw = np.repeat(gray[..., None], 3, axis=2); raw[0, 0] = 127
    elif kind == 'colored':
        raw = np.repeat(gray[..., None], 3, axis=2); raw[0, 0] = [255, 0, 0]
    elif kind == 'rgba_transparent':
        raw = np.concatenate((np.repeat(gray[..., None], 3, axis=2), np.zeros((*gray.shape, 1), np.uint8)), axis=2)
    else:
        raw = np.stack((gray, np.zeros_like(gray)), axis=2)
    path = tmp_path/'bad.png'; Image.fromarray(raw).save(path)
    with pytest.raises(ValueError, match='encoding|exact binary'):
        labels.read_binary(path)


def saved_levir_run(tmp_path, monkeypatch):
    source = tmp_path/'original'; source.mkdir()
    samples = []
    target = np.zeros((32, 40), np.uint8); target[:, 30:] = 1
    for index in range(4):
        a, b, gt = (tmp_path/f'{name}{index}.png' for name in ('a', 'b', 'gt'))
        Image.new('RGB', (40, 32), (index, 0, 0)).save(a)
        Image.new('RGB', (40, 32), (index+100, 0, 0)).save(b)
        raw = target*255
        if index in (1, 2):
            raw = np.repeat(raw[..., None], 3, axis=2)
        Image.fromarray(raw).save(gt)
        samples.append(protocol.Sample(f's{index}', (str(a), str(b)), (str(gt),)))
    entry = dict(dataset='levircd', contract=dict(protocol.spec('levircd'), expected=4),
        samples=[dict(sample=sample.serialize(), files=protocol.metadata(sample)) for sample in samples])
    value = dict(batch_size=16, tile_size=512, format_retries=0, strict_population=False,
        semantic_mode=native.MODE, world_size=2, datasets=['levircd'], code_sha256=queue.source_identity(),
        populations={'levircd':fingerprint(entry['samples'])}, prompts={'levircd':[protocol.prompt('levircd')]},
        prediction_background={}, native_semantic={}, second_query=None, levircd_query={'direction':'forward'})
    plan = dict(protocol=value, entries=[entry], gpu_ids=['0', '1'], max_samples=0,
                output_dir=str(source), identity=fingerprint(value))
    for item in entry['samples']:
        name = item['sample']['name']; rank = int(name[1:]) % 2
        folder = source/'levircd/shards'/f'rank{rank:03d}'
        (folder/'pred_mask').mkdir(parents=True, exist_ok=True)
        io.save_prediction(folder/'pred_mask'/f'{name}_forward.png', target)
        io.write_json(folder/'receipts'/f'{name}.json', dict(identity=recovery.sample_id(plan, 'levircd', item),
            shape=list(target.shape), sha256={'forward':hashlib.sha256(target.tobytes()).hexdigest()}, descriptor_shapes=[[23, 24]]))
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-mask recovery must not execute model/CUDA operations')
    monkeypatch.setattr(worker, 'generate_requests', forbidden)
    monkeypatch.setattr(guarded, 'launch', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)
    for rank in range(2):
        with pytest.raises(ValueError, match='Expected binary native LEVIR-CD GT'):
            worker.Engine(plan, rank, 2).evaluate(entry, str(source))
    io.write_json(source/'plan.json', plan)
    io.write_json(source/'run_config.json', dict(identity=plan['identity'], protocol=value))
    io.write_json(source/'runtime_finished.json', dict(exit_codes=[1, 1]))
    (source/'.queue.lock').touch()
    return source, plan, entry, samples


def test_interrupted_gt_scoring_reuses_all_saved_masks_and_resumes_cpu_only(tmp_path, monkeypatch):
    source, plan, entry, samples = saved_levir_run(tmp_path, monkeypatch)
    original = {str(path.relative_to(source)):path.read_bytes() for path in source.rglob('*') if path.is_file()}
    output = tmp_path/'fixed'
    monkeypatch.setenv('RSUNIVLM_NUM_WORKERS', '0')
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output)])
    assert repair.main() == 0
    report = io.read_json(output/'format_repair_report.json')
    assert report['offline_complete'] and report['imported']['reused_complete_samples'] == 4
    assert report['offline_model_forward_calls'] == 0
    summary = io.read_json(output/'paper_summary.json')
    assert summary['results']['levircd']['processed_samples'] == 4
    assert summary['results']['levircd']['scores']['building_iou'] == pytest.approx(1.)
    assert io.read_json(output/'plan.json')['protocol']['levircd_gt_reader'] == labels.settings()
    for rank in range(2):
        rows = [json.loads(line) for line in (output/'levircd/shards'/f'rank{rank:03d}'/'gt_reading.jsonl').read_text().splitlines()]
        assert len(rows) == 2 and sum(not row['legacy_accepted'] for row in rows) == 1
    assert {str(path.relative_to(source)):path.read_bytes() for path in source.rglob('*') if path.is_file()} == original
    assert repair.main() == 0 and io.read_json(output/'paper_summary.json')['complete']
    with pytest.raises(ValueError, match='Expected binary native'):
        protocol.load_targets(samples[1], 'levircd', (32, 40))


def test_label_audit_covers_manifest_without_modifying_gt_or_source(tmp_path, monkeypatch):
    source, plan, entry, samples = saved_levir_run(tmp_path, monkeypatch)
    report = labels.audit_run(source)
    assert report['complete'] and report['selected'] == 4
    assert report['legacy_accepted'] == 2 and report['additional_lossless_accepted'] == 2
    assert not report['model_or_cuda_called'] and not report['gt_files_modified']
    with pytest.raises(ValueError, match='outside'):
        labels.audit_run(source, source/'audit.json')
    before = worker.protocol
    bad = copy.deepcopy(plan); bad['protocol']['levircd_gt_reader'] = dict(labels.settings(), helper_sha256='changed')
    with pytest.raises(ValueError, match='differs'):
        with labels.bind_targets(bad, 0, tmp_path/'bad'):
            pass
    assert worker.protocol is before
