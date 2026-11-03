"""WSL-only CPU saved-answer recovery; no weights, model forward or CUDA."""
import copy
import fcntl
from pathlib import Path
import sys

import pytest
import torch

from baselines.rsunivlm import evaluate as io
from tools import eval_rsunivlm_guarded as guarded
from tools import repair_rsunivlm_format as tool
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_paper_worker as worker
from test_rsunivlm_failed_recovery import completed_source


def forbidden(*args, **kwargs):
    raise AssertionError('Offline repair must not perform model/CUDA operations')


def setup(tmp_path, monkeypatch, dataset='flair'):
    source, plan, entry, calls, active, failures = completed_source(tmp_path, monkeypatch, dataset)
    (source/'.queue.lock').touch()
    if dataset == 'second':
        key = recovery.keys(plan, dataset)[0]
        path = source/dataset/'shards/rank000/raw_text'/f's0_{key}_y0_x0.json'
        record = io.read_json(path)
        text = '\n'.join(['water**24']*23)
        record['response'] = record['attempts'][0]['response'] = text
        io.write_json(path, record)
    monkeypatch.setenv('RSUNIVLM_NUM_WORKERS', '0')
    monkeypatch.setattr(worker, 'generate_requests', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)
    monkeypatch.setattr(guarded, 'launch', forbidden)
    return source, plan, entry, active


@pytest.mark.parametrize('dataset', ['flair', 'second'])
def test_repair_rebuild_score_and_resume_without_gpu_or_source_changes(tmp_path, monkeypatch, dataset):
    source, plan, entry, active = setup(tmp_path, monkeypatch, dataset)
    original = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    output = tmp_path/'fixed'; active[0] = output
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output)])
    assert tool.main() == 0
    config = io.read_json(output/'run_config.json')
    assert config['identity'] != plan['identity']
    assert config['protocol']['format_retries'] == 0
    assert config['protocol']['code_sha256'] == plan['protocol']['code_sha256']
    report = io.read_json(output/'format_repair_report.json')
    assert report['normalization']['repaired_records'] == 1
    assert report['imported']['reused_complete_samples'] == 3
    assert report['offline_complete'] and report['offline_model_forward_calls'] == 0
    assert not report['gpu_continuation_invoked'] and not report['gpu_workers_started']
    summary = io.read_json(output/'paper_summary.json')
    assert summary['complete'] and summary['results'][dataset]['processed_samples'] == 4
    key = recovery.keys(plan, dataset)[0]
    fixed = io.read_json(output/dataset/'shards/rank000/raw_text'/f's0_{key}_y0_x0.json')
    assert fixed['attempts'][0]['format_repair']['original_response'] == ('building:1' if dataset == 'flair' else '\n'.join(['water**24']*23))
    assert len(fixed['attempts']) == 1
    assert {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()} == original
    assert tool.main() == 0
    again = io.read_json(output/'format_repair_report.json')
    assert again['normalization']['repaired_records'] == 0
    assert again['normalization']['previously_repaired_records'] == 1
    assert io.read_json(output/'paper_summary.json')['complete']


def test_unrepairable_and_missing_answers_stay_partial_without_fake_attempts(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    key = recovery.keys(plan, 'flair')[0]
    raw = source/'flair/shards/rank000/raw_text'/f's0_{key}_y0_x0.json'
    record = io.read_json(raw)
    record['response'] = record['attempts'][0]['response'] = 'building**24\nothers*23'
    io.write_json(raw, record)
    missing = source/'flair/shards/rank000/raw_text'/f's0_{recovery.keys(plan,"flair")[1]}_y0_x0.json'
    missing.unlink()
    output = tmp_path/'partial'; active[0] = output
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output)])
    assert tool.main() == 0
    report = io.read_json(output/'format_repair_report.json')
    assert not report['offline_complete'] and report['normalization']['repaired_records'] == 0
    assert report['normalization']['unrepairable_records'] == 1 and report['deferred_queries'] == 1
    assert not (output/'flair/shards/rank000/raw_text'/missing.name).exists()
    assert not list((output/'flair/shards/rank000/pred_mask').glob('s0_*.png'))
    result = io.read_json(output/'flair/shards/rank000/metrics.json')
    assert not result['exhausted_population'] and result['failed_samples'][0]['reason'] == 'offline_deferred_queries'
    assert result['processed'] == 1 and result['model_forward_calls_this_process'] == 0


def test_cuda_interrupted_population_can_be_repaired_offline_and_remains_incomplete(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    shard = source/'flair/shards/rank000'
    metrics = io.read_json(shard/'metrics.json')
    metrics.update(exhausted_population=False, error='CUDA error: an illegal memory access was encountered')
    io.write_json(shard/'metrics.json', metrics)
    (shard/'receipts/s2.json').unlink()
    for path in (shard/'pred_mask').glob('s2_*.png'):
        path.unlink()
    for path in (shard/'raw_text').glob('s2_*.json'):
        path.unlink()
    output = tmp_path/'interrupted'; active[0] = output
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output)])
    assert tool.main() == 0
    report = io.read_json(output/'format_repair_report.json')
    assert report['normalization']['repaired_records'] == 1 and not report['offline_complete']
    assert report['deferred_queries'] == len(recovery.keys(plan, 'flair'))
    with pytest.raises(ValueError, match='non-format error'):
        recovery.prepare(output, ['flair'])


def test_plan_only_source_lock_protocol_and_distinct_output_protection(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    output = tmp_path/'not-created'
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output), '--plan_only'])
    assert tool.main() == 0 and not output.exists()
    with (source/'.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match='still running'):
            tool.main()
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(source)])
    with pytest.raises(ValueError, match='separate'):
        tool.main()
    monkeypatch.setattr(tool.queue, 'source_identity', lambda: {'changed': 'source'})
    with pytest.raises(ValueError, match='Numerical source'):
        tool.prepare(source, [])


def test_old_valid_attempt_is_not_replaced_by_an_earlier_repair(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    key = recovery.keys(plan, 'flair')[0]
    raw = source/'flair/shards/rank000/raw_text'/f's0_{key}_y0_x0.json'
    record = io.read_json(raw)
    old = record['attempts'][0]
    second = dict(copy.deepcopy(old), response='others*24', prompt_variant=1,
                  prompt=recovery.instruction(plan, 'flair', key, 1))
    # Source budget and identities must be rebuilt for a real two-attempt history;
    # history-only selection is already covered without changing saved-run identity.
    fixed, status = tool.normalization.repair_record(dict(attempts=[old, second]),
        lambda attempt: recovery.decoded(plan, 'flair', key, attempt))
    assert status['status'] == 'already_valid' and fixed['attempts'][0] == old


def test_gpu_continuation_is_explicit_and_uses_guarded_entrypoint(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    key = recovery.keys(plan, 'flair')[0]
    path = source/'flair/shards/rank000/raw_text'/f's0_{key}_y0_x0.json'
    record = io.read_json(path)
    record['response'] = record['attempts'][0]['response'] = 'building'
    io.write_json(path, record)
    output = tmp_path/'with-retry'; active[0] = output
    dispatched = []
    monkeypatch.setattr(recovery, 'verify_runtime_assets', lambda original: None)
    monkeypatch.setattr(guarded, 'launch', lambda new, legacy: dispatched.append((new, legacy)) or 1)
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output), '--run_remaining'])
    assert tool.main() == 1
    assert len(dispatched) == 1 and dispatched[0][1] is tool.queue
    assert dispatched[0][0]['protocol']['format_retries'] == 2
    report = io.read_json(output/'format_repair_report.json')
    assert report['gpu_continuation_invoked'] and report['gpu_continuation_exit_code'] == 1
    assert not report['gpu_workers_started']  # Mock dispatch is not a worker-start receipt.
