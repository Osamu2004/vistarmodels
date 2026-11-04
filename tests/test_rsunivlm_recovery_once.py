"""WSL-only one-invocation recovery checks; no real model/CUDA is run."""
import copy
import fcntl
from pathlib import Path
import sys

import pytest

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from tools import repair_rsunivlm_format as tool
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_recovery_runtime as online
from tools import rsunivlm_recovery_reuse as reuse
from test_rsunivlm_format_repair_pipeline import setup


@pytest.mark.parametrize('dataset, key_field, key, label', [
    ('flair', 'key', 'class_01', 'road'), ('second', 'direction', 'water', 'water')])
def test_new_answers_are_repaired_before_original_decoder_consumes_another_attempt(tmp_path, dataset, key_field, key, label):
    plan = dict(identity='synthetic', output_dir=str(tmp_path), protocol={
        'semantic_mode':'native_classwise', 'runtime_format_repair':online.settings()})
    answers = [dict(response=text, generation=dict(ended_with_eos=True, hit_token_limit=False))
               for text in (label+'**2, others:22\nothers*24', 'others*24', 'coni')]
    original = copy.deepcopy(answers)
    jobs = [dict(path=tmp_path/f'q{index}.json', **{key_field:key}) for index in range(3)]
    class Engine:
        def forward(self, group, prompt):
            yield group, answers
        def evaluate(self, entry, output):
            return list(self.forward(jobs, '[SEG] original unchanged prompt'))
    online.install(Engine, plan, 0)
    result = Engine().evaluate(dict(dataset=dataset), str(tmp_path))[0][1]
    assert result[0]['response'] == label+'*2, others*22\nothers*24'
    assert result[0]['format_repair']['original_response'] == original[0]['response']
    assert result[0]['generation'] == original[0]['generation']
    assert result[1:] == original[1:] and answers == original
    counts = io.read_json(tmp_path/'runtime_format_repair_rank000.json')['datasets'][dataset]
    assert counts == dict(checked=3, already_valid=1, repaired=1, unrepairable=1)


def mock_receipt(output, plan, code):
    current = dict(identity=plan['identity'], launch_id='synthetic', owned_pids=[123, 124])
    io.write_json(output/'guarded_runtime.json', current)
    io.write_json(output/'guarded_runtime_finished.json', dict(current, workers_started=True,
        exit_codes=[code, code], surviving_owned_pids=[], failure=None))


def bad_original(source, plan):
    key = recovery.keys(plan, 'flair')[0]
    path = source/'flair/shards/rank000/raw_text'/f's0_{key}_y0_x0.json'
    record = io.read_json(path)
    record['response'] = record['attempts'][0]['response'] = 'building'
    io.write_json(path, record)
    return key, path


def test_gpu_exit_is_followed_by_repair_and_selective_cpu_rescoring(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    key, original_raw = bad_original(source, plan)
    output = tmp_path/'one_call'; active[0] = output
    monkeypatch.setattr(recovery, 'verify_runtime_assets', lambda value: None)
    def launch(new, legacy):
        path = output/'flair/shards/rank000/raw_text'/original_raw.name
        record = io.read_json(path)
        attempt = dict(prompt_variant=1, prompt=recovery.instruction(new, 'flair', key, 1),
                       response='building**1', generation=dict(ended_with_eos=True, hit_token_limit=False))
        record['attempts'].append(attempt); record.update(attempt)
        io.write_json(path, record)
        mock_receipt(output, new, 1)
        return 1
    monkeypatch.setattr(tool.guarded, 'launch', launch)
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output), '--run_remaining'])
    assert tool.main() == 0
    report = io.read_json(output/'format_repair_report.json')
    assert report['finalization_complete'] and report['final_complete']
    assert report['post_gpu_normalization']['repaired_records'] == 1
    assert report['post_gpu_rescored_datasets'] == ['flair']
    assert report['recovery_counts']['flair']['recovered_original_failed'] == 1
    assert io.read_json(output/'paper_summary.json')['complete']
    assert io.read_json(output/'guarded_runtime_finished.json')['exit_codes'] == [1, 1]
    assert io.read_json(original_raw)['response'] == 'building'


def test_fatal_or_unsealed_gpu_exit_does_not_become_success_after_cpu_work(tmp_path, monkeypatch):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    bad_original(source, plan)
    output = tmp_path/'fatal'; active[0] = output
    monkeypatch.setattr(recovery, 'verify_runtime_assets', lambda value: None)
    def launch(new, legacy):
        mock_receipt(output, new, -11)
        return 70
    monkeypatch.setattr(tool.guarded, 'launch', launch)
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--output_dir', str(output), '--run_remaining'])
    assert tool.main() == 70
    report = io.read_json(output/'format_repair_report.json')
    assert not report['final_complete'] and 'finalization_skipped' in report
    assert 'post_gpu_normalization' not in report


@pytest.mark.parametrize('valid', [True, False])
def test_import_previous_recovery_first_preserves_successes_and_used_attempts(tmp_path, monkeypatch, valid):
    source, plan, entry, active = setup(tmp_path, monkeypatch)
    key, original_raw = bad_original(source, plan)
    old, aux = tool.prepare(source, [], True)
    previous = tmp_path/'previous'; aux['output_dir'] = str(previous); active[0] = previous
    io.write_json(previous/'plan.json', aux)
    io.write_json(previous/'run_config.json', dict(identity=aux['identity'], protocol=aux['protocol']))
    recovery.import_predictions(source, previous, old, aux)
    path = previous/'flair/shards/rank000/raw_text'/original_raw.name
    record = io.read_json(path)
    for variant in (1, 2):
        attempt = dict(prompt_variant=variant, prompt=recovery.instruction(aux, 'flair', key, variant),
            response='building*1' if valid and variant == 1 else 'building',
            generation=dict(ended_with_eos=True, hit_token_limit=False))
        record['attempts'].append(attempt); record.update(attempt)
        if valid: break
    io.write_json(path, record)
    tool.score_offline(aux, previous, tool.LoaderOptions(num_workers=0))
    (previous/'.queue.lock').touch()
    original_bytes = {str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()}
    output = tmp_path/'new'; active[0] = output
    launched = []
    monkeypatch.setattr(recovery, 'verify_runtime_assets', lambda value: None)
    monkeypatch.setattr(tool.guarded, 'launch', lambda new, legacy: launched.append(new) or 1)
    monkeypatch.setattr(sys, 'argv', ['repair', '--source_run', str(source), '--reuse_run', str(previous),
                                  '--output_dir', str(output), '--run_remaining'])
    assert tool.main() == (0 if valid else 1)
    assert not launched  # Complete or exhausted: neither case needs another GPU scan.
    if valid:
        assert io.read_json(output/'format_repair_report.json')['reuse_import']['reused_complete_samples'] == 4
    else:
        imported = io.read_json(output/'flair/shards/rank000/raw_text'/original_raw.name)
        assert len(imported['attempts']) == 3
        assert [item['prompt_variant'] for item in imported['attempts']] == [0, 1, 2]
    assert {str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()} == original_bytes
    with (previous/'.queue.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match='still running'):
            tool.main()
    io.write_json(previous/'runtime.json', dict(owned_pids=[123, 124], launch_id='stopped'))
    io.write_json(previous/'runtime_finished.json', dict(owned_pids=[123, 124], launch_id='stopped', exit_codes=[-2, -2]))
    assert reuse.prepare_reuse(source, old, aux, previous)[1]['identity'] == aux['identity']
    changed = copy.deepcopy(aux)
    changed['protocol']['batch_size'] = 32
    changed['identity'] = fingerprint(changed['protocol'])
    io.write_json(previous/'plan.json', changed)
    io.write_json(previous/'run_config.json', dict(identity=changed['identity'], protocol=changed['protocol']))
    with pytest.raises(ValueError, match='numerical protocol differs'):
        reuse.prepare_reuse(source, old, aux, previous)
