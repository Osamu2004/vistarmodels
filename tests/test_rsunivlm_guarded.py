"""WSL-only supervisor regression checks; no pretrained models or CUDA calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_rsunivlm_guarded as guard


def test_missing_or_aliased_gpu_is_rejected_before_workers():
    two = '0, GPU-aaaa, 0000:02:00.0\n1, GPU-bbbb, 0000:04:00.0\n'
    assert [row['uuid'] for row in guard.parse_inventory(two, ['0', '1'])] == ['GPU-aaaa', 'GPU-bbbb']
    with pytest.raises(RuntimeError, match='missing'):
        guard.parse_inventory('0, GPU-aaaa, 0000:02:00.0\n', ['0', '1'])
    with pytest.raises(RuntimeError, match='same physical'):
        guard.parse_inventory(two, ['0', 'GPU-aaaa'])


@pytest.mark.parametrize('error', [RuntimeError('CUDA error: an illegal memory access was encountered'),
    RuntimeError('CUDA error: device-side assert triggered'), RuntimeError('CUDA error: unspecified launch failure')])
def test_cuda_fatal_bypasses_original_exception_handler(tmp_path, error):
    class Engine:
        def forward(self, jobs, prompt):
            yield jobs, ['unchanged']
        def evaluate(self, entry, output):
            raise error
    plan = dict(output_dir=str(tmp_path), identity='original', protocol={})
    guard.install_engine_guard(Engine, plan, 0)
    visited = []
    with pytest.raises(SystemExit) as caught:
        for dataset in ('flair', 'uavid'):
            visited.append(dataset)
            try:
                Engine().evaluate(dict(dataset=dataset), str(tmp_path))
            except Exception:
                pass  # Reproduces the old worker's broad exception handler.
    assert caught.value.code == guard.FATAL_CUDA_EXIT
    assert visited == ['flair']
    receipt = json.loads((tmp_path / 'errors/guard_rank0_flair.json').read_text())
    assert receipt['identity'] == 'original' and receipt['fatal_cuda']


def test_normal_format_failure_and_oom_are_not_cuda_context_failures():
    assert not guard.fatal_cuda_error(ValueError('Invalid RLE item'))
    assert not guard.fatal_cuda_error(RuntimeError('CUDA out of memory'))
    assert not guard.fatal_cuda_error(RuntimeError('GPU memory insufficient even at batch1'))
    assert not guard.fatal_exit(0) and not guard.fatal_exit(1)
    assert guard.fatal_exit(-11) and guard.fatal_exit(70) and guard.fatal_exit(71)


def test_forward_retains_original_job_and_answer_objects(tmp_path):
    answer = object()
    class Engine:
        effective_batch = 16
        def forward(self, jobs, prompt):
            yield jobs, answer
        def evaluate(self, entry, output):
            return {'complete': True}
    plan = dict(output_dir=str(tmp_path), identity='original',
                protocol=dict(model_settings=dict(max_new_tokens=32000)))
    guard.install_engine_guard(Engine, plan, 1)
    jobs = [dict(path=Path('one.json'), identity='one')]
    group, output = next(Engine().forward(jobs, 'fixed prompt'))
    assert group is jobs and output is answer
    assert json.loads((tmp_path / 'guard_inflight_rank001.json').read_text())['requested_batch'] == 16


def test_missing_device_launches_nothing_and_preserves_existing_summary(tmp_path, monkeypatch):
    summary = tmp_path / 'paper_summary.json'
    summary.write_text('historical summary')
    def missing(_):
        raise RuntimeError('Requested GPU 1 is missing')
    monkeypatch.setattr(guard, 'gpu_inventory', missing)
    monkeypatch.setattr(guard.subprocess, 'Popen', lambda *a, **k: pytest.fail('Worker must not start'))
    legacy = SimpleNamespace(merge=lambda _: pytest.fail('Must not merge stale metrics'))
    plan = dict(output_dir=str(tmp_path), identity='original', gpu_ids=['0', '1'])
    assert guard.launch(plan, legacy=legacy) == guard.FATAL_CUDA_EXIT
    assert summary.read_text() == 'historical summary'
    receipt = json.loads((tmp_path / 'guarded_runtime_finished.json').read_text())
    assert not receipt['workers_started'] and receipt['owned_pids'] == []
    assert receipt['complete'] is False


def test_crashed_worker_stops_only_its_owned_peer_and_does_not_merge(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    class Process:
        def __init__(self, pid, code):
            self.pid, self.returncode = pid, code
            self.stdout = iter(())
        def poll(self):
            return self.returncode
    crashed, peer = Process(900001, -11), Process(900002, None)
    instances = iter((crashed, peer))
    monkeypatch.setattr(guard.subprocess, 'Popen', lambda *a, **k: next(instances))
    monkeypatch.setattr(guard, 'gpu_inventory', lambda _: [dict(uuid='GPU-a'), dict(uuid='GPU-b')])
    signals = []
    def signal_owned(pid, sig):
        signals.append((pid, sig))
        assert pid == peer.pid
        peer.returncode = -int(sig)
    monkeypatch.setattr(guard.os, 'killpg', signal_owned)
    legacy = SimpleNamespace(storage=SimpleNamespace(write_json=guard.write_json),
        merge=lambda _: pytest.fail('Fatal process exits must not merge stale shard metrics'))
    plan = dict(output_dir=str(tmp_path), identity='original', gpu_ids=['0', '1'])
    assert guard.launch(plan, legacy=legacy) == guard.FATAL_CUDA_EXIT
    assert signals == [(peer.pid, guard.signal.SIGINT)]
    receipt = json.loads((tmp_path / 'guarded_runtime_finished.json').read_text())
    assert receipt['owned_pids'] == [crashed.pid, peer.pid]
    assert receipt['exit_codes'] == [-11, -int(guard.signal.SIGINT)]
    assert receipt['summary_recomputed'] is False and receipt['complete'] is False
