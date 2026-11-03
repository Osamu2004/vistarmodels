"""Fail-stop supervision around the unchanged, fingerprinted paper evaluator.

No model operations are changed. Existing masks/raw answers retain their original
identity. This controller adds GPU enumeration, crash logs and process shutdown;
it does not reset a GPU or retry a failed CUDA context.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FATAL_CUDA_EXIT = 70
STARTUP_EXIT = 71


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fatal_cuda_error(error):
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).casefold()
    # Ordinary allocator OOM is handled by the existing batch-splitting path.
    if 'out of memory' in message or 'memory insufficient' in message:
        return False
    return any(marker in message for marker in (
        'illegal memory access', 'device-side assert', 'unspecified launch failure',
        'launch timed out', 'uncorrectable ecc', 'context is destroyed',
        'cuda error: unknown error', 'cuda driver error', 'gpu has fallen off',
    ))


def parse_inventory(text, selected):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if len(fields) != 3:
            raise RuntimeError(f'Unexpected nvidia-smi inventory row: {fields!r}')
        index, uuid, pci_bus = (field.strip() for field in fields)
        if not index.isdigit() or not uuid.startswith('GPU-') or not pci_bus:
            raise RuntimeError(f'Invalid GPU inventory row: {fields!r}')
        rows.append(dict(index=index, uuid=uuid, pci_bus_id=pci_bus))
    if len({row['index'] for row in rows}) != len(rows) or len({row['uuid'] for row in rows}) != len(rows):
        raise RuntimeError('GPU inventory has duplicate device identities')
    chosen = []
    for gpu in selected:
        matches = [row for row in rows if row['index'] == gpu or
                   (gpu.startswith('GPU-') and row['uuid'].startswith(gpu))]
        if len(matches) != 1:
            raise RuntimeError(f'Requested GPU {gpu!r} is missing or ambiguous; visible GPUs: {rows}')
        chosen.append(matches[0])
    if len({row['uuid'] for row in chosen}) != len(chosen):
        raise RuntimeError('Two requested GPU identifiers resolve to the same physical device')
    return chosen


def gpu_inventory(selected):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return parse_inventory(result.stdout, selected)


def install_engine_guard(engine_class, plan, rank):
    """Pass through numerical calls; only add CPU metadata and fatal exit."""
    original_evaluate = engine_class.evaluate
    original_forward = engine_class.forward
    root = Path(plan['output_dir'])

    def forward(self, jobs, prompt):
        if jobs:
            write_json(root / f'guard_inflight_rank{rank:03d}.json', dict(
                identity=plan['identity'], rank=rank, pid=os.getpid(),
                time_utc=datetime.now(timezone.utc).isoformat(), state='dispatch_started',
                requested_batch=self.effective_batch, queued_queries=len(jobs),
                max_new_tokens=plan['protocol']['model_settings']['max_new_tokens'],
                prompt=prompt,
                queries=[dict(path=str(job['path']), identity=job['identity']) for job in jobs],
                note='Full pending dispatch; OOM splits and returned groups remain in the original implementation',
            ))
        for group, answers in original_forward(self, jobs, prompt):
            yield group, answers

    def evaluate(self, entry, output):
        try:
            return original_evaluate(self, entry, output)
        except Exception as error:
            if not fatal_cuda_error(error):
                raise
            write_json(root / 'errors' / f'guard_rank{rank}_{entry["dataset"]}.json', dict(
                identity=plan['identity'], rank=rank, pid=os.getpid(), dataset=entry['dataset'],
                error=str(error), traceback=traceback.format_exc(), fatal_cuda=True,
                action='Exit worker; do not reuse the CUDA context for another dataset',
            ))
            print(f'[guard {rank}] Fatal CUDA failure; stopping worker without another dataset: {error}', flush=True)
            # SystemExit bypasses the original worker's broad except Exception.
            raise SystemExit(FATAL_CUDA_EXIT) from error

    engine_class.forward = forward
    engine_class.evaluate = evaluate


def run_worker(plan_path, rank):
    print(f'[guard {rank}] Worker bootstrap pid={os.getpid()}; importing original evaluator', flush=True)
    plan = json.loads(Path(plan_path).read_text())
    from tools import eval_rsunivlm_paper_queue as legacy
    from tools.rsunivlm_paper_worker import Engine
    install_engine_guard(Engine, plan, rank)
    print(f'[guard {rank}] Original identity/CUDA initialization checks begin', flush=True)
    return legacy.worker(plan_path, rank)


def stop_owned(processes):
    """Signal only process groups created by this invocation, with bounded waits."""
    for sig, seconds in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        alive = [process for process in processes if process.poll() is None]
        if not alive:
            return
        for process in alive:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and any(process.poll() is None for process in alive):
            time.sleep(0.1)


def fatal_exit(returncode):
    # 1 is the original evaluator's normal partial/format-failure completion.
    return returncode is not None and returncode not in (0, 1)


def launch(plan, *, legacy):
    root = Path(plan['output_dir'])
    launch_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{os.getpid()}'
    report = dict(identity=plan['identity'], launch_id=launch_id, complete=False,
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_ids=plan['gpu_ids'], owned_pids=[], exit_codes=[], workers_started=False)
    processes, threads, pump_errors, failure = [], [], [], None

    def pump(process, gpu):
        path = root / 'logs' / f'guard_{launch_id}_gpu_{gpu}.log'
        try:
            with path.open('w', encoding='utf-8') as stream:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(f'[GPU {gpu}] {line}', end='', flush=True)
        except Exception:
            pump_errors.append(traceback.format_exc())

    try:
        report['selected_devices'] = gpu_inventory(plan['gpu_ids'])
        print(f'[guard] Selected physical GPUs: {report["selected_devices"]}', flush=True)
        for rank, gpu in enumerate(plan['gpu_ids']):
            env = os.environ.copy()
            for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
                env.pop(name, None)
            env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
            process = subprocess.Popen(
                [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()),
                 '--worker_plan', str(root / 'plan.json'), '--rank', str(rank)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace', bufsize=1, start_new_session=True,
            )
            processes.append(process)
            report['owned_pids'].append(process.pid)
            report['workers_started'] = True
            write_json(root / 'guarded_runtime.json', report)
            thread = threading.Thread(target=pump, args=(process, gpu), daemon=True)
            thread.start()
            threads.append(thread)
        # Keep the ordinary recovery tool aware of this invocation's processes.
        legacy.storage.write_json(root / 'runtime.json', dict(
            owned_pids=report['owned_pids'], gpu_ids=plan['gpu_ids'], launch_id=launch_id))
        while True:
            codes = [process.poll() for process in processes]
            if any(fatal_exit(code) for code in codes):
                failure = f'Worker crashed or reported fatal CUDA failure: {codes}; stopping peer workers'
                print(f'[guard] {failure}', flush=True)
                stop_owned(processes)
                break
            if pump_errors:
                raise RuntimeError('Worker log capture failed: ' + pump_errors[0])
            if all(code is not None for code in codes):
                break
            time.sleep(0.2)
        for thread in threads:
            thread.join(timeout=1)
        if pump_errors:
            raise RuntimeError('Worker log capture failed: ' + pump_errors[0])
        if failure is None:
            summary = legacy.merge(plan)
            report['summary_recomputed'] = True
            report['complete'] = summary['complete'] and all(process.returncode == 0 for process in processes)
        else:
            report['summary_recomputed'] = False
            print('[guard] Existing paper_summary retained; this invocation is incomplete.', flush=True)
    except KeyboardInterrupt:
        failure = 'Interrupted by user'
        report['complete'] = False
        stop_owned(processes)
    except Exception:
        failure = traceback.format_exc()
        report['complete'] = False
        stop_owned(processes)
        print(f'[guard] Launch stopped: {failure}', flush=True)
    finally:
        # A dead driver can leave a child uninterruptible even after SIGKILL.
        # Do not hang the controller forever waiting for a log pipe to close.
        for thread in threads:
            thread.join(timeout=1)
        report.update(exit_codes=[process.poll() for process in processes], failure=failure,
                      surviving_owned_pids=[process.pid for process in processes if process.poll() is None],
                      time_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / 'guarded_runtime_finished.json', report)
        if processes:
            legacy.storage.write_json(root / 'runtime_finished.json', dict(
                owned_pids=report['owned_pids'], exit_codes=report['exit_codes'],
                launch_id=launch_id, complete=report['complete'], supervisor_failure=failure))
    print(f'[guard] Report={root / "guarded_runtime_finished.json"}; complete={report["complete"]}', flush=True)
    return 0 if report['complete'] else FATAL_CUDA_EXIT if failure else 1


def main():
    if '--worker_plan' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker_plan', required=True)
        parser.add_argument('--rank', type=int, required=True)
        args = parser.parse_args()
        try:
            return run_worker(args.worker_plan, args.rank)
        except Exception:
            traceback.print_exc()
            return STARTUP_EXIT
    from tools import eval_rsunivlm_paper_queue as legacy
    legacy.launch = functools.partial(launch, legacy=legacy)
    return legacy.main()


if __name__ == '__main__':
    raise SystemExit(main())
