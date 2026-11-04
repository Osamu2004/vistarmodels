"""Verify and reuse a finished derived run without discarding its real retries."""
from __future__ import annotations

import hashlib
from pathlib import Path

from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from tools import rsunivlm_format_repair as normalization


def prepare_reuse(original_root, original, target, reuse_root):
    root = Path(reuse_root).expanduser().resolve()
    if root == Path(original_root).resolve():
        raise ValueError('reuse_run must be the derived recovery run, not the original source')
    plan, config = io.read_json(root/'plan.json'), io.read_json(root/'run_config.json')
    if not plan or not config or plan.get('identity') != fingerprint(plan.get('protocol')) or config.get('identity') != plan['identity'] or config.get('protocol') != plan['protocol']:
        raise ValueError('Reuse plan/config identity is inconsistent')
    old, aux = original['protocol'], plan['protocol']
    provenance = aux.get('format_repair', {})
    if provenance.get('policy') != normalization.POLICY or provenance.get('source_identity') != original['identity'] or Path(provenance.get('source_run', '')).resolve() != Path(original_root).resolve():
        raise ValueError('Reuse run does not derive from the same original source and repair policy')
    if plan.get('gpu_ids') != original.get('gpu_ids') or plan.get('max_samples') != original.get('max_samples'):
        raise ValueError('Reuse sharding/sample limit differs from the original run')
    for key in ('schema', 'semantic_mode', 'world_size', 'batch_size', 'tile_size', 'strict_population',
                'format_failure_policy', 'source_revision', 'upstream_code_sha256', 'checkpoint',
                'vision_files', 'model_settings', 'code_sha256'):
        if aux.get(key) != old.get(key):
            raise ValueError(f'Reuse numerical protocol differs: {key}')
    if aux.get('format_retries') not in (0, 1, 2):
        raise ValueError('Unknown reuse retry budget')
    if aux['format_retries'] > target['protocol']['format_retries']:
        raise ValueError('Reuse has already consumed a larger retry budget; retain --run_remaining instead of resetting it')
    if aux.get('constrained_rle'):
        required = target['protocol'].get('constrained_rle')
        if not required or any(aux['constrained_rle'].get(key) != required.get(key) for key in ('policy', 'rows', 'cols')):
            raise ValueError('Reuse contains constrained outputs; explicitly retain the same RLE fallback/grid')
    finished = io.read_json(root/'runtime_finished.json')
    current = io.read_json(root/'runtime.json')
    offline = io.read_json(root/'offline_runtime_finished.json')
    if finished:
        if any(code is None for code in finished.get('exit_codes', [None])) or (current and any(current.get(key) != finished.get(key) for key in ('owned_pids', 'launch_id'))):
            raise ValueError('Reuse invocation has not finished cleanly enough to take a snapshot')
    elif current or not offline or offline.get('identity') != plan['identity'] or offline.get('mode') != 'offline_format_repair':
        raise ValueError('Wait for reuse_run to finish before importing its results')
    available = {entry['dataset']:entry for entry in plan['entries']}
    selected = [entry['dataset'] for entry in target['entries'] if entry['dataset'] in available]
    if not selected:
        raise ValueError('Reuse run contains none of the selected datasets')
    for name in selected:
        if fingerprint(available[name]['samples']) != old['populations'].get(name) or aux.get('populations', {}).get(name) != old['populations'].get(name):
            raise ValueError(f'Reuse population differs: {name}')
        if aux.get('prediction_background', {}).get(name) != old.get('prediction_background', {}).get(name):
            raise ValueError(f'Reuse label interpretation differs: {name}')
    record = dict(path=str(root), identity=plan['identity'], datasets=selected,
        source_plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='import finished derived masks/answers first; preserve all actual attempts; original source fills gaps')
    return root, plan, record
