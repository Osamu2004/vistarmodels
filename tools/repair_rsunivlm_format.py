"""Offline saved-RLE repair and CPU rescoring; GPU continuation is explicit."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm
from baselines.rsunivlm import evaluate as io
from baselines.rsunivlm.protocols import fingerprint
from tools import eval_rsunivlm_paper_queue as queue
from tools import eval_rsunivlm_guarded as guarded
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_format_repair as normalization
from tools import rsunivlm_levir_labels as levir_labels
from tools.rsunivlm_data_loading import LoaderOptions, bind_ordered_reader


def prepare(source, requested, run_remaining=False, levircd_gt_mode='exact_binary'):
    old = io.read_json(source/'plan.json')
    config = io.read_json(source/'run_config.json')
    if not old or not config or old.get('identity') != fingerprint(old.get('protocol')) or config.get('protocol') != old['protocol'] or config.get('identity') != old['identity']:
        raise ValueError('Source plan/config identity is missing or inconsistent')
    if old['protocol'].get('format_repair'):
        raise ValueError('Use the original source_run again to resume this repair, not its derived output')
    finished = io.read_json(source/'runtime_finished.json')
    if not finished or any(code is None for code in finished.get('exit_codes', [None])):
        raise ValueError('Wait for the original queue to exit and write its runtime_finished.json')
    current = io.read_json(source/'runtime.json')
    if current and (current.get('owned_pids') != finished.get('owned_pids') or current.get('launch_id') != finished.get('launch_id')):
        raise ValueError('The latest source invocation has no matching finished receipt')
    if old['protocol'].get('code_sha256') != queue.source_identity():
        raise ValueError('Numerical source changed since the original run')
    available = [entry['dataset'] for entry in old['entries']]
    if not available or len(set(available)) != len(available) or set(available)-set(protocol.DATASETS):
        raise ValueError('Invalid source dataset list')
    gpus = old.get('gpu_ids', [])
    if not gpus or len(set(gpus)) != len(gpus) or any(not isinstance(gpu, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', gpu) for gpu in gpus):
        raise ValueError('Invalid original GPU/shard identifiers')
    if len(requested) != len(set(requested)) or set(requested)-set(available):
        raise ValueError('Choose distinct datasets present in the source run')
    new = copy.deepcopy(old)
    new['entries'] = [entry for entry in new['entries'] if not requested or entry['dataset'] in requested]
    value = new['protocol']
    value['datasets'] = [entry['dataset'] for entry in new['entries']]
    for entry in new['entries']:
        if fingerprint(entry['samples']) != old['protocol']['populations'].get(entry['dataset']):
            raise ValueError('Source dataset manifest changed')
    for field in ('populations', 'prediction_background', 'native_semantic', 'prompts'):
        value[field] = {name: item for name, item in value.get(field, {}).items() if name in value['datasets']}
    if 'second' not in value['datasets']:
        value['second_query'] = None
    if 'levircd' not in value['datasets']:
        value['levircd_query'] = None
        value.pop('levircd_gt_reader', None)
    else:
        value['levircd_gt_reader'] = levir_labels.settings(levircd_gt_mode)
    if run_remaining:
        value['format_retries'] = 2
        for name in value['datasets']:
            value['prompts'][name] = ({key: [recovery.instruction(new, name, key, variant) for variant in range(3)]
                for key in recovery.keys(new, name)} if native.enabled(value.get('semantic_mode'), name) or name == 'second'
                else [protocol.prompt(name, variant) for variant in range(3)])
            if native.enabled(value.get('semantic_mode'), name):
                value['native_semantic'][name] = native.settings(name, 2)
    value['format_repair'] = dict(policy=normalization.POLICY, rules=list(normalization.RULES),
        source_run=str(source), source_identity=old['identity'],
        source_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        importer_sha256=hashlib.sha256((ROOT/'tools/retry_rsunivlm_failed.py').read_bytes()).hexdigest(),
        normalizer_sha256=hashlib.sha256((ROOT/'tools/rsunivlm_format_repair.py').read_bytes()).hexdigest(),
        mode='normalize_then_bounded_model_retry' if run_remaining else 'offline_normalize_and_score_only',
        keep_original_text=True, selection='first repairable attempt only when no original attempt is valid; no GT selection')
    new['identity'] = fingerprint(value)
    return old, new


def query_context(plan, entry, path, record, sample_ids):
    dataset = entry['dataset']
    key = record.get('query_key') if native.enabled(plan['protocol'].get('semantic_mode'), dataset) or dataset == 'second' else record.get('direction')
    if key not in recovery.keys(plan, dataset):
        raise ValueError(f'Unknown query key in {path}')
    spatial = record.get('spatial', {})
    box = spatial.get('source_box', [])
    if len(box) != 4 or not all(isinstance(value, int) for value in box):
        raise ValueError(f'Invalid source box in {path}')
    suffix = f'_{key}_y{box[1]}_x{box[0]}.json'
    if not path.name.endswith(suffix):
        raise ValueError(f'Query filename/spatial mismatch: {path}')
    name = path.name[:-len(suffix)]
    if name not in sample_ids or record.get('identity') != fingerprint([sample_ids[name], key, spatial]):
        raise ValueError(f'Query identity differs: {path}')
    return key


def normalize_cache(plan, output):
    counts = Counter()
    audit_path = output/'format_repair_scan.jsonl'
    temporary = audit_path.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as audit:
        for entry in plan['entries']:
            dataset = entry['dataset']
            for rank in range(len(plan['gpu_ids'])):
                selected = entry['samples'][:plan['max_samples']] if plan['max_samples'] else entry['samples']
                sample_ids = {item['sample']['name']: recovery.sample_id(plan, dataset, item)
                              for item in selected[rank::len(plan['gpu_ids'])]}
                folder = output/dataset/'shards'/f'rank{rank:03d}'/'raw_text'
                for path in tqdm(sorted(folder.glob('*.json')), desc=f'Normalize {dataset}/rank{rank}', unit='query'):
                    if path.is_symlink():
                        raise ValueError(f'Unexpected query symlink: {path}')
                    record = io.read_json(path)
                    if not record:
                        raise ValueError(f'Invalid saved JSON: {path}')
                    key = query_context(plan, entry, path, record, sample_ids)
                    decode = lambda attempt: recovery.decoded(plan, dataset, key, attempt)
                    fixed, result = normalization.repair_record(record, decode)
                    status = result['status']
                    if status == 'already_valid' and record.get('format_repair'):
                        counts['previously_repaired_records'] += 1
                    else:
                        counts[status+'_records'] += 1
                    if status == 'repaired':
                        fixed['format_repair']['record_before_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                        io.write_json(path, fixed)
                    if status != 'already_valid':
                        audit.write(json.dumps(dict(path=str(path), dataset=dataset, rank=rank, **result), ensure_ascii=False)+'\n')
    temporary.replace(audit_path)
    counts['total_repaired_records'] = counts['repaired_records']+counts['previously_repaired_records']
    for field in ('repaired_records', 'previously_repaired_records', 'already_valid_records', 'unrepairable_records'):
        counts.setdefault(field, 0)
    return dict(counts)


class OfflineEngine(worker.Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deferred_queries = {}

    def core(self):
        raise AssertionError('Offline repair must never load a model')

    def forward(self, jobs, prompt):
        for job in jobs:
            self.deferred_queries[str(job['path'])] = ('missing_saved_answer' if not job['attempts'] else 'pending_fixed_template_retry')
        # No answer or generation attempt is fabricated for missing queries.
        if False:
            yield


def score_offline(plan, output, options):
    deferred_total = 0
    for entry in plan['entries']:
        dataset = entry['dataset']
        for rank in range(len(plan['gpu_ids'])):
            engine = OfflineEngine(plan, rank, len(plan['gpu_ids']))
            with bind_ordered_reader(engine, entry, str(output), options):
                if dataset == 'levircd':
                    with levir_labels.bind_targets(plan, rank, output):
                        result = engine.evaluate(entry, str(output))
                else:
                    result = engine.evaluate(entry, str(output))
            deferred_total += len(engine.deferred_queries)
            shard = output/dataset/'shards'/f'rank{rank:03d}'
            for failure in result.get('failed_samples', []):
                reasons = sorted({engine.deferred_queries[path] for path in failure.get('raw_paths', []) if path in engine.deferred_queries})
                if reasons:
                    failure.update(reason='offline_deferred_queries', deferred_reasons=reasons,
                        error='Saved queries require model generation; offline repair did not run the model')
                    io.write_json(shard/'errors'/f'{failure["sample"]}.json', failure)
            result.update(offline_only=True, model_forward_calls_this_process=0,
                          deferred_queries=engine.deferred_queries)
            if engine.deferred_queries:
                result['exhausted_population'] = False
            io.write_json(shard/'metrics.json', result)
            io.write_json(shard/'failed_samples.json', dict(num_failed=result.get('num_failed', 0), failed_samples=result.get('failed_samples', [])))
            status = io.read_json(shard/'status.json') or {}
            status.update(offline_only=True, exhausted_population=result['exhausted_population'],
                          deferred_queries=len(engine.deferred_queries))
            io.write_json(shard/'status.json', status)
    summary = queue.merge(plan)
    summary.update(offline_only=True, model_forward_calls_this_process=0, deferred_queries=deferred_total,
                   format_repair_policy=normalization.POLICY)
    io.write_json(output/'paper_summary.json', summary)
    io.write_json(output/'offline_runtime_finished.json', dict(identity=plan['identity'], owned_pids=[], exit_codes=[],
        mode='offline_format_repair', complete=summary['complete'], model_forward_calls=0))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', required=True)
    parser.add_argument('--datasets', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--plan_only', action='store_true')
    parser.add_argument('--run_remaining', action='store_true', help='Explicitly enable guarded GPU inference with at most two existing fixed retry templates')
    parser.add_argument('--levircd_gt_mode', choices=levir_labels.GT_MODES, default='exact_binary',
                        help='Explicit LEVIR GT reader; vistar_positive matches the existing Vistar grayscale >0 evaluation rule')
    args = parser.parse_args()
    source = Path(args.source_run).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    requested = [name.strip() for name in args.datasets.split(',')] if args.datasets else []
    with (source/'.queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('The original queue is still running; wait for it to exit')
        old, new = prepare(source, requested, args.run_remaining, levircd_gt_mode=args.levircd_gt_mode)
        suffix = '_formatfix_v1_retry2' if args.run_remaining else '_formatfix_v1'
        requested_output = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.with_name(source.name+suffix)
        if requested_output == source or source in requested_output.parents or requested_output in source.parents:
            raise ValueError('Repair output must be separate from, and not contain, the original run')
        output, resolution = queue.resolve_output_root(requested_output, new['protocol'])
        new['output_dir'] = str(output)
        print(f'[formatfix] Actual OUTPUT_DIR={output}; datasets={new["protocol"]["datasets"]}; GPU continuation={args.run_remaining}', flush=True)
        if args.plan_only:
            print('Plan only; no answers imported, no repair/score or model executed.', flush=True)
            return 0
        output.mkdir(parents=True, exist_ok=True)
        with (output/'.queue.lock').open('a') as target_lock:
            fcntl.flock(target_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            io.write_json(output/'run_config.json', dict(identity=new['identity'], protocol=new['protocol']))
            io.write_json(output/'plan.json', new)
            if resolution:
                io.write_json(output/'output_resolution.json', resolution)
            imported = recovery.import_predictions(source, output, old, new)
            repaired = normalize_cache(new, output)
            print(f'[formatfix] Raw-text normalization: {repaired}; CPU reconstruction/scoring begins', flush=True)
            summary = score_offline(new, output, LoaderOptions.from_env())
            report = dict(source_run=str(source),
                source_identity=old['identity'], identity=new['identity'], imported=imported,
                levircd_gt_reader=new['protocol'].get('levircd_gt_reader'),
                normalization=repaired, offline_complete=summary['complete'],
                deferred_queries=summary['deferred_queries'], offline_model_forward_calls=0,
                gpu_continuation_requested=args.run_remaining, gpu_continuation_invoked=False,
                gpu_workers_started=False,
                final_complete=summary['complete'])
            io.write_json(output/'format_repair_report.json', report)
            print(f'[formatfix] CPU phase complete; selected population complete={summary["complete"]}; deferred queries={summary["deferred_queries"]}', flush=True)
            if args.run_remaining and not summary['complete']:
                recovery.verify_runtime_assets(old)
                (output/'logs').mkdir(exist_ok=True)
                report['gpu_continuation_invoked'] = True
                io.write_json(output/'format_repair_report.json', report)
                code = guarded.launch(new, legacy=queue)
                report['gpu_continuation_exit_code'] = code
                runtime = io.read_json(output/'guarded_runtime_finished.json') or {}
                report['gpu_workers_started'] = bool(runtime.get('workers_started'))
                latest = io.read_json(output/'paper_summary.json') or summary
                report['final_complete'] = bool(latest.get('complete'))
                io.write_json(output/'format_repair_report.json', report)
                return code
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
