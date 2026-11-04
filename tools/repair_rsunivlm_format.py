"""Offline saved-RLE repair and CPU rescoring; GPU continuation is explicit."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
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
from tools import rsunivlm_recovery_runtime as online_repair
from tools import rsunivlm_recovery_reuse as reuse
from tools import rsunivlm_constrained_rle as constrained
from tools.rsunivlm_data_loading import LoaderOptions, bind_ordered_reader


def prepare(source, requested, run_remaining=False, levircd_gt_mode='exact_binary', constrained_rle=False, rle_rows=24, rle_cols=24):
    if constrained_rle and not run_remaining:
        raise ValueError('--constrained_rle requires --run_remaining')
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
        value['runtime_format_repair'] = online_repair.settings()
        value['format_retries'] = 2
        for name in value['datasets']:
            value['prompts'][name] = ({key: [recovery.instruction(new, name, key, variant) for variant in range(3)]
                for key in recovery.keys(new, name)} if native.enabled(value.get('semantic_mode'), name) or name == 'second'
                else [protocol.prompt(name, variant) for variant in range(3)])
            if native.enabled(value.get('semantic_mode'), name):
                value['native_semantic'][name] = native.settings(name, 2)
    if constrained_rle:
        if value.get('semantic_mode') != native.MODE and any(name not in ('levircd', 'second') for name in value['datasets']):
            raise ValueError('Constrained semantic fallback requires native_classwise queries')
        value['constrained_rle'] = constrained.settings(rle_rows, rle_cols)
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


def normalize_cache(plan, output, audit_name='format_repair_scan.jsonl'):
    counts = Counter()
    repaired_datasets = set()
    audit_path = output/audit_name
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
                        repaired_datasets.add(dataset)
                        fixed['format_repair']['record_before_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                        io.write_json(path, fixed)
                    if status != 'already_valid':
                        audit.write(json.dumps(dict(path=str(path), dataset=dataset, rank=rank, **result), ensure_ascii=False)+'\n')
                    if (plan['protocol'].get('constrained_rle') and status == 'unrepairable' and
                            len(record.get('attempts', [])) == plan['protocol']['format_retries']+1 and
                            not any(attempt.get('generation', {}).get('rle_constraints') for attempt in record.get('attempts', []))):
                        counts['eligible_constrained_legacy_queries'] += 1
    temporary.replace(audit_path)
    counts['total_repaired_records'] = counts['repaired_records']+counts['previously_repaired_records']
    for field in ('repaired_records', 'previously_repaired_records', 'already_valid_records', 'unrepairable_records', 'eligible_constrained_legacy_queries'):
        counts.setdefault(field, 0)
    return dict(counts, repaired_datasets=sorted(repaired_datasets))


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


def score_offline(plan, output, options, datasets=None):
    deferred_total = 0
    for entry in plan['entries']:
        dataset = entry['dataset']
        if datasets is not None and dataset not in datasets:
            continue
        for rank in range(len(plan['gpu_ids'])):
            engine = OfflineEngine(plan, rank, len(plan['gpu_ids']))
            with constrained.bind_decoders(plan), bind_ordered_reader(engine, entry, str(output), options):
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
            constrained.annotate_scored_samples(plan, dataset, rank, output, result)
            if engine.deferred_queries:
                result['exhausted_population'] = False
            io.write_json(shard/'metrics.json', result)
            io.write_json(shard/'failed_samples.json', dict(num_failed=result.get('num_failed', 0), failed_samples=result.get('failed_samples', [])))
            status = io.read_json(shard/'status.json') or {}
            status.update(offline_only=True, exhausted_population=result['exhausted_population'],
                          deferred_queries=len(engine.deferred_queries))
            io.write_json(shard/'status.json', status)
    summary = queue.merge(plan)
    summary.update(offline_only=datasets is None, model_forward_calls_this_process=0, deferred_queries=deferred_total,
                   scoring_phase='initial_offline_cpu' if datasets is None else 'post_gpu_cpu_rescore',
                   format_repair_policy=normalization.POLICY)
    if plan['protocol'].get('constrained_rle'):
        summary.update(constrained_rle_policy=plan['protocol']['constrained_rle'], official_original_decoding_protocol=False)
    io.write_json(output/'paper_summary.json', summary)
    io.write_json(output/'offline_runtime_finished.json', dict(identity=plan['identity'], owned_pids=[], exit_codes=[],
        mode='offline_format_repair', complete=summary['complete'], model_forward_calls=0))
    return summary


def recovery_counts(source, output, plan):
    result = {}
    for entry in plan['entries']:
        name = entry['dataset']
        failed, old_ok, now_ok = set(), set(), set()
        for root, old in ((Path(source), True), (Path(output), False)):
            for path in (root/name/'shards').glob('rank*/metrics.json'):
                value = io.read_json(path) or {}
                names = {row['sample'] for row in value.get('rows', [])}
                if old:
                    old_ok.update(names)
                    failed.update(row['sample'] for row in value.get('failed_samples', []))
                else:
                    now_ok.update(names)
        result[name] = dict(original_failed=len(failed), recovered_original_failed=len(failed & now_ok),
            original_failed_not_yet_recovered=len(failed-now_ok),
            newly_scored_prior_unaccounted=len(now_ok-failed-old_ok), successful_images=len(now_ok))
    return result


def constrained_sample_counts(plan, output):
    if not plan['protocol'].get('constrained_rle'):
        return {}
    return {entry['dataset']:sum(
        (io.read_json(output/entry['dataset']/'shards'/f'rank{rank:03d}'/'constrained_samples.json') or {}).get('successfully_scored_constrained_samples', 0)
        for rank in range(len(plan['gpu_ids']))) for entry in plan['entries']}


def finalize_gpu(plan, output, source, report, code, options):
    runtime = io.read_json(output/'guarded_runtime_finished.json') or {}
    current = io.read_json(output/'guarded_runtime.json') or {}
    report['gpu_continuation_exit_code'] = code
    report['gpu_workers_started'] = bool(runtime.get('workers_started'))
    codes = runtime.get('exit_codes', [])
    clean = (code in (0, 1) and runtime.get('identity') == plan['identity'] and
             runtime.get('launch_id') == current.get('launch_id') and runtime.get('launch_id') is not None and
             runtime.get('owned_pids') == current.get('owned_pids') and
             len(codes) == len(plan['gpu_ids']) and all(value in (0, 1) for value in codes) and
             not runtime.get('failure') and not runtime.get('surviving_owned_pids'))
    if not clean:
        report.update(final_complete=False, finalization_skipped='GPU exit/identity receipt is missing, interrupted or fatal')
        io.write_json(output/'format_repair_report.json', report)
        return code if code else 1
    repaired = normalize_cache(plan, output, audit_name='format_repair_scan_post_gpu.jsonl')
    report['post_gpu_normalization'] = repaired
    if repaired['repaired_records']:
        summary = score_offline(plan, output, options, datasets=set(repaired['repaired_datasets']))
        report['post_gpu_rescored_datasets'] = repaired['repaired_datasets']
    else:
        summary = io.read_json(output/'paper_summary.json')
        if not summary:
            raise ValueError('GPU workers exited without a scoring summary')
        report['post_gpu_rescored_datasets'] = []
    report['recovery_counts'] = recovery_counts(source, output, plan)
    report['runtime_format_repair'] = [io.read_json(output/f'runtime_format_repair_rank{rank:03d}.json')
                                     for rank in range(len(plan['gpu_ids']))]
    if plan['protocol'].get('constrained_rle'):
        report['constrained_rle_workers'] = [io.read_json(output/f'constrained_rle_rank{rank:03d}.json')
                                            for rank in range(len(plan['gpu_ids']))]
        summary.update(constrained_rle_policy=plan['protocol']['constrained_rle'], official_original_decoding_protocol=False)
        io.write_json(output/'paper_summary.json', summary)
        report['constrained_samples'] = constrained_sample_counts(plan, output)
    report.update(finalization_complete=True, final_complete=bool(summary['complete']))
    io.write_json(output/'format_repair_report.json', report)
    for name, counts in report['recovery_counts'].items():
        print(f'[recovery result] {name}: {counts}', flush=True)
    print(f'[recovery result] Complete={report["final_complete"]}; report={output/"format_repair_report.json"}', flush=True)
    return 0 if report['final_complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', required=True)
    parser.add_argument('--datasets', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--reuse_run', default='', help='Import a finished previous derived run before the original source, preserving successful work and used retries')
    parser.add_argument('--plan_only', action='store_true')
    parser.add_argument('--run_remaining', action='store_true', help='Explicitly enable guarded GPU inference with at most two existing fixed retry templates')
    parser.add_argument('--levircd_gt_mode', choices=levir_labels.GT_MODES, default='exact_binary',
                        help='Explicit LEVIR GT reader; vistar_positive matches the existing Vistar grayscale >0 evaluation rule')
    parser.add_argument('--constrained_rle', action='store_true', help='Opt-in grammar-constrained last retry and one fallback for exhausted legacy queries; a separately recorded decoding protocol')
    parser.add_argument('--rle_rows', type=int, default=24)
    parser.add_argument('--rle_cols', type=int, default=24)
    args = parser.parse_args()
    source = Path(args.source_run).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    requested = [name.strip() for name in args.datasets.split(',')] if args.datasets else []
    with ExitStack() as stack:
        lock = stack.enter_context((source/'.queue.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('The original queue is still running; wait for it to exit')
        old, new = prepare(source, requested, args.run_remaining, levircd_gt_mode=args.levircd_gt_mode,
                           constrained_rle=args.constrained_rle, rle_rows=args.rle_rows, rle_cols=args.rle_cols)
        reused = None
        if args.reuse_run:
            reused_root = Path(args.reuse_run).expanduser().resolve()
            if not reused_root.is_dir():
                raise FileNotFoundError(reused_root)
            reused_lock = stack.enter_context((reused_root/'.queue.lock').open('a'))
            try:
                fcntl.flock(reused_lock, fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('The previous recovery queue is still running; wait for it to exit before reuse')
            reused_root, reused_plan, reused_record = reuse.prepare_reuse(source, old, new, reused_root)
            new['protocol']['format_repair']['reuse_run'] = reused_record
            new['identity'] = fingerprint(new['protocol'])
            reused = (reused_root, reused_plan, reused_record)
        suffix = '_formatfix_rle_v1' if args.constrained_rle else '_formatfix_once_v2' if reused else '_formatfix_v1_retry2' if args.run_remaining else '_formatfix_v1'
        requested_output = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.with_name(source.name+suffix)
        if requested_output == source or source in requested_output.parents or requested_output in source.parents:
            raise ValueError('Repair output must be separate from, and not contain, the original run')
        if reused and (requested_output == reused[0] or reused[0] in requested_output.parents or requested_output in reused[0].parents):
            raise ValueError('New output must be separate from the reused recovery run')
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
            reuse_import = None
            if reused:
                reuse_target = copy.deepcopy(new)
                reuse_target['entries'] = [entry for entry in new['entries'] if entry['dataset'] in reused[2]['datasets']]
                reuse_import = recovery.import_predictions(reused[0], output, reused[1], reuse_target)
            imported = recovery.import_predictions(source, output, old, new)
            repaired = normalize_cache(new, output)
            print(f'[formatfix] Raw-text normalization: {repaired}; CPU reconstruction/scoring begins', flush=True)
            summary = score_offline(new, output, LoaderOptions.from_env())
            report = dict(source_run=str(source),
                source_identity=old['identity'], identity=new['identity'], imported=imported,
                reused_derived_run=str(reused[0]) if reused else None, reuse_import=reuse_import,
                levircd_gt_reader=new['protocol'].get('levircd_gt_reader'),
                constrained_rle=new['protocol'].get('constrained_rle'),
                constrained_samples=constrained_sample_counts(new, output),
                normalization=repaired, offline_complete=summary['complete'],
                deferred_queries=summary['deferred_queries'], offline_model_forward_calls=0,
                gpu_continuation_requested=args.run_remaining, gpu_continuation_invoked=False,
                gpu_workers_started=False,
                final_complete=summary['complete'], recovery_counts=recovery_counts(source, output, new))
            io.write_json(output/'format_repair_report.json', report)
            print(f'[formatfix] CPU phase complete; selected population complete={summary["complete"]}; deferred queries={summary["deferred_queries"]}', flush=True)
            if args.run_remaining and not summary['complete']:
                if not summary['deferred_queries'] and not repaired['eligible_constrained_legacy_queries']:
                    report.update(finalization_complete=True,
                        gpu_continuation_skipped='All remaining invalid histories exhausted the existing fixed retry budget')
                    io.write_json(output/'format_repair_report.json', report)
                    for name, counts in report['recovery_counts'].items():
                        print(f'[recovery result] {name}: {counts}', flush=True)
                    print(f'[recovery result] Complete=False; fixed retry budget exhausted; report={output/"format_repair_report.json"}', flush=True)
                    return 1
                recovery.verify_runtime_assets(old)
                (output/'logs').mkdir(exist_ok=True)
                report['gpu_continuation_invoked'] = True
                io.write_json(output/'format_repair_report.json', report)
                code = guarded.launch(new, legacy=queue)
                return finalize_gpu(new, output, source, report, code, LoaderOptions.from_env())
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
