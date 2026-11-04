"""Validate mechanical format repair before consuming the next fixed retry."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from tools import rsunivlm_format_repair as normalization

POLICY = 'normalize_each_returned_answer_then_original_decode_v1'


def settings():
    return dict(policy=POLICY, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalizer_sha256=hashlib.sha256(Path(normalization.__file__).read_bytes()).hexdigest(),
        changes_model_forward=False, changes_prompts=False, changes_geometry=False,
        extra_generation_attempts=0, selection='first valid original or fully validated mechanical repair; no GT selection')


def install(engine_class, plan, rank):
    configured = plan['protocol'].get('runtime_format_repair')
    if not configured:
        return
    if configured != settings():
        raise ValueError('Runtime format repair source/policy differs from the saved plan')
    from baselines.rsunivlm import evaluate as io
    from tools import rsunivlm_native_semantic as native
    from tools import rsunivlm_paper_protocol as protocol
    original_forward, original_evaluate = engine_class.forward, engine_class.evaluate
    path = Path(plan['output_dir'])/f'runtime_format_repair_rank{rank:03d}.json'

    def evaluate(self, entry, output):
        previous = getattr(self, '_repair_dataset', None)
        self._repair_dataset = entry['dataset']
        try:
            return original_evaluate(self, entry, output)
        finally:
            self._repair_dataset = previous
            io.write_json(path, dict(identity=plan['identity'], rank=rank,
                datasets=getattr(self, '_repair_counts', {}), policy=configured))

    def forward(self, jobs, prompt):
        dataset = getattr(self, '_repair_dataset', None)
        if dataset is None:
            raise ValueError('Runtime repair requires an active dataset context')
        if not hasattr(self, '_repair_counts'):
            self._repair_counts = {}
        counts = self._repair_counts.setdefault(dataset, dict(checked=0, already_valid=0, repaired=0, unrepairable=0))
        for group, answers in original_forward(self, jobs, prompt):
            if len(answers) != len(group):
                raise ValueError('Cannot repair a mismatched answer batch')
            fixed = []
            for job, answer in zip(group, answers):
                key = job.get('key', job.get('direction'))
                def decode(candidate):
                    if native.enabled(plan['protocol'].get('semantic_mode'), dataset):
                        return native.decode_binary(candidate)[0]
                    return protocol.decode(candidate, dataset, query_key=key if dataset == 'second' else None)
                value, result = normalization.repair_record(copy.deepcopy(answer), decode)
                counts['checked'] += 1
                status = result['status']
                counts[status] += 1
                if status == 'repaired':
                    print(f'[recovery format] {dataset}/{job["path"].name}: separator repaired; original decoder accepted', flush=True)
                fixed.append(value)
            # The original evaluator saves this actual answer as one attempt.
            # Its response, generation/EOS flags and exact pre-repair text survive.
            yield group, fixed

    engine_class.forward, engine_class.evaluate = forward, evaluate
