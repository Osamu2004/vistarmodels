"""Opt-in finite RLE fallback: explicit metadata, original decoding, no GT."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import re

from tools.rsunivlm_rle_grammar import RLEGrammar, RLELogitsProcessor

POLICY = 'finite_query_binary_rle_greedy_v1'
REFERENCE = 'https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/text_generation'


def settings(rows=24, cols=24):
    if not 1 <= rows <= 64 or not 1 <= cols <= 64:
        raise ValueError('Constrained RLE grid sides must be in 1..64')
    return dict(policy=POLICY, rows=rows, cols=cols,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grammar_sha256=hashlib.sha256(Path(__file__).with_name('rsunivlm_rle_grammar.py').read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(__file__).with_name('rsunivlm_constrained_runtime.py').read_bytes()).hexdigest(),
        labels='others and the fixed query foreground; never read GT',
        activation='last fixed retry; one additional constrained answer for exhausted legacy histories',
        legacy_extra_answers=1, maximum_total_saved_answers=4,
        changes_generation=True, official_original_protocol=False,
        generation_api='HF logits_processor; empty-prefix-safe finite RLE mask; greedy num_beams=1', reference=REFERENCE)


def labels_for(plan, dataset, key):
    from tools import rsunivlm_native_semantic as native
    from tools import rsunivlm_paper_protocol as protocol
    if native.enabled(plan['protocol'].get('semantic_mode'), dataset):
        foreground = native.query(dataset, key)[2]
    elif dataset == 'second':
        foreground = protocol.second_query(key)[1]
    elif dataset == 'levircd':
        foreground = 'building'
    else:
        raise ValueError('Constrained binary fallback requires the native per-class semantic protocol')
    return ['others', foreground]


def metadata(plan, dataset, key):
    configured = plan['protocol']['constrained_rle']
    return dict(policy=POLICY, rows=configured['rows'], cols=configured['cols'],
        labels=labels_for(plan, dataset, key), helper_sha256=configured['helper_sha256'],
        algorithm='finite_rle_logits_processor; greedy num_beams=1')


def validate_grid(record, grid):
    value = record.get('generation', {}).get('rle_constraints')
    if not value:
        return
    rows, cols = value.get('rows'), value.get('cols')
    if value.get('policy') != POLICY or not isinstance(rows, int) or not isinstance(cols, int) or not 1 <= rows <= 64 or not 1 <= cols <= 64:
        raise ValueError('Unknown or invalid constrained RLE metadata')
    if list(grid.shape) != [rows, cols] or record.get('generation', {}).get('ended_with_eos') is not True:
        raise ValueError('Constrained RLE must be a complete declared grid ending in actual EOS')
    if record.get('generation', {}).get('constrained_validation_error'):
        raise ValueError(record['generation']['constrained_validation_error'])
    allowed = set(value.get('labels', []))
    lines = record.get('response', '').splitlines()
    if len(lines) != rows or len(allowed) != 2 or 'others' not in allowed:
        raise ValueError('Constrained RLE has foreign labels or noncanonical rows')
    for line in lines:
        for item in line.split(','):
            match = re.fullmatch(r'\s*([A-Za-z][A-Za-z _\'-]*)\s*\*\s*([1-9][0-9]*)\s*', item)
            if not match or match[1].strip() not in allowed:
                raise ValueError('Constrained RLE text escaped its declared labels/count grammar')


def validate_history(plan, dataset, key, attempts, instruction):
    budget = plan['protocol']['format_retries']
    configured = plan['protocol'].get('constrained_rle')
    maximum = budget+1+int(bool(configured))
    if not isinstance(attempts, list) or not attempts or len(attempts) > maximum:
        raise ValueError('Invalid source attempt history')
    if sum(bool(attempt.get('generation', {}).get('rle_constraints')) for attempt in attempts) > 1:
        raise ValueError('Only one constrained answer is allowed per query history')
    for index, attempt in enumerate(attempts):
        extra = index > budget
        if attempt.get('prompt_variant') != index or attempt.get('prompt') != instruction(plan, dataset, key, min(index, budget)):
            raise ValueError('Source prompt differs')
        info = attempt.get('generation', {}).get('rle_constraints')
        if info:
            if not configured or index < budget or info != metadata(plan, dataset, key):
                raise ValueError('Source constrained decoder metadata differs')
        elif extra:
            raise ValueError('An extra attempt requires the explicitly recorded RLE fallback')


def annotate_scored_samples(plan, dataset, rank, output, result):
    if not plan['protocol'].get('constrained_rle'):
        return
    from baselines.rsunivlm import evaluate as io
    from tools import retry_rsunivlm_failed as recovery
    root = Path(output)/dataset/'shards'/f'rank{rank:03d}'
    samples = []
    for row in result.get('rows', []):
        path = root/'receipts'/f'{row["sample"]}.json'
        receipt = io.read_json(path)
        if not receipt:
            raise ValueError('A scored constrained-recovery sample has no prediction receipt')
        queries = receipt.get('constrained_rle_queries', [])
        if not row.get('reused_prediction'):
            queries = []
            height, width = receipt['shape']; size = plan['protocol']['tile_size']
            for key in recovery.keys(plan, dataset):
                for y in range(0, height, size):
                    for x in range(0, width, size):
                        raw_path = root/'raw_text'/f'{row["sample"]}_{key}_y{y}_x{x}.json'
                        record = io.read_json(raw_path) or {}
                        for index, attempt in enumerate(record.get('attempts', [])):
                            try:
                                recovery.decoded(plan, dataset, key, attempt)
                            except (ValueError, KeyError):
                                continue
                            if attempt.get('generation', {}).get('rle_constraints'):
                                queries.append(dict(path=str(raw_path), selected_attempt=index,
                                    constraints=attempt['generation']['rle_constraints']))
                            break
            if queries:
                receipt.update(constrained_rle_queries=queries, constrained_rle_query_count=len(queries))
            else:
                receipt.pop('constrained_rle_queries', None); receipt.pop('constrained_rle_query_count', None)
            io.write_json(path, receipt)
        if queries:
            samples.append(dict(sample=row['sample'], constrained_queries=len(queries)))
    io.write_json(root/'constrained_samples.json', dict(identity=plan['identity'], dataset=dataset,
        rank=rank, successfully_scored_constrained_samples=len(samples), samples=samples))


@contextmanager
def bind_decoders(plan):
    configured = plan['protocol'].get('constrained_rle')
    if not configured:
        yield
        return
    if configured != settings(configured['rows'], configured['cols']):
        raise ValueError('Constrained RLE source/policy differs from the saved plan')
    from tools import rsunivlm_native_semantic as native
    from tools import rsunivlm_paper_protocol as protocol
    old_native, old_decode = native.decode_binary, protocol.decode
    def binary(record):
        value = old_native(record)
        validate_grid(record, value[0])
        return value
    def decode(record, dataset, query_key=None):
        value = old_decode(record, dataset, query_key=query_key)
        validate_grid(record, value)
        return value
    native.decode_binary, protocol.decode = binary, decode
    try:
        yield
    finally:
        native.decode_binary, protocol.decode = old_native, old_decode


def generate_answers(core, requests, prompt, plan, dataset, key):
    from transformers.generation.logits_process import LogitsProcessorList
    from tools.rsunivlm_batch_runtime import generate_requests
    configured = plan['protocol']['constrained_rle']
    cache = getattr(core, '_rle_grammar_cache', None)
    if cache is None:
        core._rle_grammar_cache = cache = {}
    label_names = labels_for(plan, dataset, key)
    cache_key = (tuple(label_names), configured['rows'], configured['cols'])
    if cache_key not in cache:
        cache[cache_key] = RLEGrammar(core.tokenizer, label_names, configured['rows'], configured['cols'])
    grammar = cache[cache_key]
    if core.max_new_tokens < grammar.required_token_budget:
        raise ValueError(f'Constrained output token budget insufficient: need >= {grammar.required_token_budget}, got {core.max_new_tokens}')
    callback = grammar.prefix_callback()
    had_override = 'generate' in vars(core.model)
    previous_override = vars(core.model).get('generate')
    original = core.model.generate
    def generate(*args, **kwargs):
        if kwargs.get('do_sample', False) or kwargs.get('num_beams', 1) != 1 or kwargs.get('prefix_allowed_tokens_fn') is not None or kwargs.get('logits_processor'):
            raise ValueError('RLE fallback requires ordinary greedy generation without another prefix callback')
        kwargs.update(prefix_allowed_tokens_fn=None, logits_processor=LogitsProcessorList([RLELogitsProcessor(callback)]),
            eos_token_id=grammar.eos, pad_token_id=grammar.eos,
            min_new_tokens=0, min_length=0, forced_bos_token_id=None, forced_eos_token_id=None,
            no_repeat_ngram_size=0, bad_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
            max_time=None, stop_strings=None)
        return original(*args, **kwargs)
    core.model.generate = generate
    try:
        answers = generate_requests(core, requests, prompt)
    finally:
        if had_override:
            core.model.generate = previous_override
        else:
            delattr(core.model, 'generate')
    result = copy.deepcopy(answers)
    for index, answer in enumerate(result):
        complete = callback.states.get(index, {}).get('row') == grammar.rows
        answer.setdefault('generation', {}).update(decoding='greedy_constrained_rle',
            rle_constraints=metadata(plan, dataset, key),
            grammar_worst_case_tokens=grammar.required_token_budget,
            constrained_eos_token_id=grammar.eos, grammar_completed=complete)
        if not complete:
            answer['generation']['constrained_validation_error'] = 'The actual prefix callback did not complete the declared RLE grid'
    return result
