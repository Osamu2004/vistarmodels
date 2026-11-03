"""Lossless delimiter normalization followed by the caller's original decoder."""
from __future__ import annotations

import copy
import hashlib
import re

POLICY = 'explicit_run_separator_normalization_v1'
RULES = ('ASCII colon between an explicit label/count -> *',
         'two or more ASCII asterisks between an explicit label/count -> *')
_RUN = re.compile(r"\s*(?P<label>[A-Za-z][A-Za-z _'-]*)\s*(?P<sep>\*+|:)\s*(?P<count>[0-9]+)\s*")


def normalize_separators(text):
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError('Expected bounded descriptor text')
    lines, edits = [], []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        items = []
        for item_number, item in enumerate(line.split(','), 1):
            match = _RUN.fullmatch(item)
            if match and match['sep'] != '*':
                start, end = match.span('sep')
                items.append(item[:start]+'*'+item[end:])
                edits.append(dict(physical_line=line_number, item=item_number,
                    label=match['label'].strip(), count_text=match['count'],
                    old_separator=match['sep'], new_separator='*'))
            else:
                items.append(item)
        lines.append(','.join(items))
    return ''.join(lines), edits


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def repair_record(record, decoder):
    """Return an unchanged record or a fully decoder-validated copy.

    Already-valid query histories are never reconsidered. Only when every
    original attempt is invalid do we select the earliest repairable attempt.
    No new generation attempt, label, run length, row or EOS flag is invented.
    """
    if not isinstance(record, dict):
        raise ValueError('Expected a query-record object')
    history = record.get('attempts')
    if history is not None and (not isinstance(history, list) or not history):
        raise ValueError('Invalid query attempt history')
    attempts = history if isinstance(history, list) and history else [record]
    if any(not isinstance(attempt, dict) for attempt in attempts):
        raise ValueError('Invalid query attempt history')
    original_errors = []
    for index, attempt in enumerate(attempts):
        try:
            decoder(attempt)
        except (ValueError, KeyError) as error:
            original_errors.append(str(error))
        else:
            return record, dict(status='already_valid', selected_attempt=index)
    rejected = []
    for index, attempt in enumerate(attempts):
        generation = attempt.get('generation', {})
        if generation.get('ended_with_eos') is False:
            rejected.append(dict(attempt=index, error='Explicitly unfinished generation; not repaired'))
            continue
        try:
            candidate_text, edits = normalize_separators(attempt.get('response'))
        except ValueError as error:
            rejected.append(dict(attempt=index, error=str(error)))
            continue
        if not edits:
            rejected.append(dict(attempt=index, error='No permitted delimiter normalization'))
            continue
        candidate = copy.deepcopy(attempt)
        candidate['response'] = candidate_text
        try:
            grid = decoder(candidate)
        except (ValueError, KeyError) as error:
            rejected.append(dict(attempt=index, error=str(error), proposed_edits=len(edits)))
            continue
        audit = dict(policy=POLICY, original_response=attempt['response'],
            original_response_sha256=_digest(attempt['response']),
            normalized_response_sha256=_digest(candidate_text), edits=edits,
            original_decode_error=original_errors[index], decoded_shape=list(grid.shape))
        candidate['format_repair'] = audit
        candidate.pop('decode_error', None)
        result = copy.deepcopy(record)
        if history:
            result['attempts'][index] = candidate
        for field in ('response', 'prompt', 'prompt_variant', 'generation'):
            if field in candidate:
                result[field] = copy.deepcopy(candidate[field])
        result.pop('decode_error', None)
        result['grid_shape'] = list(grid.shape)
        result['format_repair'] = dict(policy=POLICY, selected_attempt=index,
            original_response_sha256=audit['original_response_sha256'],
            normalized_response_sha256=audit['normalized_response_sha256'])
        if not history:
            result['format_repair'].update(audit)
        return result, dict(status='repaired', selected_attempt=index, edits=edits,
                            decoded_shape=list(grid.shape))
    return record, dict(status='unrepairable', original_errors=original_errors, rejected_candidates=rejected)
