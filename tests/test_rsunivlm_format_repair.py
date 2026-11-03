"""WSL-only synthetic delimiter repair checks, with the original decoders."""
import copy

import numpy as np
import pytest

from tools import rsunivlm_format_repair as repair
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_protocol as protocol


def decode(record):
    return native.decode_binary(record)[0]


def record(text, **generation):
    return dict(response=text, generation={'ended_with_eos': True, 'hit_token_limit': False, **generation})


@pytest.mark.parametrize('before,after,shape', [
    ('road**2, others:22\nothers*24', 'road*2, others*22\nothers*24', (2,24)),
    ('building:1', 'building*1', (1,1)),
    ('road***24\nothers*24', 'road*24\nothers*24', (2,24)),
    ('  Road  ** 02, others:22\r\nothers*24\r\n', '  Road  * 02, others*22\r\nothers*24\r\n', (2,24)),
    ('```text\nmask:\nroad**24\nothers:24\n```', '```text\nmask:\nroad*24\nothers*24\n```', (2,24)),
])
def test_only_explicit_separators_change_and_whole_original_decoder_accepts(before, after, shape):
    source = record(before)
    original = copy.deepcopy(source)
    fixed, result = repair.repair_record(source, decode)
    assert result['status'] == 'repaired' and fixed['response'] == after
    assert tuple(decode(fixed).shape) == shape
    np.testing.assert_array_equal(decode(fixed), decode(record(after)))
    assert fixed['generation'] == source['generation'] and source == original
    assert fixed['format_repair']['original_response'] == before
    again, status = repair.repair_record(fixed, decode)
    assert status['status'] == 'already_valid' and again is fixed


@pytest.mark.parametrize('text', [
    'road', 'road**0', 'road**65', 'road**-1', 'road**2.5',
    'road**24\nothers:23', 'road**2, water', 'road**24,',
    'road:**24', 'road：24',
    '```\nroad**24', '**road**24',
])
def test_missing_counts_bad_geometry_or_unknown_syntax_are_never_filled(text):
    source = record(text)
    original = copy.deepcopy(source)
    unchanged, result = repair.repair_record(source, decode)
    assert result['status'] == 'unrepairable'
    assert unchanged is source and unchanged == original
    assert 'format_repair' not in unchanged


@pytest.mark.parametrize('hit_limit', [False, True])
def test_explicit_unfinished_generation_and_eos_flags_are_not_repaired(hit_limit):
    source = dict(response='road**24', generation=dict(ended_with_eos=False, hit_token_limit=hit_limit))
    unchanged, result = repair.repair_record(source, decode)
    assert unchanged is source and result['status'] == 'unrepairable'
    assert unchanged['generation'] == dict(ended_with_eos=False, hit_token_limit=hit_limit)


def test_existing_valid_attempt_wins_without_repairing_an_earlier_bad_attempt():
    attempts = [dict(record('road**24'), prompt_variant=0), dict(record('others*24'), prompt_variant=1)]
    source = dict(response='others*24', attempts=attempts)
    original = copy.deepcopy(source)
    result, audit = repair.repair_record(source, decode)
    assert result is source and result == original
    assert audit == dict(status='already_valid', selected_attempt=1)


def test_first_decoder_valid_repair_selected_without_new_generation_attempt():
    attempts = [dict(record('road**24\nroad*23'), prompt_variant=0, prompt='first'),
                dict(record('road:24\nothers*24'), prompt_variant=1, prompt='second')]
    source = dict(attempts=attempts, response=attempts[-1]['response'])
    original = copy.deepcopy(source)
    fixed, audit = repair.repair_record(source, decode)
    assert audit['selected_attempt'] == 1 and len(fixed['attempts']) == 2
    assert fixed['attempts'][0] == original['attempts'][0]
    assert fixed['attempts'][1]['format_repair']['original_response'] == attempts[1]['response']
    assert fixed['prompt'] == 'second' and fixed['prompt_variant'] == 1
    assert fixed['attempts'][1]['generation'] == attempts[1]['generation']
    assert source == original


def test_original_category_mapping_is_not_extended_by_separator_repair():
    water = lambda rec: protocol.decode(rec, 'second', query_key='water')
    fixed, audit = repair.repair_record(record('water**24\nothers:24'), water)
    assert audit['status'] == 'repaired' and water(fixed).tolist() == [[1]*24, [0]*24]
    source = record('building:24\nothers*24')
    unchanged, audit = repair.repair_record(source, water)
    assert unchanged is source and audit['status'] == 'unrepairable'


def test_native_label_semantics_remain_exactly_as_in_the_original_decoder():
    # The official-demo binary decoder treats every syntactically valid label
    # except "others" as foreground; normalization must not add a whitelist.
    source = record('mask says road**24')
    fixed, audit = repair.repair_record(source, decode)
    assert audit['status'] == 'repaired'
    assert fixed['response'] == 'mask says road*24'
    np.testing.assert_array_equal(decode(fixed), decode(record('mask says road*24')))


def test_valid_response_is_byte_identical_and_unsupported_history_is_rejected():
    source = record('others *24\n road *24\n')
    unchanged, result = repair.repair_record(source, decode)
    assert unchanged is source and result['status'] == 'already_valid'
    assert unchanged['response'] == 'others *24\n road *24\n'
    for history in ([], {}, ['not a record']):
        with pytest.raises(ValueError, match='history'):
            repair.repair_record(dict(response='road**24', attempts=history), decode)
