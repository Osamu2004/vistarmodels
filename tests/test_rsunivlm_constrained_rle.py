"""WSL-only grammar and cache regressions, with fake logits/model answers."""
import copy
import random
from types import SimpleNamespace

import pytest
import torch

from baselines.rsunivlm import evaluate as io
from tools import repair_rsunivlm_format as tool
from tools import retry_rsunivlm_failed as recovery
from tools import rsunivlm_constrained_rle as rle
from tools import rsunivlm_constrained_runtime as runtime
from tools import rsunivlm_native_semantic as native
from tools import rsunivlm_paper_worker as worker
from tools.rsunivlm_rle_grammar import RLEGrammar, RLELogitsProcessor
from test_rsunivlm_format_repair_pipeline import setup


class ByteTokenizer:
    eos_token_id = 256
    all_special_ids = [256]
    def encode(self, text, **kwargs):
        return list(text.encode('ascii'))
    def decode(self, tokens, skip_special_tokens=False, **kwargs):
        return bytes(token for token in tokens if token != 256 or not skip_special_tokens).decode('ascii')


@pytest.mark.parametrize('rows,cols', [(1, 1), (2, 3), (3, 2)])
def test_every_sampled_grammar_path_is_a_rectangular_original_decoder_input(rows, cols):
    grammar = RLEGrammar(ByteTokenizer(), ['others', 'water'], rows, cols)
    rng = random.Random(0)
    for _ in range(20):
        callback, tokens = grammar.prefix_callback(), []
        assert grammar.eos not in callback(0, tokens)
        for step in range(grammar.required_token_budget):
            allowed = callback(0, tokens)
            token = rng.choice(allowed)
            tokens.append(token)
            if token == grammar.eos:
                break
        assert tokens[-1] == grammar.eos
        text = ByteTokenizer().decode(tokens[:-1])
        grid, labels = native.decode_binary(dict(response=text, generation=dict(ended_with_eos=True)))
        assert list(grid.shape) == [rows, cols] and set(labels) <= {'others', 'water'}
        assert callback(0, tokens) == [grammar.eos]
        assert callback(1, [])  # Another batch member has an independent empty state.


def test_hf_logits_processor_masks_illegal_high_logits_on_cpu():
    from transformers.generation.logits_process import LogitsProcessorList
    grammar = RLEGrammar(ByteTokenizer(), ['others', 'water'], 1, 2)
    processor = LogitsProcessorList([RLELogitsProcessor(grammar.prefix_callback())])
    prefix = torch.empty((1, 0), dtype=torch.long)
    for _ in range(grammar.required_token_budget):
        logits = torch.zeros((1, 257)); logits[0, 0] = 1000  # Illegal token is deliberately best.
        masked = processor(prefix, logits)
        token = int(masked.argmax(dim=-1).item())
        assert token != 0
        prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)
        if token == grammar.eos:
            break
    assert token == grammar.eos
    assert native.decode_binary(dict(response=ByteTokenizer().decode(prefix[0, :-1].tolist())))[0].shape == (1, 2)


def test_asymmetric_batch_completion_accepts_only_eos_padding_and_rejects_nonfinite_logits():
    grammar = RLEGrammar(ByteTokenizer(), ['others', 'water'], 1, 2)
    processor = RLELogitsProcessor(grammar.prefix_callback())
    sequences = [ByteTokenizer().encode('others *2')+[256],
                 ByteTokenizer().encode('water *1, others *1')+[256]]
    prefix = torch.empty((2, 0), dtype=torch.long)
    for step in range(max(map(len, sequences))):
        masked = processor(prefix, torch.zeros((2, 257)))
        chosen = [row[step] if step < len(row) else 256 for row in sequences]
        assert all(torch.isfinite(masked[index, token]) for index, token in enumerate(chosen))
        prefix = torch.cat((prefix, torch.tensor(chosen).view(2, 1)), dim=1)
    empty = torch.empty((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match='finite valid model logits'):
        RLELogitsProcessor(grammar.prefix_callback())(empty, torch.full((1, 257), -torch.inf))


def test_reject_prompt_slicing_invalid_tokens_nonroundtrip_and_bad_grid():
    grammar = RLEGrammar(ByteTokenizer(), ['others', 'water'], 2, 3)
    with pytest.raises(ValueError, match='empty generated prefix'):
        grammar.prefix_callback()(0, [99])
    callback = grammar.prefix_callback(); callback(0, [])
    with pytest.raises(ValueError, match='outside'):
        callback(0, [0])
    class Broken(ByteTokenizer):
        def decode(self, *args, **kwargs):
            return ''
    with pytest.raises(ValueError, match='round-trip'):
        RLEGrammar(Broken(), ['others', 'water'], 2, 3)
    with pytest.raises(ValueError, match='1..64'):
        RLEGrammar(ByteTokenizer(), ['others', 'water'], 0, 3)


def configured_plan():
    return dict(protocol=dict(semantic_mode=native.MODE, format_retries=2, constrained_rle=rle.settings(2, 3)))


def test_generate_hook_reaches_backend_and_restores_original_model_method(monkeypatch):
    from tools import rsunivlm_batch_runtime as batch
    plan = configured_plan(); key = native.queries('flair')[0][0]
    class Model:
        def generate(self, inputs, **kwargs):
            processor, tokens = kwargs['logits_processor'], []
            assert kwargs['eos_token_id'] == 256 and kwargs['min_new_tokens'] == 0
            for _ in range(1000):
                logits = torch.zeros((1, 257)); logits[0, 0] = 1000
                masked = processor(torch.tensor([tokens], dtype=torch.long), logits)
                token = int(masked.argmax(dim=-1).item()); tokens.append(token)
                if token == 256: break
            return tokens
    model = Model(); original = model.generate
    core = SimpleNamespace(tokenizer=ByteTokenizer(), model=model, max_new_tokens=10000)
    def generate(core, requests, prompt):
        tokens = core.model.generate(None, do_sample=False, num_beams=1)
        return [dict(response=core.tokenizer.decode(tokens, skip_special_tokens=True),
                     generation=dict(ended_with_eos=tokens[-1] == 256, hit_token_limit=False))]
    monkeypatch.setattr(batch, 'generate_requests', generate)
    answer = rle.generate_answers(core, [[object()]], native.prompt('flair', key, 2), plan, 'flair', key)[0]
    assert model.generate == original and 'generate' not in vars(model)
    grid = recovery.decoded(plan, 'flair', key, answer)
    assert grid.shape == (2, 3) and answer['generation']['rle_constraints'] == rle.metadata(plan, 'flair', key)
    core.max_new_tokens = 1
    with pytest.raises(ValueError, match='budget insufficient'):
        rle.generate_answers(core, [[object()]], '[SEG] unchanged', plan, 'flair', key)


def test_extra_history_is_explicit_and_decoder_rejects_wrong_grid_or_foreign_labels():
    plan = configured_plan(); key = native.queries('flair')[0][0]
    attempts = [dict(prompt_variant=index, prompt=recovery.instruction(plan, 'flair', key, index),
                     response='coni', generation={}) for index in range(3)]
    extra = dict(prompt_variant=3, prompt=recovery.instruction(plan, 'flair', key, 2),
        response='others *3\nothers *3', generation=dict(ended_with_eos=True, rle_constraints=rle.metadata(plan, 'flair', key)))
    rle.validate_history(plan, 'flair', key, attempts+[extra], recovery.instruction)
    assert recovery.decoded(plan, 'flair', key, extra).shape == (2, 3)
    for bad_text in ('others *3', 'water *3\nothers *3'):
        bad = copy.deepcopy(extra); bad['response'] = bad_text
        with pytest.raises(ValueError, match='grid|labels'):
            recovery.decoded(plan, 'flair', key, bad)
    bad = copy.deepcopy(extra); bad['generation']['ended_with_eos'] = False
    with pytest.raises(ValueError, match='EOS'):
        recovery.decoded(plan, 'flair', key, bad)
    with pytest.raises(ValueError, match='attempt history'):
        rle.validate_history(plan, 'flair', key, attempts+[extra, extra], recovery.instruction)


@pytest.mark.parametrize('exhausted,malformed', [(True, False), (True, True), (False, False)])
def test_only_failed_queries_use_one_constrained_answer_and_resume_without_replay(tmp_path, monkeypatch, exhausted, malformed):
    source, original, entry, active = setup(tmp_path, monkeypatch)
    key = recovery.keys(original, 'flair')[0]
    raw_name = f's0_{key}_y0_x0.json'
    raw = source/'flair/shards/rank000/raw_text'/raw_name
    record = io.read_json(raw); record['response'] = record['attempts'][0]['response'] = 'building'
    io.write_json(raw, record)
    old, plan = tool.prepare(source, [], True, constrained_rle=True, rle_rows=2, rle_cols=3)
    output = tmp_path/'rle'; plan['output_dir'] = str(output); active[0] = output
    recovery.import_predictions(source, output, old, plan)
    target = output/'flair/shards/rank000/raw_text'/raw_name
    if exhausted:
        record = io.read_json(target)
        for variant in (1, 2):
            attempt = dict(prompt_variant=variant, prompt=recovery.instruction(plan, 'flair', key, variant),
                           response='coni', generation=dict(ended_with_eos=True, hit_token_limit=False))
            record['attempts'].append(attempt); record.update(attempt)
        io.write_json(target, record)
    tool.score_offline(plan, output, tool.LoaderOptions(num_workers=0))
    calls, free = [], []
    def generate(core, requests, prompt, current, dataset, query_key):
        calls.append((dataset, query_key, len(requests)))
        return [dict(response='others *2' if malformed else 'others *3\nothers *3',
            generation=dict(ended_with_eos=True, hit_token_limit=False,
                            rle_constraints=rle.metadata(current, dataset, query_key))) for _ in requests]
    monkeypatch.setattr(rle, 'generate_answers', generate)
    def free_generate(core, requests, prompt):
        free.append(prompt)
        return [dict(response='coni', generation=dict(ended_with_eos=True, hit_token_limit=False)) for _ in requests]
    monkeypatch.setattr(worker, 'generate_requests', free_generate)
    class Engine(worker.Engine):
        pass
    runtime.install(Engine, plan, 0)
    result = Engine(plan, 0, 2).evaluate(entry, str(output))
    assert result['complete'] == (not malformed) and len(calls) == 1
    assert len(free) == (0 if exhausted else 1)
    saved = io.read_json(target)
    assert len(saved['attempts']) == (4 if exhausted else 3)
    assert saved['attempts'][-1]['generation']['rle_constraints'] == rle.metadata(plan, 'flair', key)
    rle.validate_history(plan, 'flair', key, saved['attempts'], recovery.instruction)
    manifest = io.read_json(output/'flair/shards/rank000/constrained_samples.json')
    assert manifest['successfully_scored_constrained_samples'] == (0 if malformed else 1)
    if not malformed:
        assert io.read_json(output/'flair/shards/rank000/receipts/s0.json')['constrained_rle_query_count'] == 1
    Engine(plan, 0, 2).evaluate(entry, str(output))
    assert len(calls) == 1  # A valid result or a used constrained attempt is never replayed.
    assert io.read_json(raw)['response'] == 'building'
    if malformed:
        assert not list((output/'flair/shards/rank000/pred_mask').glob('s0_*.png'))
