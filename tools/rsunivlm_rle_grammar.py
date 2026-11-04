"""Finite token grammar for model-generated rectangular class/count RLE.

No tensor/model dependency: labels, counts and EOS remain model choices within
the legal finite language. The original decoder must still validate the text.
"""
from __future__ import annotations

import re


class RLEGrammar:
    def __init__(self, tokenizer, labels, rows=24, cols=24):
        if not 1 <= rows <= 64 or not 1 <= cols <= 64:
            raise ValueError('RLE grid sides must be in 1..64')
        self.rows, self.cols = rows, cols
        self.labels = tuple(labels)
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels) or any(
                not re.fullmatch(r'[A-Za-z][A-Za-z _\'-]*', label) for label in self.labels):
            raise ValueError('Expected distinct explicit ASCII class labels')
        self.eos = tokenizer.eos_token_id
        if not isinstance(self.eos, int) or self.eos < 0:
            raise ValueError('Tokenizer needs an explicit integer EOS')
        special = set(getattr(tokenizer, 'all_special_ids', ())) | {self.eos}
        self.encoded = {}
        for label in self.labels:
            for count in range(1, cols+1):
                for end in (', ', '\n', ''):
                    text = f'{label} *{count}{end}'
                    ids = tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=False))
                    if not ids or any(token < 0 or token in special for token in ids):
                        raise ValueError('A grammar fragment contains empty/special token IDs')
                    if tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
                        raise ValueError(f'Tokenizer does not round-trip the exact RLE fragment: {text!r}')
                    self.encoded[text] = ids
        # Every complete run consumes at least one cell. Include final EOS.
        self.required_token_budget = rows*cols*max(map(len, self.encoded.values()))+1
        self.tries = {}
        for remaining in range(1, cols+1):
            for last_row in (False, True):
                tree = {}
                for label in self.labels:
                    for count in range(1, remaining+1):
                        end = ', ' if count < remaining else '' if last_row else '\n'
                        node = tree
                        for token in self.encoded[f'{label} *{count}{end}']:
                            if None in node:
                                raise ValueError('Tokenizer has ambiguous RLE fragment prefixes')
                            node = node.setdefault(token, {})
                        if node:
                            raise ValueError('Tokenizer has colliding RLE fragment tokenizations')
                        node[None] = count
                self.tries[remaining, last_row] = tree

    def prefix_callback(self):
        return RLEPrefix(self)


class RLEPrefix:
    """Greedy batch callback, scoped to one model.generate invocation."""
    def __init__(self, grammar):
        self.grammar = grammar
        self.states = {}

    def __call__(self, batch_id, input_ids):
        grammar = self.grammar
        length = len(input_ids)
        if batch_id not in self.states:
            # Official GMoE uses inputs_embeds; Transformers 4.45.2 starts
            # its generated input_ids at length zero. Never guess a prompt slice.
            if length:
                raise ValueError('Expected an empty generated prefix for official GMoE inputs_embeds')
            self.states[batch_id] = dict(seen=0, row=0, remaining=grammar.cols,
                                        node=grammar.tries[grammar.cols, grammar.rows == 1])
        state = self.states[batch_id]
        if length < state['seen']:
            raise ValueError('RLE prefix rewind/beam search is unsupported; use greedy num_beams=1')
        suffix = input_ids[state['seen']:]
        tokens = suffix.tolist() if hasattr(suffix, 'tolist') else list(suffix)
        for token in tokens:
            token = int(token)
            if state['row'] == grammar.rows:
                if token != grammar.eos:
                    raise ValueError('Only EOS/padded EOS may follow a complete RLE grid')
                continue
            node = state['node'].get(token)
            if node is None:
                raise ValueError('The model produced a token outside the active RLE grammar')
            state['node'] = node
            if None in node:
                state['remaining'] -= node[None]
                if state['remaining'] == 0:
                    state['row'] += 1
                    state['remaining'] = grammar.cols
                if state['row'] != grammar.rows:
                    state['node'] = grammar.tries[state['remaining'], state['row'] == grammar.rows-1]
        state['seen'] = length
        return [grammar.eos] if state['row'] == grammar.rows else list(state['node'])


class RLELogitsProcessor:
    """HF-compatible mask that also supports input_ids shaped (batch, 0).

    The stock 4.45.2 prefix processor uses view(-1, beams, seq_len), which is
    ambiguous when seq_len=0. Enumerate rows directly instead; never add BOS
    or alter the official inputs_embeds sequence to work around that issue.
    """
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, input_ids, scores):
        import torch
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError('RLE logits processor expects matching greedy batch rows')
        masked = torch.full_like(scores, -torch.inf)
        for batch_id in range(input_ids.shape[0]):
            allowed = self.callback(batch_id, input_ids[batch_id])
            values = scores[batch_id, allowed]
            if torch.isnan(values).any() or torch.isposinf(values).any() or not torch.isfinite(values).any():
                raise ValueError('No finite valid model logits for the current RLE state')
            masked[batch_id, allowed] = values
        return masked
