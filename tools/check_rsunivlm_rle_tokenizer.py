"""WSL CPU-only local-tokenizer grammar preflight; no model or GT loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(plan, rows, cols):
    from transformers import Qwen2TokenizerFast
    from tools import retry_rsunivlm_failed as recovery
    from tools import rsunivlm_constrained_rle as rle
    from tools.rsunivlm_rle_grammar import RLEGrammar
    path = plan['protocol']['model_settings']['model_path']
    tokenizer = Qwen2TokenizerFast.from_pretrained(path, local_files_only=True)
    checked, largest_budget = [], 0
    for entry in plan['entries']:
        name = entry['dataset']
        for key in recovery.keys(plan, name):
            labels = rle.labels_for(plan, name, key)
            grammar = RLEGrammar(tokenizer, labels, rows, cols)
            largest_budget = max(largest_budget, grammar.required_token_budget)
            for mixed in (False, True):
                tokens = []
                for row in range(rows):
                    runs = [(labels[0], cols)] if not mixed or cols == 1 else [(labels[0], cols//2), (labels[1], cols-cols//2)]
                    for index, (label, count) in enumerate(runs):
                        end = ', ' if index+1 < len(runs) else '\n' if row+1 < rows else ''
                        tokens.extend(grammar.encoded[f'{label} *{count}{end}'])
                callback, prefix = grammar.prefix_callback(), []
                for token in tokens+[grammar.eos]:
                    if token not in callback(0, prefix):
                        raise ValueError(f'Tokenizer grammar transition failed: {name}/{key}')
                    prefix.append(token)
                text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
                record = dict(response=text, generation=dict(ended_with_eos=True, hit_token_limit=False))
                grid = recovery.decoded(plan, name, key, record)
                if list(grid.shape) != [rows, cols] or len(prefix) > grammar.required_token_budget:
                    raise ValueError(f'Whole-text token composition/decoder check failed: {name}/{key}')
            checked.append(dict(dataset=name, key=key, labels=labels, required_token_budget=grammar.required_token_budget))
    if largest_budget > plan['protocol']['model_settings']['max_new_tokens']:
        raise ValueError(f'Increase the explicit generation budget to at least {largest_budget}')
    return dict(complete=True, rows=rows, cols=cols, checked_queries=len(checked),
        tokenizer_class=type(tokenizer).__name__, largest_required_token_budget=largest_budget,
        model_forward=False, cuda_called=False, GT_loaded=False, synthetic_grammar_checks=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', required=True)
    parser.add_argument('--rows', type=int, default=24)
    parser.add_argument('--cols', type=int, default=24)
    args = parser.parse_args()
    source = Path(args.source_run).expanduser().resolve()
    plan = json.loads((source/'plan.json').read_text())
    report = check(plan, args.rows, args.cols)
    output = source.with_name(source.name+f'_rle_tokenizer_check_{args.rows}x{args.cols}.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key:value for key, value in report.items() if key != 'synthetic_grammar_checks'}, ensure_ascii=False))
    print(f'RLE tokenizer preflight: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
