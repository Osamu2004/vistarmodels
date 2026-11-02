"""Opt-in LoveDA tile batching and explicit bareland/barren alias recovery."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import baselines.rsunivlm.evaluate as legacy
from baselines.rsunivlm import protocols as protocol
from tools import rsunivlm_batch_runtime as runtime
from tools.rsunivlm_batch_import import build_protocol, import_raw_answers, resolve_output_root


@contextmanager
def alias_decoder():
    original = protocol.label_map
    previous_version = legacy.PROTOCOL_VERSION
    def extended(dataset):
        mapping = original(dataset)
        if dataset == "loveda":
            mapping.update({word: mapping[target] for word, target in runtime.ALIAS_EXTENSION.items()})
        return mapping
    protocol.label_map, legacy.PROTOCOL_VERSION = extended, 4
    try:
        yield
    finally:
        protocol.label_map, legacy.PROTOCOL_VERSION = original, previous_version


class BatchBackend(legacy.LazyModel):
    def __init__(self, args, batch_size, verify, root, protocol_identity):
        super().__init__(args)
        self.requested_batch_size = self.effective_batch_size = batch_size
        self.verify = verify
        self.root = Path(root)
        self.verified_sizes = set()
        self.verification_seconds = 0.
        self.oom_splits = 0
        self.verification_records = []
        previous = legacy.read_json(self.root / "batch_verification.json")
        self.protocol_identity = protocol_identity
        if previous and previous.get("protocol_identity") == self.protocol_identity:
            self.verification_records = previous.get("records", [])
            self.verified_sizes = {row["batch_size"] for row in self.verification_records
                                   if row.get("cases") and all(case["match"] for case in row["cases"])}

    def ensure_model(self):
        if self.model is None:
            from baselines.rsunivlm.model import RSUniVLM
            self.model = RSUniVLM(legacy.normalize_path(self.args.source_root),
                legacy.normalize_path(self.args.model_path), device=self.args.device, dtype=self.args.dtype,
                attention=self.args.attention, max_new_tokens=self.args.max_new_tokens,
                vision_tower=self.args.vision_tower)

    @staticmethod
    def matching(first, second, shape):
        values, valid = [], []
        for record in (first, second):
            try:
                grid = legacy.decode_saved_response(record, "loveda")
                values.append(protocol.restore_grid(grid, shape))
                valid.append(True)
            except (ValueError, KeyError):
                values.append(None)
                valid.append(False)
        if all(valid):
            return bool(np.array_equal(*values))
        # Invalid initial answers are allowed to enter the bounded formatter,
        # but only when both execution paths produced the identical text.
        flags = ("ended_with_eos", "hit_token_limit")
        return (not any(valid) and first["response"] == second["response"] and all(
            first.get("generation", {}).get(key) == second.get("generation", {}).get(key) for key in flags))

    def generate_batch(self, images, prompt):
        self.ensure_model()
        size = len(images)
        answers = runtime.generate_batch(self.model, images, prompt,
            verify_processor=self.verify and size > 1 and size not in self.verified_sizes)
        if self.verify and size > 1 and size not in self.verified_sizes:
            start = time.monotonic()
            checks = []
            for image, answer in zip(images, answers):
                serial = self.model.generate([image], prompt)
                reference = dict(response=serial, generation=copy.deepcopy(self.model.last_generation))
                checks.append(dict(match=self.matching(answer, reference, image.size[::-1]),
                                   batched=answer, serial=reference))
            self.verification_seconds += time.monotonic()-start
            self.verification_records.append(dict(batch_size=size, cases=checks))
            legacy.write_json(self.root / "batch_verification.json", dict(
                protocol_identity=self.protocol_identity,
                policy="exact restored class-mask equality; exact text when both invalid; no GT",
                records=self.verification_records))
            if not all(row["match"] for row in checks):
                raise RuntimeError("Batched/single predictions differ; inspect batch_verification.json; no new image score appended")
            self.verified_sizes.add(size)
            print(f"[RSUniVLM batch] Verified batch{size} against single-image execution", flush=True)
        return answers


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--tile_batch_size", type=int, choices=(1,2,4), default=4)
    extra.add_argument("--import_v4_root", default="")
    extra.add_argument("--verify_batch", type=int, choices=(0,1), default=1)
    options, remaining = extra.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = legacy.parse_args()
    if args.dataset != "loveda" or args.loveda_tile_size != 512:
        raise ValueError("This batch runtime supports LoveDA with 512 source tiles only")
    if args.overwrite:
        raise ValueError("Use a separate output directory instead of overwriting the v4 experiment")
    root = legacy.normalize_path(args.output_dir)
    adapter_sources = sorted((ROOT / "baselines/rsunivlm").glob("*.py"))
    old_base = dict(protocol_version=3, source_revision=protocol.source_revision(legacy.normalize_path(args.source_root)),
        checkpoint=protocol.checkpoint_identity(legacy.normalize_path(args.model_path)),
        dtype=args.dtype, attention=args.attention, max_new_tokens=args.max_new_tokens,
        vision_tower_override=args.vision_tower,
        adapter_source=protocol.fingerprint([path.read_text() for path in adapter_sources]))
    extension_sources = [Path(__file__), ROOT/"tools/rsunivlm_batch_runtime.py", ROOT/"tools/rsunivlm_batch_import.py"]
    new_base = dict(old_base, protocol_version=4, tile_batch_size=options.tile_batch_size,
        alias_extension=runtime.ALIAS_EXTENSION, batch_policy="one image per sequence; adaptive OOM splitting; same greedy prompts",
        batch_verify=bool(options.verify_batch), extension_sources_sha256={str(path.relative_to(ROOT)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in extension_sources})
    expected, _, samples, _ = build_protocol(args, new_base)
    if args.preflight:
        print(json.dumps(dict(preflight="passed; no model loaded", samples=len(samples), protocol=expected), indent=2))
        return
    root, output_resolution = resolve_output_root(root, expected)
    args.output_dir = str(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch_eval.lock").open("a") as lock, alias_decoder():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output_resolution is not None:
            legacy.write_json(root / "output_resolution.json", output_resolution)
        if options.import_v4_root:
            import_raw_answers(options.import_v4_root, root, args, old_base, new_base)
        backend = BatchBackend(args, options.tile_batch_size, options.verify_batch, root,
                               protocol.fingerprint(expected))
        original_predict, original_score = legacy.predict_loveda_tiled, legacy.score
        score_index = 0
        def predict(*values, **kwargs):
            result = runtime.predict_tiled(*values, **kwargs)
            info = result[1]
            print(f"[RSUniVLM batch] {values[2].name}: tiles={info['num_tiles']}, "
                  f"raw_reused={info['reused_queries']}, batch={backend.effective_batch_size}, "
                  f"generate={info['tile_generation_seconds']:.2f}s, verify={info['batch_verification_seconds']:.2f}s", flush=True)
            return result
        def score(*values, **kwargs):
            nonlocal score_index
            start = time.monotonic()
            result = original_score(*values, **kwargs)
            event = dict(sample=samples[score_index].name, scoring_seconds=time.monotonic()-start,
                         boundary_wfm=args.boundary_wfm)
            score_index += 1
            with (root / "scoring_timing.jsonl").open("a") as stream:
                stream.write(json.dumps(event)+"\n")
            return result
        legacy.predict_loveda_tiled, legacy.score = predict, score
        try:
            result = legacy.evaluate_dataset(args, "loveda", backend, new_base)
            result.update(tile_batch_size=options.tile_batch_size, effective_tile_batch_size=backend.effective_batch_size,
                          alias_extension=runtime.ALIAS_EXTENSION, verified_batch_sizes=sorted(backend.verified_sizes),
                          oom_splits=backend.oom_splits)
            legacy.write_json(root / "metrics.json", result)
            print(json.dumps(dict(batch_output=str(root), requested_batch_size=options.tile_batch_size,
                effective_batch_size=backend.effective_batch_size, verified_batch_sizes=sorted(backend.verified_sizes),
                oom_splits=backend.oom_splits), indent=2), flush=True)
        finally:
            legacy.predict_loveda_tiled, legacy.score = original_predict, original_score


if __name__ == "__main__":
    main()
