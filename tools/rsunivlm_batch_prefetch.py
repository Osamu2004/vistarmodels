"""Bounded RGB-only lookahead to fill genuine cross-image tile batches.

The original evaluator still restores, saves and scores each source image.
Lookahead never loads GT pixels or retains images after raw answers are saved.
"""
from __future__ import annotations

from pathlib import Path
import math
import time

from PIL import Image

import baselines.rsunivlm.evaluate as legacy
from baselines.rsunivlm.protocols import fingerprint, input_identity
from tools import rsunivlm_batch_runtime as runtime

PREFETCH_POLICY = "contiguous RGB sources up to lcm(batch,4) tiles; bounded source lookahead; no GT pixels"


class CrossImagePrefetch:
    def __init__(self, samples, protocol_identity):
        self.samples = list(samples)
        self.positions = {sample.name: index for index, sample in enumerate(self.samples)}
        self.protocol_identity = protocol_identity
        self.pending_info = {}

    def prepare(self, backend, image, sample, raw_dir, identity, *, tile_size, overwrite):
        if sample.name in self.pending_info:
            return self.pending_info.pop(sample.name)
        requested = backend.requested_batch_size
        if requested <= 0:
            raise ValueError("Batch size must be a positive integer")
        target_tiles = math.lcm(requested,4)
        start = self.positions[sample.name]
        plans, jobs = [], []
        output = Path(raw_dir).parent
        # Normal LoveDA sources have four tiles. LCM keeps odd-sized batches
        # full too, while at most B source images are held in this old entry.
        for source in self.samples[start:start+requested]:
            item_id = fingerprint([self.protocol_identity, input_identity(source)])
            if source.name == sample.name:
                if item_id != identity:
                    raise ValueError("Current sample identity differs from prefetch protocol")
                rgb = image
            else:
                with Image.open(source.images[0]) as opened:
                    rgb = opened.convert("RGB")
            try:
                saved = legacy.read_json(output / "receipts" / f"{source.name}.json")
                cached = None if overwrite else legacy.cached_prediction(
                    output / "pred_mask" / f"{source.name}_pred_mask.png", saved,
                    identity=item_id, shape=(rgb.height,rgb.width), num_classes=7)
                if cached is not None:
                    continue
                source_jobs = runtime.make_tile_jobs(rgb, source, raw_dir, item_id, tile_size=tile_size)
                plans.append((source.name, source_jobs))
                jobs.extend(source_jobs)
            finally:
                if rgb is not image:
                    rgb.close()
            if len(jobs) >= target_tiles:
                break
        if not jobs:
            return {}
        group_id = fingerprint([job["identity"] for job in jobs])
        started, before_verify = time.monotonic(), backend.verification_seconds
        # Failed future descriptors stay in their raw records. The strict
        # per-source predictor raises when that source reaches the scorer;
        # valid earlier sources are never silently skipped or mislabelled.
        grids, reused = runtime.query_tiles(backend, jobs, overwrite=overwrite, defer_format_errors=True)
        wall, verify = time.monotonic()-started, backend.verification_seconds-before_verify
        anchor = plans[0][0]
        for name, source_jobs in plans:
            sizes = set()
            for job in source_jobs:
                record = legacy.read_json(job["path"]) or {}
                size = record.get("generation", {}).get("inference_batch_size")
                if size is not None:
                    sizes.add(size)
            self.pending_info[name] = dict(
                prefetch_group_id=group_id, prefetch_source_images=len(plans),
                prefetch_tile_queries=len(jobs), prefetch_group_wall_seconds=wall,
                prefetch_wall_seconds_charged=wall if name == anchor else 0.,
                prefetch_verify_seconds_charged=verify if name == anchor else 0.,
                prefetch_reused_queries=sum(bool(reused.get(job["path"])) for job in source_jobs),
                prefetch_failed_queries=sum(job["path"] not in grids for job in source_jobs),
                generation_actual_batch_sizes=sorted(sizes))
        print(f"[RSUniVLM prefetch] sources={len(plans)}, tiles={len(jobs)}, "
              f"new_or_failed={len(jobs)-sum(reused.values())}, capacity={backend.effective_batch_size}, "
              f"wall={wall:.2f}s, verify={verify:.2f}s", flush=True)
        return self.pending_info.pop(sample.name, {})
