"""Batch independent LoveDA tile requests through the original GMoE model."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import baselines.rsunivlm.evaluate as legacy
from baselines.rsunivlm.protocols import (
    FORMAT_RETRY_POLICY, LOVEDA_PROMPT, RLE_GRID_POLICY, fingerprint,
    format_retry_prompt, loveda_spatial_settings, restore_grid,
)

ALIAS_EXTENSION = {"bareland": "barren", "bare land": "barren"}


def generate_batch(core, images, prompt, *, verify_processor=False):
    """One RGB image per sequence; all sequences have the identical prompt."""
    if not prompt.startswith("[SEG]"):
        raise ValueError("Dense inference requires a [SEG] instruction")
    if not images or len(images) > 4 or len({image.size for image in images}) != 1:
        raise ValueError("Use one to four equally-sized source tiles")
    start = time.monotonic()
    if len(images) == 1:
        response = core.generate(images, prompt)
        return [dict(response=response, generation=dict(copy.deepcopy(core.last_generation), inference_batch_size=1,
                     batch_elapsed_seconds=time.monotonic()-start))]
    if getattr(core.model.config, "image_aspect_ratio", "square") != "square":
        raise ValueError("Batch runtime is scoped to the recorded square-image checkpoint")
    conversation = copy.deepcopy(core.template)
    conversation.append_message(conversation.roles[0], "<image>\n" + prompt)
    conversation.append_message(conversation.roles[1], None)
    single = core.tokenizer_image_token(conversation.get_prompt(), core.tokenizer,
        core.image_token_index, return_tensors="pt")
    if single.numel() >= core.context_len:
        raise ValueError("Prompt exceeds the checkpoint context")
    tokens = single.unsqueeze(0).repeat(len(images), 1).to(core.device)
    pixels = core.process_images(images, core.processor, core.model.config)
    if verify_processor:
        for image, batch_pixels in zip(images, pixels):
            single_pixels = core.process_images([image], core.processor, core.model.config)[0]
            if not torch.equal(batch_pixels.cpu(), single_pixels.cpu()):
                raise ValueError("Batch and single-image visual preprocessing differ")
    pixels = [value.to(device=core.device, dtype=core.dtype) for value in pixels]
    if len(pixels) != len(images) or any(value.ndim != 3 for value in pixels) or len({tuple(v.shape) for v in pixels}) != 1:
        raise ValueError("Expected one matching CHW vision tensor per sequence")
    with torch.inference_mode():
        output = core.model.generate(tokens, images=pixels, image_sizes=[image.size for image in images],
            attention_mask=torch.ones_like(tokens, dtype=torch.bool), modalities=["image"]*len(images),
            granularity=2, do_sample=False, num_beams=1, max_new_tokens=core.max_new_tokens,
            temperature=1.0, top_p=1.0, top_k=50, use_cache=True,
            pad_token_id=core.tokenizer.eos_token_id)
    if output.ndim != 2 or output.shape[0] != len(images):
        raise ValueError("GMoE output batch differs from the input request batch")
    rows = output.detach().cpu().tolist()
    eos = core.model.generation_config.eos_token_id
    eos_ids = {value for value in (eos if isinstance(eos, (list, tuple)) else [eos]) if value is not None}
    eos_ids.add(core.tokenizer.eos_token_id)
    results = []
    elapsed = time.monotonic()-start
    for row in rows:
        first_eos = next((i for i, token in enumerate(row) if token in eos_ids), None)
        # Finished shorter sequences are padded with EOS while the longest
        # sequence continues. Never report those pads as generated content.
        length = len(row) if first_eos is None else first_eos+1
        response = core.tokenizer.batch_decode([row[:length]], skip_special_tokens=True)[0].strip()
        results.append(dict(response=response, generation=dict(sequence_tokens=length,
            batch_generated_steps=len(row), max_new_tokens=core.max_new_tokens,
            ended_with_eos=first_eos is not None, hit_token_limit=length >= core.max_new_tokens,
            attention_mask="all_valid_identical_prompt_lengths", decoding="greedy",
            inference_batch_size=len(images), batch_elapsed_seconds=elapsed)))
    return results


def accepted_attempt(record):
    """Pick the earliest valid stored answer under the explicit alias extension."""
    attempts = record.get("attempts") or [{key: record[key] for key in
        ("prompt", "response", "generation", "decode_error") if key in record}]
    for index, attempt in enumerate(attempts):
        try:
            grid = legacy.decode_saved_response(attempt, "loveda")
        except (ValueError, KeyError):
            continue
        selected = {key: copy.deepcopy(value) for key, value in attempt.items() if key != "decode_error"}
        value = {**record, **selected, "attempts": attempts, "grid_shape": list(grid.shape),
                 "grid_policy": RLE_GRID_POLICY, "selected_prompt_variant": index,
                 "format_retries_used": max(0, len(attempts)-1)}
        value.pop("decode_error", None)
        return grid, value
    return None, record


def generated_chunks(backend, jobs, prompt):
    offset = 0
    while offset < len(jobs):
        count = min(backend.effective_batch_size, len(jobs)-offset)
        group = jobs[offset:offset+count]
        backend.verification_context = [dict(raw_path=str(job["path"]), identity=job["identity"])
                                        for job in group]
        failed = False
        try:
            answers = backend.generate_batch([job["image"] for job in group], prompt)
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:
            if count == 1:
                raise RuntimeError("RSUniVLM still exceeds GPU memory at batch1")
            # Leave the except block before clearing failed CUDA intermediates.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            backend.effective_batch_size = max(1, count//2)
            backend.oom_splits += 1
            print(f"[RSUniVLM batch] OOM: {count} -> {backend.effective_batch_size}", flush=True)
            continue
        if len(answers) != len(group):
            raise ValueError("Missing batched model responses")
        yield group, answers
        offset += count


def query_tiles(backend, jobs, *, overwrite=False):
    grids, reused, histories = {}, {}, {}
    budget = backend.format_retries
    if budget not in (0, 1, 2):
        raise ValueError("Expected the recorded format budget")
    for job in jobs:
        path = job["path"]
        record = None if overwrite else legacy.read_json(path)
        if record and record.get("identity") == job["identity"]:
            grid, selected = accepted_attempt(record)
            if grid is not None:
                legacy.write_json(path, selected)
                grids[path], reused[path] = grid, True
                continue
            histories[path] = copy.deepcopy(record.get("attempts") or [{key: record[key] for key in
                ("prompt", "response", "generation", "decode_error") if key in record}])
        else:
            histories[path] = []
        if len(histories[path]) >= 1+budget:
            raise ValueError(f"Stored answers remain invalid after alias normalization; inspect {path}")
    for variant in range(1+budget):
        pending = [job for job in jobs if job["path"] not in grids and len(histories[job["path"]]) == variant]
        prompt = LOVEDA_PROMPT if variant == 0 else format_retry_prompt("loveda", variant)
        for group, answers in generated_chunks(backend, pending, prompt):
            for job, answer in zip(group, answers):
                path = job["path"]
                attempt = dict(prompt_variant=variant, prompt=prompt, **copy.deepcopy(answer))
                histories[path].append(attempt)
                record = dict(identity=job["identity"], initial_prompt=LOVEDA_PROMPT, **attempt,
                    attempts=histories[path], format_retry_policy=FORMAT_RETRY_POLICY,
                    format_retries_used=variant, spatial=job["spatial"])
                old = legacy.read_json(path)
                if old and old.get("identity") == job["identity"] and old.get("cache_import"):
                    record["cache_import"] = old["cache_import"]
                legacy.write_json(path, record)
                try:
                    grid = legacy.decode_saved_response(record, "loveda")
                except (ValueError, KeyError) as error:
                    attempt["decode_error"] = str(error)
                    legacy.write_json(path, dict(record, decode_error=str(error)))
                    print(f"[RSUniVLM batch format] {path.name}: {error}; attempt={variant+1}/{1+budget}", flush=True)
                else:
                    legacy.write_json(path, dict(record, grid_shape=list(grid.shape), grid_policy=RLE_GRID_POLICY))
                    grids[path], reused[path] = grid, False
    failed = [str(job["path"]) for job in jobs if job["path"] not in grids]
    if failed:
        raise ValueError(f"Format attempts exhausted; original answers preserved: {failed}")
    return grids, reused


def predict_tiled(backend, image, sample, raw_dir, identity, *, tile_size, overwrite):
    settings = loveda_spatial_settings(tile_size)
    if tile_size != 512 or image.mode != "RGB":
        raise ValueError("Batched LoveDA retains 512-pixel RGB source tiles")
    width, height = image.size
    jobs = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            h, w = min(tile_size, height-y), min(tile_size, width-x)
            tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            tile.paste(image.crop((x, y, x+w, y+h)), (0, 0))
            metadata = dict(source_box=[x,y,x+w,y+h], source_extent=[width,height], source_tile_size=tile_size,
                            padding=settings["rgb_padding"], valid_height=h, valid_width=w)
            jobs.append(dict(image=tile, path=raw_dir/f"{sample.name}_tile{tile_size}_y{y}_x{x}_seg.json",
                identity=fingerprint([identity,"loveda_source_tile",metadata]), spatial=metadata,
                x=x, y=y, h=h, w=w))
    start, before_verify = time.monotonic(), backend.verification_seconds
    grids, reused = query_tiles(backend, jobs, overwrite=overwrite)
    generation_seconds = time.monotonic()-start
    prediction = np.empty((height,width), np.uint8)
    audit = {key: 0 for key in legacy.FORMAT_AUDIT_FIELDS}
    shapes = []
    for job in jobs:
        grid = grids[job["path"]]
        restored = restore_grid(grid, (tile_size,tile_size))
        prediction[job["y"]:job["y"]+job["h"], job["x"]:job["x"]+job["w"]] = restored[:job["h"],:job["w"]]
        shapes.append(list(grid.shape))
        for key, value in legacy.query_format_audit(job["path"]).items():
            audit[key] += value
    return prediction, dict(queries=len(jobs), reused_queries=sum(reused.values()), num_tiles=len(jobs),
        source_tile_size=tile_size, descriptor_shapes=shapes, **audit,
        tile_generation_seconds=generation_seconds,
        batch_verification_seconds=backend.verification_seconds-before_verify,
        requested_tile_batch_size=backend.requested_batch_size, effective_tile_batch_size=backend.effective_batch_size)
