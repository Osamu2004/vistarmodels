"""Validate v4 provenance and import raw model answers into a new batch run.

Predictions and scores are never copied: masks are reconstructed from the first
valid raw attempt under the explicit synonym extension, then scored normally.
"""
import copy
import fcntl
import hashlib
from pathlib import Path

from PIL import Image

import baselines.rsunivlm.evaluate as legacy
from baselines.rsunivlm import protocols as protocol


def changed_fields(stored, requested, prefix="protocol"):
    """Name differing fields without dumping model metadata or large lists."""
    if isinstance(stored, dict) and isinstance(requested, dict):
        fields = []
        for key in sorted(stored.keys() | requested.keys()):
            name = f"{prefix}.{key}"
            if key not in stored or key not in requested:
                fields.append(name)
            else:
                fields.extend(changed_fields(stored[key], requested[key], name))
        return fields
    return [] if stored == requested else [prefix]


def output_conflicts(root, requested):
    """Return conflicts; never infer a protocol for unidentified artifacts."""
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"Output path is not a directory: {root}")
    config_path = root / "run_config.json"
    config = legacy.read_json(config_path)
    if config is not None:
        fields = changed_fields(config.get("protocol"), requested)
        if config.get("identity") != protocol.fingerprint(requested):
            fields.insert(0, "identity")
        return fields
    if config_path.exists():
        return ["run_config.json unreadable"]
    artifacts = ("raw_text/*.json", "pred_mask/*.png", "receipts/*.json",
                 "metrics.json", "input_manifest.json", "cache_import.json")
    if any(any(root.glob(pattern)) for pattern in artifacts):
        return ["run_config.json missing for existing artifacts"]
    return []


def resolve_output_root(requested_root, requested_protocol):
    """Keep incompatible results intact and choose a stable protocol sibling."""
    requested_root = Path(requested_root).resolve()
    conflicts = output_conflicts(requested_root, requested_protocol)
    if not conflicts:
        return requested_root, None
    identity = protocol.fingerprint(requested_protocol)
    sibling = requested_root.with_name(f"{requested_root.name}_p{identity[:12]}")
    sibling_conflicts = output_conflicts(sibling, requested_protocol)
    if sibling_conflicts:
        raise ValueError(f"Protocol-specific output also conflicts: {sibling}: {sibling_conflicts}; "
                         "choose another OUTPUT_DIR; all existing files preserved")
    previous = legacy.read_json(requested_root / "run_config.json") or {}
    receipt = dict(schema="rsunivlm_output_resolution_v1", requested_root=str(requested_root),
        actual_root=str(sibling), previous_identity=previous.get("identity"),
        requested_identity=identity, differing_fields=conflicts,
        policy="preserve incompatible output; use independent protocol-specific directory",
        copied_scores=False, copied_prediction_masks=False)
    print(f"[RSUniVLM output] Preserving {requested_root}; differing fields: {', '.join(conflicts)}", flush=True)
    print(f"[RSUniVLM output] Actual OUTPUT_DIR={sibling}", flush=True)
    return sibling, receipt


def build_protocol(args, base):
    root = args.data_root or args.loveda_root
    samples, audit = protocol.discover("loveda", legacy.normalize_path(root), "val")
    expected = protocol.expected_samples("loveda", "val")
    if len(samples) != expected and not args.allow_nonstandard_population:
        raise ValueError(f"Expected {expected} LoveDA images, got {len(samples)}")
    manifest = [{"name": sample.name, "files": protocol.input_identity(sample)} for sample in samples]
    spatial = protocol.loveda_spatial_settings(args.loveda_tile_size)
    value = {**base, "dataset": "loveda", "split": "val",
        "dataset_population": protocol.fingerprint(manifest), "discovered_samples": len(samples),
        "prompt": protocol.LOVEDA_PROMPT, "format_retries": getattr(args, "format_retries", 0),
        "format_retry_policy": protocol.FORMAT_RETRY_POLICY, "cd_direction": None,
        "nominal_grid_size": protocol.GRID_SIZE, "rle_grid_policy": protocol.RLE_GRID_POLICY,
        "max_descriptor_side": protocol.MAX_DESCRIPTOR_SIDE, "restoration": spatial["restoration"],
        "nearest_implementation": "opencv_INTER_NEAREST", "classes": list(protocol.LOVEDA_CLASSES),
        "label_protocol": "raw0/255 ignored; raw1..7 -> 0..6", "loveda_spatial": spatial}
    return value, manifest, samples, audit


def import_raw_answers(source_root, output_root, args, old_base, new_base):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if source_root == output_root:
        raise ValueError("Use a separate batch output directory; preserve the original v4 run")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    old_protocol, _, samples, _ = build_protocol(args, old_base)
    new_protocol, manifest, _, audit = build_protocol(args, new_base)
    old_id, new_id = protocol.fingerprint(old_protocol), protocol.fingerprint(new_protocol)
    previous = legacy.read_json(source_root / "run_config.json")
    if not previous or previous.get("identity") != old_id or previous.get("protocol") != old_protocol:
        raise ValueError("v4 source checkpoint/data/prompt/code differs; refusing cache import")
    source_manifest = legacy.read_json(source_root / "input_manifest.json")
    if not source_manifest or source_manifest.get("samples") != manifest:
        raise ValueError("v4 input metadata differs from the current data population")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = legacy.read_json(output_root / "run_config.json")
    if destination and (destination.get("identity") != new_id or destination.get("protocol") != new_protocol):
        fields = changed_fields(destination.get("protocol"), new_protocol)
        raise ValueError(f"Batch output already belongs to a different protocol: {output_root}; "
                         f"differing fields: {fields or ['identity']}")
    raw_dir = output_root / "raw_text"
    raw_dir.mkdir(exist_ok=True)
    legacy.write_json(output_root / "run_config.json", dict(identity=new_id, protocol=new_protocol, audit=audit,
        metric_settings=dict(boundary_wfm=args.boundary_wfm, wfm_ignore_margin=2)))
    legacy.write_json(output_root / "input_manifest.json", dict(samples=manifest))
    count, skipped = 0, 0
    # The original evaluator holds an exclusive writer lock while active.
    # Do not import half-written answers from a still-running v4 evaluation.
    with (source_root / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        for sample in samples:
            identity = protocol.input_identity(sample)
            old_item = protocol.fingerprint([old_id, identity])
            new_item = protocol.fingerprint([new_id, identity])
            with Image.open(sample.images[0]) as image:
                width, height = image.size
            size = args.loveda_tile_size
            for y in range(0, height, size):
                for x in range(0, width, size):
                    name = f"{sample.name}_tile{size}_y{y}_x{x}_seg.json"
                    path = source_root / "raw_text" / name
                    if not path.is_file():
                        continue
                    h, w = min(size,height-y), min(size,width-x)
                    spatial = dict(source_box=[x,y,x+w,y+h], source_extent=[width,height], source_tile_size=size,
                        padding="black_right_bottom", valid_height=h, valid_width=w)
                    record = legacy.read_json(path)
                    old_tile = protocol.fingerprint([old_item,"loveda_source_tile",spatial])
                    new_tile = protocol.fingerprint([new_item,"loveda_source_tile",spatial])
                    if (not record or record.get("identity") != old_tile or record.get("spatial") != spatial
                            or record.get("initial_prompt", record.get("prompt")) != protocol.LOVEDA_PROMPT):
                        raise ValueError(f"v4 raw tile identity or geometry mismatch: {path}")
                    attempts = record.get("attempts") or [record]
                    if len(attempts) > 1+args.format_retries:
                        raise ValueError(f"v4 raw answer exceeds its recorded retry budget: {path}")
                    for index, attempt in enumerate(attempts):
                        expected_prompt = protocol.LOVEDA_PROMPT if index == 0 else protocol.format_retry_prompt("loveda", index)
                        if attempt.get("prompt") != expected_prompt:
                            raise ValueError(f"v4 attempt prompt mismatch: {path}")
                    target = raw_dir / name
                    if target.exists():
                        current = legacy.read_json(target)
                        if not current or current.get("identity") != new_tile:
                            raise ValueError(f"Existing batch tile has another identity: {target}")
                        skipped += 1
                        continue
                    imported = copy.deepcopy(record)
                    imported["identity"] = new_tile
                    imported["cache_import"] = dict(source_path=str(path), source_identity=old_tile,
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        old_protocol_identity=old_id, legacy_inference_batch_size=1,
                        original_decode_error=record.get("decode_error"))
                    legacy.write_json(target, imported)
                    count += 1
    receipt = dict(schema="rsunivlm_v4_raw_import_v1", source=str(source_root), destination=str(output_root),
        source_protocol_identity=old_id, destination_protocol_identity=new_id, imported_queries=count,
        kept_existing_queries=skipped, copied_scores=False, copied_prediction_masks=False,
        policy="redecode earliest valid stored attempt with explicit aliases; rescore against unchanged GT")
    legacy.write_json(output_root / "cache_import.json", receipt)
    print(f"[RSUniVLM import] {count} raw tile answers imported; {skipped} existing; no copied scores", flush=True)
    return receipt
