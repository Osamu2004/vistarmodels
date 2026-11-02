"""WSL-only synthetic parity check against the actual Alters source functions.

No weights/CUDA are needed. Set ALTERS_SOURCE_ROOT to the Vistar code checkout.
The local development audit only parses this file; it does not execute it.
"""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as functional

from tools import rsunivlm_paper_protocol as protocol
from tools import rsunivlm_paper_worker as worker


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ.get("ALTERS_SOURCE_ROOT", str(ROOT.parent / "vistar")))
FUNCTIONS = ("_ceil_to_multiple", "_pad_image_to_tile", "_tile_coords")
ENGINES = ("loveda", "uavid", "chn6_road", "second")


def reference_functions(engine):
    """Compile only the three audited geometry functions, not a model module."""
    path = REFERENCE_ROOT / f"eval_flux2_{engine}.py"
    if not path.is_file():
        pytest.fail(f"Missing Alters reference {path}; set ALTERS_SOURCE_ROOT")
    source = path.read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    audit = json.loads((ROOT / "baselines/rsunivlm/alters_tiling_audit_20261102.json").read_text())
    assert len(selected) == len(FUNCTIONS)
    for node in selected:
        # Source digest is portable across Python 3.10/3.12+ AST schema changes.
        digest = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
        assert digest == audit["reference_engines"][engine]["functions"][node.name]["source_sha256"], (
            f"Alters geometry changed: {path}:{node.name}; re-audit before comparison"
        )
    header = ast.parse("from __future__ import annotations").body
    scope = {"torch": torch}
    exec(compile(ast.Module(body=header + selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def image_pattern(width, height, seed):
    y, x = np.indices((height, width), dtype=np.int32)
    return np.stack(((x + 3*y + seed*17) % 251,
                     (2*x + y + seed*31) % 253,
                     (x + y + seed*43) % 255), axis=-1).astype(np.uint8)


def descriptor(tile, classes):
    # Deliberately rectangular: catch resizing the valid edge instead of the
    # full padded tile, and any change to nearest-neighbor pixel alignment.
    y, x = np.indices((23, 24))
    return ((x + 2*y + int(tile[0, 0, 0])) % len(classes)).astype(np.uint8)


def rle(grid, classes):
    return "\n".join(", ".join(f"{classes[value]}*1" for value in row) for row in grid)


@pytest.mark.parametrize("engine", ENGINES)
def test_alters_geometry_functions_are_identical(engine):
    actual = reference_functions(engine)
    base = reference_functions("loveda")
    image = torch.arange(3*513*777).reshape(3, 513, 777).float()
    assert torch.equal(actual["_pad_image_to_tile"](image, 512), base["_pad_image_to_tile"](image, 512))
    assert actual["_tile_coords"](1024, 1024, 512) == base["_tile_coords"](1024, 1024, 512)


@pytest.mark.parametrize("dataset,width,height", [
    ("loveda", 1, 1), ("loveda", 512, 512), ("loveda", 1024, 1024),
    ("loveda", 513, 777), ("loveda", 1920, 1080), ("second", 513, 777),
])
def test_queue_pixels_and_stitch_match_alters_source(tmp_path, monkeypatch, dataset, width, height):
    reference = reference_functions("second" if dataset == "second" else "loveda")
    classes = protocol.spec(dataset)["classes"]
    arrays = [image_pattern(width, height, seed) for seed in range(2 if dataset == "second" else 1)]
    image_paths, label_paths = [], []
    for index, array in enumerate(arrays):
        image = tmp_path / f"rgb{index}.png"
        label = tmp_path / f"label{index}.png"
        Image.fromarray(array).save(image)
        Image.fromarray(np.ones((height, width), np.uint8)).save(label)
        image_paths.append(str(image)); label_paths.append(str(label))
    sample = protocol.Sample("geometry", tuple(image_paths), tuple(label_paths))
    entry = dict(dataset=dataset, samples=[dict(sample=sample.serialize(), files=protocol.metadata(sample))])
    plan = dict(identity="synthetic-alters-geometry", max_samples=0,
                protocol=dict(batch_size=5, tile_size=512, format_retries=0, strict_population=False))

    padded = [reference["_pad_image_to_tile"](torch.from_numpy(array.copy()).permute(2, 0, 1).float()/255., 512)
              for array in arrays]
    coords = reference["_tile_coords"](*padded[0].shape[-2:], 512)
    directions = ("forward", "reverse") if dataset == "second" else ("mask",)
    expected_requests, expected_masks = [], {}
    for direction in directions:
        sources = padded[::-1] if direction == "reverse" else padded
        canvas = torch.empty(padded[0].shape[-2:], dtype=torch.uint8)
        for y, x in coords:
            tiles = [torch.round(source[:, y:y+512, x:x+512]*255).byte().permute(1, 2, 0).numpy() for source in sources]
            expected_requests.append(tiles)
            grid = descriptor(tiles[0], classes)
            expanded = functional.interpolate(torch.from_numpy(grid)[None, None].float(), size=(512, 512), mode="nearest")[0, 0].byte()
            canvas[y:y+512, x:x+512] = expanded
        expected_masks[direction] = canvas[:height, :width].numpy()

    seen = []
    def fake_generate(core, requests, prompt):
        answers = []
        for request in requests:
            tiles = [np.asarray(image).copy() for image in request]
            seen.append(tiles)
            answers.append(dict(response=rle(descriptor(tiles[0], classes), classes),
                generation=dict(inference_batch_size=len(requests), ended_with_eos=True, hit_token_limit=False)))
        return answers
    monkeypatch.setattr(worker, "generate_requests", fake_generate)
    monkeypatch.setattr(worker.Engine, "core", lambda self: None)
    monkeypatch.setattr(worker, "score_semantic_boundary_wfm", lambda *args, **kwargs: {})
    result = worker.Engine(plan, 0, 1).evaluate(entry, tmp_path / "output")
    assert result["complete"] and result["processed"] == 1
    assert len(seen) == len(expected_requests)
    for actual, expected in zip(seen, expected_requests):
        assert len(actual) == len(expected)
        for tile, ref_tile in zip(actual, expected):
            np.testing.assert_array_equal(tile, ref_tile)
    root = tmp_path / "output" / dataset / "shards/rank000/pred_mask"
    for direction, expected in expected_masks.items():
        with Image.open(root / f"geometry_{direction}.png") as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
