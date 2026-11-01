"""Run in WSL: CPU protocol tests use synthetic files and a fake text model.

These checks deliberately do not download or instantiate RSUniVLM weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from baselines.rsunivlm.evaluate import (
    cached_prediction, evaluate_dataset, parse_args, predict_sample, score,
)
from baselines.rsunivlm.model import keep_pixel_expert_during_generation
from baselines.rsunivlm.protocols import (
    CD_PROMPT, LOVEDA_PROMPT, Sample, decode_rle, discover, fuse_change, restore_grid, source_revision,
)


def rle(row: str) -> str:
    return "\n".join([row] * 24)


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[list[Image.Image], str]] = []

    def generate(self, images: list[Image.Image], prompt: str) -> str:
        self.calls.append((images, prompt))
        return next(self.responses)


def args(root: Path, output: Path, **overrides: object) -> argparse.Namespace:
    values = dict(dataset="loveda", data_root=str(root), loveda_root=str(root), levircd_root=str(root),
                  split="test", output_dir=str(output), max_samples=0,
                  allow_nonstandard_population=True, cd_direction="symmetric_union",
                  overwrite=False, save_images=True, boundary_wfm=False,
                  loveda_tile_size=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def make_loveda(root: Path) -> None:
    folder = root / "Val" / "Urban"
    for name in ("images_png", "masks_png"):
        (folder / name).mkdir(parents=True)
    Image.fromarray(np.full((48, 72, 3), 123, dtype=np.uint8)).save(folder / "images_png" / "1.png")
    # Raw2 -> building; raw0 must not affect any metric denominator.
    target = np.full((48, 72), 2, dtype=np.uint8)
    target[:, :12] = 0
    Image.fromarray(target).save(folder / "masks_png" / "1.png")


def make_levir(root: Path) -> None:
    for folder, value in (("A", 20), ("B", 180), ("label", 255)):
        directory = root / "test" / folder
        directory.mkdir(parents=True)
        shape = (48, 72) if folder == "label" else (48, 72, 3)
        Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(directory / "case.png")


@pytest.fixture
def source_checkout(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    root = tmp_path / "upstream"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("VALUE = 1\n")
    (root / "tracked.pyc").write_bytes(b"original bytecode fixture")
    subprocess.run(["git", "-C", str(root), "add", "source.py", "tracked.pyc"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Protocol Test",
                    "-c", "user.email=protocol-test@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr("baselines.rsunivlm.protocols.UPSTREAM_COMMIT", revision)
    return root, revision


@pytest.mark.parametrize("filename", ["source.py", "tracked.pyc"])
def test_pinned_source_allows_modified_tracked_files(source_checkout, filename: str) -> None:
    root, revision = source_checkout
    (root / filename).write_bytes(b"local change\n")
    assert subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    assert source_revision(root) == revision


def test_source_revision_still_checks_pinned_commit(source_checkout, monkeypatch) -> None:
    root, _ = source_checkout
    monkeypatch.setattr("baselines.rsunivlm.protocols.UPSTREAM_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="!= pinned"):
        source_revision(root)


def test_seven_class_decode_and_native_extent() -> None:
    grid = decode_rle(rle("background*3,building*3, road*3, water*3, barren*3, forest*3, agriculture*6"), "loveda")
    assert set(np.unique(grid)) == set(range(7))
    restored = restore_grid(grid, (31, 67))
    assert restored.shape == (31, 67)
    assert set(np.unique(restored)) == set(range(7))
    assert restored[0, 0] == 0 and restored[-1, -1] == 6


@pytest.mark.parametrize("text", [
    rle("building*23"), rle("building*25"), rle("building*0"),
    rle("building*999999999999"), rle("building*-1"),
    rle("hallucinated road*24"), "No changes detected.",
    "\n".join(["building*24"] * 23), rle("building*12; others*12"),
])
def test_malformed_responses_are_not_silently_background(text: str) -> None:
    with pytest.raises(ValueError):
        decode_rle(text, "levircd")


def test_quoted_labels_and_code_fence() -> None:
    grid = decode_rle("```text\n" + rle("others*12, agricultural land*12") + "\n```", "loveda")
    assert np.all(grid[:, :12] == 0) and np.all(grid[:, 12:] == 6)


def test_symmetric_cd_uses_both_orders_and_reuses_raw_text(tmp_path: Path) -> None:
    a, b = Image.new("RGB", (72, 48), "black"), Image.new("RGB", (72, 48), "white")
    sample = Sample("case", (tmp_path / "A", tmp_path / "B"), tmp_path / "label")
    backend = FakeModel([rle("building*12, others*12"), rle("others*12, building*12")])
    result, details = predict_sample(backend, [a, b], sample, "levircd", tmp_path, "run",
                                     cd_direction="symmetric_union", overwrite=False)
    assert np.all(result == 1)
    assert details == {"queries": 2, "reused_queries": 0}
    assert backend.calls[0] == ([a, b], CD_PROMPT)
    assert backend.calls[1] == ([b, a], CD_PROMPT)
    cached, details = predict_sample(FakeModel([]), [a, b], sample, "levircd", tmp_path, "run",
                                     cd_direction="symmetric_union", overwrite=False)
    np.testing.assert_array_equal(cached, result)
    assert details["reused_queries"] == 2


def test_change_union_cannot_introduce_semantic_class_ids() -> None:
    with pytest.raises(ValueError):
        fuse_change([np.full((24, 24), 2)])


def test_pixel_expert_survives_hf_preparation_for_each_decode_step() -> None:
    class FakeHF:
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
            # Qwen preparation may drop an extra model kwarg.
            return {"input_ids": input_ids, "past_key_values": past_key_values}
    model = FakeHF()
    keep_pixel_expert_during_generation(model)
    for token, cache in ((1, None), (2, "kv")):
        values = model.prepare_inputs_for_generation(token, past_key_values=cache, granularity=0)
        assert values == {"input_ids": token, "past_key_values": cache, "granularity": 2}


def test_loveda_mapping_and_ignored_pixels_do_not_inflate_accuracy() -> None:
    target = np.asarray([[255, 1], [1, 1]])
    prediction = np.asarray([[0, 1], [1, 0]])
    matrix, metrics = score(prediction, target, "loveda", boundary_wfm=False)
    assert matrix.sum() == 3 and matrix[1, 1] == 2 and matrix[1, 0] == 1
    assert metrics["iou_building"] == pytest.approx(2 / 3)
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 3)


def test_loveda_fake_inference_has_no_gt_input_and_is_restartable(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    backend = FakeModel([rle("building*24")])
    options = args(data, output)
    result = evaluate_dataset(options, "loveda", backend, {"checkpoint": "fake"})
    assert result["complete"] and not result["is_full_benchmark"]
    assert result["processed_samples"] == 1
    assert result["scores"]["iou_building"] == 1
    assert len(backend.calls) == 1
    images, prompt = backend.calls[0]
    assert len(images) == 1 and images[0].size == (72, 48)
    assert prompt == LOVEDA_PROMPT and np.all(np.asarray(images[0]) == 123)
    path = output / "pred_mask" / "Val_Urban_1_pred_mask.png"
    assert np.all(np.asarray(Image.open(path)) == 1)
    evaluate_dataset(options, "loveda", FakeModel([]), {"checkpoint": "fake"})
    row = json.loads((output / "predictions.jsonl").read_text())
    assert row["reused_prediction"]
    # Changed checkpoint/source identity must not reuse this prediction.
    with pytest.raises(RuntimeError, match="different model/data/prompt/source"):
        evaluate_dataset(options, "loveda", FakeModel([]), {"checkpoint": "different"})


def test_loveda_tile512_preserves_quadrants_and_scores_native_extent(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    folder = data / "Val" / "Urban"
    image = np.empty((1024, 1024, 3), dtype=np.uint8)
    target = np.empty((1024, 1024), dtype=np.uint8)
    for class_id, (y, x) in enumerate(((0, 0), (0, 512), (512, 0), (512, 512))):
        image[y:y+512, x:x+512] = 40 * (class_id + 1)
        target[y:y+512, x:x+512] = class_id + 1
    Image.fromarray(image).save(folder / "images_png" / "1.png")
    Image.fromarray(target).save(folder / "masks_png" / "1.png")
    backend = FakeModel([rle(f"{name}*24") for name in ("background", "building", "road", "water")])
    options = args(data, output, loveda_tile_size=512)
    result = evaluate_dataset(options, "loveda", backend, {})
    assert result["complete"] and result["processed_samples"] == 1
    assert result["scores"]["pixel_accuracy"] == 1
    assert len(backend.calls) == 4
    for index, (images, prompt) in enumerate(backend.calls):
        assert len(images) == 1 and images[0].mode == "RGB" and images[0].size == (512, 512)
        assert np.all(np.asarray(images[0]) == 40 * (index + 1))
        assert prompt == LOVEDA_PROMPT
    saved = np.asarray(Image.open(output / "pred_mask" / "Val_Urban_1_pred_mask.png"))
    np.testing.assert_array_equal(saved, target - 1)
    row = json.loads((output / "predictions.jsonl").read_text())
    assert row["height"] == row["width"] == 1024
    assert row["num_tiles"] == row["queries"] == 4 and row["source_tile_size"] == 512
    spatial = result["loveda_spatial"]
    assert spatial["source_stride"] == 512 and spatial["vision_input_size"] == [384, 384]
    assert spatial["mask_grid_size"] == 24
    evaluate_dataset(options, "loveda", FakeModel([]), {})
    assert json.loads((output / "predictions.jsonl").read_text())["reused_prediction"]


def test_loveda_border_tiles_pad_black_then_restore_before_trimming(tmp_path: Path) -> None:
    # Both border strips are smaller than half a tile. Resizing a grid directly
    # to the valid strip would incorrectly insert its right/bottom half there.
    source = Image.new("RGB", (600, 530), (123, 123, 123))
    sample = Sample("border", (tmp_path / "rgb.png",), tmp_path / "gt.png")
    response = "\n".join(["building*12, road*12"] * 12 + ["water*12, forest*12"] * 12)
    backend = FakeModel([response] * 4)
    prediction, info = predict_sample(backend, [source], sample, "loveda", tmp_path, "run",
                                     cd_direction="symmetric_union", overwrite=False, loveda_tile_size=512)
    assert prediction.shape == (530, 600) and info["num_tiles"] == 4
    assert np.all(prediction[512:, 512:] == 1)
    assert np.all(prediction[:256, 512:] == 1) and np.all(prediction[256:512, 512:] == 3)
    assert np.all(prediction[512:, :256] == 1) and np.all(prediction[512:, 256:512] == 2)
    for (images, _), (valid_h, valid_w) in zip(backend.calls, ((512, 512), (512, 88), (18, 512), (18, 88))):
        tile = np.asarray(images[0])
        assert tile.shape == (512, 512, 3)
        assert np.all(tile[:valid_h, :valid_w] == 123)
        assert np.all(tile[valid_h:] == 0) and np.all(tile[:, valid_w:] == 0)
    raw = json.loads((tmp_path / "border_tile512_y512_x512_seg.json").read_text())
    assert raw["spatial"]["source_box"] == [512, 512, 600, 530]
    assert raw["spatial"]["source_extent"] == [600, 530]


def test_loveda_failed_tile_retries_only_missing_or_invalid_responses(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    folder = data / "Val" / "Urban"
    Image.new("RGB", (1024, 1024), (123, 123, 123)).save(folder / "images_png" / "1.png")
    Image.new("L", (1024, 1024), 2).save(folder / "masks_png" / "1.png")
    options = args(data, output, loveda_tile_size=512)
    with pytest.raises(ValueError):
        evaluate_dataset(options, "loveda", FakeModel([rle("building*24"), "bad tile"]), {})
    assert not json.loads((output / "metrics.json").read_text())["complete"]
    assert not list((output / "pred_mask").glob("*.png"))
    failed = output / "raw_text" / "Val_Urban_1_tile512_y0_x512_seg.json"
    assert json.loads(failed.read_text())["response"] == "bad tile"
    retry = FakeModel([rle("building*24")] * 3)
    result = evaluate_dataset(options, "loveda", retry, {})
    assert result["complete"] and result["scores"]["pixel_accuracy"] == 1
    assert len(retry.calls) == 3
    row = json.loads((output / "predictions.jsonl").read_text())
    assert row["queries"] == 4 and row["reused_queries"] == 1


def test_loveda_whole_image_cache_cannot_be_used_for_tiled_evaluation(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    evaluate_dataset(args(data, output), "loveda", FakeModel([rle("building*24")]), {})
    with pytest.raises(RuntimeError, match="different model/data/prompt/source"):
        evaluate_dataset(args(data, output, loveda_tile_size=512), "loveda", FakeModel([]), {})


@pytest.mark.parametrize("tile_size", [None, "0", "512"])
def test_loveda_cli_default_and_legacy_mode(monkeypatch, tile_size: str | None) -> None:
    argv = ["evaluate.py", "--dataset", "loveda", "--output_dir", "/tmp/rsunivlm"]
    if tile_size is not None:
        argv += ["--loveda_tile_size", tile_size]
    monkeypatch.setattr("sys.argv", argv)
    assert parse_args().loveda_tile_size == (512 if tile_size is None else int(tile_size))


def test_loveda_cli_rejects_unapproved_tile_size(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--dataset", "loveda", "--output_dir", "/tmp/rsunivlm",
                                     "--loveda_tile_size", "256"])
    with pytest.raises(SystemExit):
        parse_args()


def test_failure_preserves_text_but_never_publishes_a_complete_score(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    with pytest.raises(ValueError):
        evaluate_dataset(args(data, output), "loveda", FakeModel(["not a mask"]), {})
    assert not json.loads((output / "metrics.json").read_text())["complete"]
    raw = json.loads(next((output / "raw_text").glob("*.json")).read_text())
    assert raw["response"] == "not a mask"
    assert not list((output / "pred_mask").glob("*.png"))


def test_levir_artifact_format_and_restart(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "output"
    make_levir(data)
    options = args(data, output, dataset="levircd")
    result = evaluate_dataset(options, "levircd", FakeModel([rle("building*24"), rle("others*24")]), {})
    assert result["scores"]["f1_score_change"] == pytest.approx(1)
    assert np.all(np.asarray(Image.open(output / "pred_mask" / "case_pred_mask.png")) == 255)
    evaluate_dataset(options, "levircd", FakeModel([]), {})
    assert json.loads((output / "predictions.jsonl").read_text())["reused_prediction"]


def test_missing_levir_partner_is_not_silently_dropped(tmp_path: Path) -> None:
    make_levir(tmp_path)
    Image.new("RGB", (72, 48)).save(tmp_path / "test" / "A" / "extra.png")
    with pytest.raises(ValueError, match="stem sets differ"):
        discover("levircd", tmp_path, "test")


def test_truncated_cache_and_replaced_valid_png_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pred.png"
    prediction = np.zeros((24, 24), dtype=np.uint8)
    receipt = {"identity": "run", "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}
    Image.fromarray(prediction).save(path)
    assert cached_prediction(path, receipt, identity="run", shape=(24, 24), num_classes=7) is not None
    Image.fromarray(prediction + 1).save(path)
    assert cached_prediction(path, receipt, identity="run", shape=(24, 24), num_classes=7) is None
    path.write_bytes(b"truncated png")
    assert cached_prediction(path, receipt, identity="run", shape=(24, 24), num_classes=7) is None


def test_development_subset_cannot_be_called_full_benchmark(tmp_path: Path) -> None:
    make_loveda(tmp_path / "data")
    with pytest.raises(ValueError, match="expected 1669"):
        evaluate_dataset(args(tmp_path / "data", tmp_path / "output", allow_nonstandard_population=False),
                         "loveda", FakeModel([]), {})
