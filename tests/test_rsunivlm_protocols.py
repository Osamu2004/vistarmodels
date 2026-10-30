"""Run in WSL: CPU protocol tests use synthetic files and a fake text model.

These checks deliberately do not download or instantiate RSUniVLM weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from baselines.rsunivlm.evaluate import (
    cached_prediction, evaluate_dataset, predict_sample, score,
)
from baselines.rsunivlm.model import keep_pixel_expert_during_generation
from baselines.rsunivlm.protocols import (
    CD_PROMPT, LOVEDA_PROMPT, Sample, decode_rle, discover, fuse_change, restore_grid,
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
                  overwrite=False, save_images=True, boundary_wfm=False)
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
