"""WSL-only format recovery checks; no weights/CUDA and no GT-driven choice."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from baselines.rsunivlm.evaluate import query_grid, query_format_audit
from baselines.rsunivlm.protocols import LOVEDA_PROMPT, decode_rle, format_retry_prompt
from test_rsunivlm_protocols import FakeModel, args, evaluate_dataset, make_loveda, rle


FAILED_RESPONSE = "\n".join(["others *24"] * 23 + ["others *23, class *1"])


def backend(responses):
    value = FakeModel(responses)
    value.format_retries = 2
    value.last_generation = dict(sequence_tokens=124, max_new_tokens=8192,
        ended_with_eos=True, hit_token_limit=False)
    return value


def test_real_failure_is_still_invalid_and_unknown_label_is_not_background():
    with pytest.raises(ValueError, match="Unknown loveda label.*row 24"):
        decode_rle(FAILED_RESPONSE, "loveda", grid_size=None)
    assert "class*count" not in LOVEDA_PROMPT
    for attempt in (1, 2):
        assert "class*count" not in format_retry_prompt("loveda", attempt)
        assert "class *" not in format_retry_prompt("loveda", attempt)


def test_retry_keeps_first_answer_and_uses_first_valid_output(tmp_path):
    raw = tmp_path / "raw.json"
    images = [Image.new("RGB", (512, 512), "blue")]
    model = backend([FAILED_RESPONSE, rle("road*24"), rle("building*24")])
    grid, reused = query_grid(model, images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    np.testing.assert_array_equal(grid, np.full((24, 24), 2, np.uint8))
    assert not reused and len(model.calls) == 2
    assert model.calls[0][0] is model.calls[1][0] is images
    assert model.calls[0][1] == LOVEDA_PROMPT
    assert model.calls[1][1] == format_retry_prompt("loveda", 1)
    record = json.loads(raw.read_text())
    assert record["attempts"][0]["response"] == FAILED_RESPONSE
    assert "Unknown loveda label" in record["attempts"][0]["decode_error"]
    assert record["response"] == rle("road*24") and "decode_error" not in record
    assert record["format_retries_used"] == 1
    cached_model = backend([])
    actual, reused = query_grid(cached_model, images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    assert reused and not cached_model.calls
    np.testing.assert_array_equal(actual, grid)
    assert query_format_audit(raw) == dict(format_queries=1, format_attempts=2,
                                         format_retry_queries=1, format_retry_attempts=1)


def test_valid_first_answer_never_triggers_retries(tmp_path):
    model = backend([rle("building*24")])
    raw = tmp_path / "raw.json"
    query_grid(model, [Image.new("RGB", (512, 512))], LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    assert len(model.calls) == 1 and query_format_audit(raw)["format_retry_attempts"] == 0


def test_retries_are_bounded_and_cannot_create_a_valid_mask_from_unknown_names(tmp_path):
    model = backend([FAILED_RESPONSE] * 3)
    raw = tmp_path / "raw.json"
    images = [Image.new("RGB", (512, 512))]
    with pytest.raises(ValueError, match="Unknown loveda label"):
        query_grid(model, images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    record = json.loads(raw.read_text())
    assert len(model.calls) == len(record["attempts"]) == 3 and "grid_shape" not in record
    assert len({prompt for _, prompt in model.calls}) == 3
    stopped = backend([])
    with pytest.raises(ValueError, match="already failed"):
        query_grid(stopped, images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    assert not stopped.calls and json.loads(raw.read_text()) == record


def test_restart_after_interrupted_retry_keeps_the_failed_first_response(tmp_path):
    raw = tmp_path / "raw.json"
    images = [Image.new("RGB", (512, 512))]
    with pytest.raises(StopIteration):
        query_grid(backend([FAILED_RESPONSE]), images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    resumed = backend([rle("forest*24")])
    grid, _ = query_grid(resumed, images, LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    assert resumed.calls[0][1] == format_retry_prompt("loveda", 1)
    assert json.loads(raw.read_text())["attempts"][0]["response"] == FAILED_RESPONSE
    np.testing.assert_array_equal(grid, np.full((24, 24), 5, np.uint8))


def test_truncated_text_cannot_win_just_because_its_rows_parse(tmp_path):
    class Truncated(FakeModel):
        format_retries = 2
        def generate(self, images, prompt):
            result = super().generate(images, prompt)
            self.last_generation = dict(hit_token_limit=len(self.calls) == 1,
                                        ended_with_eos=len(self.calls) != 1)
            return result
    model = Truncated([rle("road*24"), rle("water*24")])
    raw = tmp_path / "raw.json"
    grid, _ = query_grid(model, [Image.new("RGB", (512, 512))], LOVEDA_PROMPT, "loveda", raw, "run", overwrite=False)
    np.testing.assert_array_equal(grid, np.full((24, 24), 3, np.uint8))
    assert "MAX_NEW_TOKENS" in json.loads(raw.read_text())["attempts"][0]["decode_error"]


def test_dataset_reports_retries_and_new_policy_cannot_mix_with_old_outputs(tmp_path):
    data, output = tmp_path / "data", tmp_path / "output"
    make_loveda(data)
    options = args(data, output, save_images=False, format_retries=2)
    model = backend([FAILED_RESPONSE, rle("building*24")])
    result = evaluate_dataset(options, "loveda", model, {})
    assert result["complete"] and result["format_audit"]["format_retry_queries"] == 1
    assert result["format_audit"]["format_attempts"] == 2
    resumed = evaluate_dataset(options, "loveda", backend([]), {})
    assert resumed["format_audit"] == result["format_audit"]
    options.format_retries = 0
    with pytest.raises(RuntimeError, match="different model/data/prompt/source"):
        evaluate_dataset(options, "loveda", backend([]), {})
