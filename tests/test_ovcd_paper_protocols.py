"""WSL CPU contract tests; authoring host only performs AST/shell checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from baselines.ovcd import evaluate as evaluation
from baselines.ovcd.backends import union_masks, official_support_pairs


class DummyBackend:
    def predict(self, first, second):
        # The condition order must survive edge padding and reconstruction.
        return [(first[...,0]>second[...,0]).astype(np.uint8)]


def test_official_support_accepts_unmatched_images_without_fabricating_masks(tmp_path):
    support = tmp_path / "LEVIR-CD"
    images = support / "building"
    masks = tmp_path / "LEVIR-CD_mask/building"
    images.mkdir(parents=True); masks.mkdir(parents=True)
    for name in ("one", "two", "three", "unmatched"):
        (images / (name + ".png")).write_bytes(b"image")
    for name in ("one", "two", "three"):
        (masks / (name + ".png")).write_bytes(b"mask")
    pairs = official_support_pairs(support, "building")
    assert {image.stem for image, mask in pairs} == {"one", "two", "three"}
    assert not (masks / "unmatched.png").exists()
    with pytest.raises(ValueError, match="three"):
        official_support_pairs(support, "background")


def test_tiles_preserve_native_extent_and_order_without_overlap():
    first=np.zeros((513,519,3),np.uint8)
    second=first.copy()
    first[490:,500:,0]=255
    expected=(first[...,0]>second[...,0]).astype(np.uint8)
    actual=evaluation.predict(DummyBackend(),[first,second],512)[0]
    np.testing.assert_array_equal(actual,expected)
    assert not evaluation.predict(DummyBackend(),[second,first],512)[0].any()


def test_no_proposals_are_valid_zero_predictions_and_union_does_not_overflow():
    assert not union_masks([], (3,4)).any()
    one=np.ones((3,4),bool)
    assert np.all(union_masks([one]*256,(3,4))==1)


def test_bad_proposal_extent_is_never_resized_silently():
    with pytest.raises(ValueError,match="extent"):
        union_masks([np.zeros((3,2))],(3,4))


def test_second_overlap_retains_both_class_ious_and_fixed_dense_priority(tmp_path):
    a=np.zeros((24,24),np.uint8);b=a.copy()
    a[5:14,7:17]=1;b[5:14,7:17]=5
    paths=[]
    for i,array in enumerate((a,b)):
        path=tmp_path/f"label{i}.png";Image.fromarray(array).save(path);paths.append(str(path))
    memberships=[((a==c)|(b==c)).astype(np.uint8) for c in range(1,7)]
    row=evaluation.score_masks("second",SimpleNamespace(name="case",labels=paths),memberships)
    assert row["class_counts"]["1"]["tp"]==row["class_counts"]["5"]["tp"]==90
    assert evaluation.dense(memberships)[8,9]==5
    assert row["wfm_3px_edge"]>0.999


def test_levir_matches_original_positive_gray_rule(tmp_path):
    raw=np.zeros((24,24),np.uint8);raw[6:16,6:16]=156
    path=tmp_path/"label.png";Image.fromarray(raw).save(path)
    row=evaluation.score_masks("levircd",SimpleNamespace(name="case",labels=[str(path)]),[(raw>0).astype(np.uint8)])
    assert row["class_counts"]["1"]["tp"]==100


def test_second_class_specific_official_settings_are_not_levir_defaults():
    path=Path(__file__).resolve().parents[1]/"configs/ovcd/dynamicearth_second_mci.json"
    settings=json.loads(path.read_text())["classes"]
    assert [v["class_id"] for v in settings]==list(range(1,7))
    assert settings[0]["sam"]["points_per_side"]==16
    assert settings[0]["change_threshold"]==135
    assert settings[2]["change_threshold"]==settings[3]["change_threshold"]==145
    assert all(v["is_instance_class"] is False for v in settings)


def test_second_sample_list_is_the_same_bytes_as_alters():
    assert evaluation.sha(evaluation.data.SPLIT_FILE)==evaluation.SPLIT_SHA
