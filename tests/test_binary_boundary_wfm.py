from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from binary_boundary_wfm import (  # noqa: E402
    aggregate_binary_boundary_wfm,
    score_binary_boundary_wfm,
    score_semantic_boundary_wfm,
)


class BinaryBoundaryWfmTest(unittest.TestCase):
    @staticmethod
    def _rectangle_pair(shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
        target = np.zeros((32, 32), dtype=np.uint8)
        target[8:24, 8:24] = 1
        prediction = np.zeros_like(target)
        prediction[8:24, 8 + shift : 24 + shift] = 1
        return prediction, target

    def test_exact_prediction_scores_one_for_both_boundary_classes(self) -> None:
        prediction, target = self._rectangle_pair()
        result = score_binary_boundary_wfm(prediction, target)

        self.assertAlmostEqual(result["wfm_3px_nonedge"], 1.0)
        self.assertAlmostEqual(result["wfm_3px_edge"], 1.0)
        self.assertEqual(result["wfm_3px_gt_edge_pixels"], 256)
        self.assertEqual(result["wfm_3px_pred_edge_pixels"], 256)

    def test_shifted_rectangle_matches_flair_reference_values(self) -> None:
        prediction, target = self._rectangle_pair(shift=3)
        result = score_binary_boundary_wfm(prediction, target)

        self.assertAlmostEqual(
            result["wfm_3px_nonedge"],
            0.8894793365518833,
            places=12,
        )
        self.assertAlmostEqual(
            result["wfm_3px_edge"],
            0.6777887721484734,
            places=12,
        )

    def test_absent_gt_edge_class_is_not_added_to_class_mean(self) -> None:
        target = np.zeros((16, 16), dtype=np.uint8)
        result = score_binary_boundary_wfm(target.copy(), target)
        aggregate = aggregate_binary_boundary_wfm([result])

        self.assertAlmostEqual(result["wfm_3px_nonedge"], 1.0)
        self.assertIsNone(result["wfm_3px_edge"])
        self.assertEqual(aggregate["wfm_3px_nonedge_num_images"], 1)
        self.assertEqual(aggregate["wfm_3px_edge_num_images"], 0)
        self.assertAlmostEqual(aggregate["wfm_3px"], 1.0)

    def test_aggregate_averages_classes_before_taking_minimum(self) -> None:
        result = aggregate_binary_boundary_wfm(
            [
                {"wfm_3px_nonedge": 0.8, "wfm_3px_edge": 0.2},
                {"wfm_3px_nonedge": 0.4, "wfm_3px_edge": None},
            ]
        )

        self.assertAlmostEqual(result["wfm_3px_nonedge"], 0.6)
        self.assertAlmostEqual(result["wfm_3px_edge"], 0.2)
        self.assertAlmostEqual(result["wfm_3px"], 0.2)
        self.assertAlmostEqual(result["wfm_3px_percent"], 20.0)
        self.assertEqual(result["wfm_num_samples"], 2)
        self.assertEqual(
            result["wfm_protocol"]["per_class_aggregation"],
            "mean_over_gt_present_images",
        )

    def test_native_whole_image_score_is_not_a_tile_score_average(self) -> None:
        target = np.zeros((64, 64), dtype=np.uint8)
        target[16:48, 24:40] = 1
        prediction = np.zeros_like(target)
        prediction[16:48, 27:43] = 1

        whole_image = aggregate_binary_boundary_wfm(
            [score_binary_boundary_wfm(prediction, target)]
        )
        tile_rows = [
            score_binary_boundary_wfm(
                prediction[y : y + 32, x : x + 32],
                target[y : y + 32, x : x + 32],
            )
            for y in (0, 32)
            for x in (0, 32)
        ]
        tile_average = aggregate_binary_boundary_wfm(tile_rows)

        self.assertAlmostEqual(whole_image["wfm_3px"], 0.5742894810638524)
        self.assertNotAlmostEqual(
            whole_image["wfm_3px"],
            tile_average["wfm_3px"],
            places=6,
        )

    def test_rejects_nonbinary_or_mismatched_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary labels"):
            score_binary_boundary_wfm(
                np.asarray([[0, 2]], dtype=np.uint8),
                np.asarray([[0, 1]], dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            score_binary_boundary_wfm(
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((3, 4), dtype=np.uint8),
            )

    def test_semantic_scorer_matches_binary_scorer_without_ignore(self) -> None:
        prediction, target = self._rectangle_pair(shift=3)
        binary = score_binary_boundary_wfm(prediction, target)
        semantic = score_semantic_boundary_wfm(
            prediction,
            target,
            num_classes=12,
        )

        self.assertAlmostEqual(
            semantic["wfm_3px_nonedge"],
            binary["wfm_3px_nonedge"],
            places=14,
        )
        self.assertAlmostEqual(
            semantic["wfm_3px_edge"],
            binary["wfm_3px_edge"],
            places=14,
        )
        self.assertEqual(semantic["wfm_3px_gt_ignore_pixels"], 0)
        self.assertEqual(semantic["wfm_3px_ignore_excluded_pixels"], 0)

    def test_semantic_ignore_region_and_support_are_excluded(self) -> None:
        target = np.zeros((32, 32), dtype=np.int64)
        target[8:24, 8:24] = 5
        target[2:6, 2:6] = 255
        prediction = np.array(target, copy=True)
        prediction[2:6, 2:6] = 11

        result = score_semantic_boundary_wfm(
            prediction,
            target,
            ignore_index=255,
            ignore_margin=2,
            num_classes=12,
        )

        self.assertAlmostEqual(result["wfm_3px_nonedge"], 1.0)
        self.assertAlmostEqual(result["wfm_3px_edge"], 1.0)
        self.assertEqual(result["wfm_3px_gt_ignore_pixels"], 16)
        self.assertEqual(result["wfm_3px_ignore_excluded_pixels"], 64)
        self.assertEqual(result["wfm_3px_valid_pixels"], 32 * 32 - 64)
        self.assertEqual(result["wfm_3px_ignore_margin"], 2)

    def test_semantic_multiclass_shift_has_frozen_reference_values(self) -> None:
        target = np.zeros((40, 40), dtype=np.int64)
        target[6:22, 6:20] = 3
        target[18:35, 20:34] = 7
        target[2:8, 30:38] = 255
        prediction = np.zeros_like(target)
        prediction[6:22, 9:23] = 3
        prediction[16:33, 20:34] = 7
        prediction[target == 255] = 11

        result = score_semantic_boundary_wfm(
            prediction,
            target,
            ignore_index=255,
            ignore_margin=2,
            num_classes=12,
        )

        self.assertAlmostEqual(
            result["wfm_3px_nonedge"],
            0.8916501758541153,
            places=12,
        )
        self.assertAlmostEqual(
            result["wfm_3px_edge"],
            0.7688650347305421,
            places=12,
        )

    def test_semantic_scorer_rejects_ignore_prediction_on_valid_gt(self) -> None:
        target = np.zeros((8, 8), dtype=np.int64)
        prediction = np.array(target, copy=True)
        prediction[3, 3] = 255
        with self.assertRaisesRegex(ValueError, "valid GT pixels"):
            score_semantic_boundary_wfm(
                prediction,
                target,
                num_classes=12,
            )


if __name__ == "__main__":
    unittest.main()
