from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from flair_protocol import (  # noqa: E402
    FLAIR1_EXPECTED_SAMPLES,
    FLAIR1_TEST_DOMAIN_COUNTS,
    FLAIR_GSNET_CLASSES,
    FLAIR_GSNET_MODEL_CLASSES,
    FLAIR_IGNORE_VISUALIZATION_RGB,
    FLAIR_VISUAL_PALETTE_U8,
    IGNORE_INDEX,
    discover_flair1_test,
    flair_confusion_matrix,
    flair_metrics_from_confusion,
    load_flair_mask_array,
    load_flair_rgb_u8,
    map_flair_raw_mask,
    normalize_wsl_path,
)


class FlairProtocolTest(unittest.TestCase):
    def test_constants_match_gsnet_class_order(self) -> None:
        self.assertEqual(sum(FLAIR1_TEST_DOMAIN_COUNTS.values()), 15_700)
        self.assertEqual(FLAIR1_EXPECTED_SAMPLES, 15_700)
        self.assertEqual(len(FLAIR_GSNET_CLASSES), 12)
        self.assertEqual(
            FLAIR_GSNET_CLASSES[:4],
            ("building", "pervious surface", "impervious surface", "bare soil"),
        )
        self.assertEqual(FLAIR_GSNET_MODEL_CLASSES[1], "pervious-surface")
        self.assertEqual(FLAIR_GSNET_MODEL_CLASSES[2], "impervious-surface")
        self.assertEqual(len(set(FLAIR_VISUAL_PALETTE_U8)), 12)
        self.assertNotIn(FLAIR_IGNORE_VISUALIZATION_RGB, FLAIR_VISUAL_PALETTE_U8)

    def test_wsl_path_normalization(self) -> None:
        path = normalize_wsl_path(
            r"\\wsl.localhost\Ubuntu-22.04\root\data\FLAIR-1-2\data\flair#1-test"
        )
        self.assertEqual(
            path.as_posix(),
            "/root/data/FLAIR-1-2/data/flair#1-test",
        )

    def test_partial_discovery_pairs_by_domain_zone_and_numeric_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (
                root / "D012_2019" / "aerial" / "Z2_ABC" / "IMG_10.tif",
                root / "D012_2019" / "labels" / "Z2_ABC" / "MSK_10.tif",
                root / "D012_2019" / "aerial" / "Z2_ABC" / "IMG_2.tif",
                root / "D012_2019" / "labels" / "Z2_ABC" / "MSK_2.tif",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            records, audit = discover_flair1_test(root, strict=False)

            self.assertEqual([record.sample_id for record in records], ["2", "10"])
            self.assertEqual(
                records[0].output_name,
                "D012_2019_Z2_ABC_2",
            )
            self.assertEqual(audit["num_pairs"], 2)
            self.assertEqual(audit["domain_counts"]["D012_2019"], 2)
            self.assertEqual(audit["num_zones"], 1)
            with self.assertRaisesRegex(RuntimeError, "Invalid official FLAIR"):
                discover_flair1_test(root, strict=True)

    def test_discovery_rejects_unpaired_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = (
                Path(temporary)
                / "D012_2019"
                / "aerial"
                / "Z1_ABC"
                / "IMG_1.tif"
            )
            image.parent.mkdir(parents=True)
            image.touch()
            with self.assertRaisesRegex(RuntimeError, "images lack masks"):
                discover_flair1_test(temporary, strict=False)

    def test_raw_mapping_retains_only_one_through_twelve(self) -> None:
        raw = np.asarray([list(range(20)) + [255]], dtype=np.uint8)
        mapped = map_flair_raw_mask(raw)

        self.assertTrue(np.all(mapped[0, 1:13] == np.arange(12)))
        self.assertEqual(mapped[0, 0], IGNORE_INDEX)
        self.assertTrue(np.all(mapped[0, 13:] == IGNORE_INDEX))
        with self.assertRaisesRegex(ValueError, "Unexpected raw FLAIR"):
            map_flair_raw_mask(np.asarray([[20]], dtype=np.uint8))

    def test_tiff_loaders_select_rgb_and_map_mask(self) -> None:
        try:
            import tifffile
        except ImportError:
            self.skipTest("tifffile is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "IMG_1.tif"
            mask_path = root / "MSK_1.tif"
            image = np.zeros((5, 4, 6), dtype=np.uint8)
            image[0] = 10
            image[1] = 20
            image[2] = 30
            image[3] = 40
            image[4] = 50
            tifffile.imwrite(image_path, image, photometric="minisblack")
            tifffile.imwrite(
                mask_path,
                np.asarray([[1, 12, 13], [0, 19, 255]], dtype=np.uint8),
            )

            rgb = load_flair_rgb_u8(image_path)
            mapped = load_flair_mask_array(mask_path)

            self.assertEqual(rgb.shape, (4, 6, 3))
            self.assertTrue(np.all(rgb[0, 0] == np.asarray([10, 20, 30])))
            np.testing.assert_array_equal(
                mapped,
                np.asarray(
                    [[0, 11, IGNORE_INDEX], [IGNORE_INDEX] * 3],
                    dtype=np.int64,
                ),
            )

    def test_confusion_and_metrics_exclude_ignore(self) -> None:
        target = np.asarray([[0, 1, 255], [1, 2, 2]], dtype=np.int64)
        prediction = np.asarray([[0, 0, 11], [1, 2, 1]], dtype=np.int64)
        confusion = flair_confusion_matrix(prediction, target)
        metrics = flair_metrics_from_confusion(confusion)

        self.assertEqual(int(confusion.sum()), 5)
        self.assertEqual(confusion[0, 0], 1)
        self.assertEqual(confusion[1, 0], 1)
        self.assertEqual(confusion[1, 1], 1)
        self.assertEqual(confusion[2, 1], 1)
        self.assertEqual(confusion[2, 2], 1)
        self.assertAlmostEqual(metrics["iou_building"], 0.5)
        self.assertAlmostEqual(metrics["iou_pervious_surface"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["iou_impervious_surface"], 0.5)
        self.assertAlmostEqual(metrics["miou"], 4.0 / 9.0)
        self.assertAlmostEqual(metrics["macc"], (1.0 + 0.5 + 0.5) / 3.0)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.6)
        self.assertEqual(metrics["valid_pixels"], 5)


if __name__ == "__main__":
    unittest.main()
