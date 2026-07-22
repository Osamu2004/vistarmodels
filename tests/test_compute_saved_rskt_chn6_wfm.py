from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "compute_saved_rskt_chn6_wfm.py"
SPEC = importlib.util.spec_from_file_location("compute_saved_rskt_chn6_wfm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComputeSavedRsktChn6WfmTest(unittest.TestCase):
    def test_complete_saved_predictions_reproduce_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "CHN6-CUG" / "val"
            image_dir = data_root / "images"
            mask_dir = data_root / "gt"
            prediction_dir = root / "evaluation" / "pred_mask"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            prediction_dir.mkdir(parents=True)

            target = np.zeros((32, 32), dtype=np.uint8)
            target[8:24, 8:24] = 255
            prediction = np.zeros_like(target)
            prediction[8:24, 10:26] = 1
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
                image_dir / "sample.png"
            )
            Image.fromarray(target).save(mask_dir / "sample.png")
            Image.fromarray(prediction).save(
                prediction_dir / "000000_sample_pred_mask.png"
            )

            target_binary = (target != 0).astype(np.uint8)
            counts = MODULE._confusion(prediction, target_binary)
            metrics = MODULE._metrics(counts)
            metrics_path = root / "evaluation" / "metrics.json"
            metrics_path.write_text(
                json.dumps({**counts, **metrics}),
                encoding="utf-8",
            )

            argv = [
                str(MODULE_PATH),
                str(root / "evaluation"),
                "--data_root",
                str(data_root),
                "--workers",
                "1",
                "--expected_num_samples",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                MODULE.main()

            result = json.loads(
                (root / "evaluation" / "wfm_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            updated = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertTrue(result["complete_coverage"])
            self.assertEqual(result["existing_metrics_validation"]["tp"], "exact_match")
            self.assertAlmostEqual(result["road_iou"], metrics["road_iou"])
            self.assertAlmostEqual(updated["wfm_3px"], result["wfm_3px"])
            self.assertTrue((root / "evaluation" / "metrics_before_wfm.json").is_file())

    def test_existing_confusion_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            MODULE._validate_existing_metrics(
                {"tp": 9},
                {"tp": 8, "fp": 1, "fn": 2, "tn": 3},
                {"road_iou": 0.0, "background_iou": 0.0, "miou": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
