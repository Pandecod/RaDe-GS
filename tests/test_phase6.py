import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_pipeline.commands import commands
from research_pipeline.config import DEFAULTS
from research_pipeline.geometry_evaluator import threshold_metrics


class Phase6(unittest.TestCase):
    def test_euclidean_chamfer_and_strict_threshold(self):
        values = threshold_metrics(np.array([0.1, 0.2]), np.array([0.0, 0.3]), 0.2)
        self.assertAlmostEqual(values["chamfer_distance"], 0.3)
        self.assertEqual(values["precision"], 0.5)
        self.assertEqual(values["recall"], 0.5)
        self.assertEqual(values["f1_score"], 0.5)
        no_threshold = threshold_metrics(np.array([1.0]), np.array([2.0]), None)
        self.assertEqual(no_threshold["chamfer_distance"], 3.0)
        self.assertIsNone(no_threshold["precision"])

    def test_geometry_command_uses_scene_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["geometry_evaluation"].update(enabled=True, gt_root=str(root / "gt"))
            command = commands(config, root / "data/togog", root / "out/togog")["geometry_evaluation"]
            self.assertIn(str(root / "gt/togog.ply"), command)
            self.assertIn(str(root / "out/togog/recon_post.ply"), command)
            self.assertNotIn("--distance-threshold", command)


if __name__ == "__main__":
    unittest.main()
