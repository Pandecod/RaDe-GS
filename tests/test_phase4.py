from pathlib import Path
import tempfile
import unittest
from research_pipeline.image_evaluator import evaluation_pairs, render_statistics

class Phase4(unittest.TestCase):
    def test_empty_render_has_null_rates_and_cannot_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(render_statistics(Path(tmp), 1), {"image_count": 0, "seconds_per_image": None, "fps": None})
            with self.assertRaises(ValueError):
                evaluation_pairs(Path(tmp))
    def test_evaluation_split_pairing_and_numeric_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, iteration in (("train", 9), ("train", 10), ("test", 10)):
                for kind in ("renders", "gt"):
                    folder = root / split / f"ours_{iteration}" / kind
                    folder.mkdir(parents=True)
                    (folder / "00000.png").write_bytes(b"fixture")
            split, folder, pairs = evaluation_pairs(root)
            self.assertEqual(split, "test")
            self.assertEqual(folder.name, "ours_10")
            self.assertEqual(render_statistics(root, 2)["image_count"], 2)
            self.assertEqual(render_statistics(root, 2)["fps"], 1)
            pairs[0][1].unlink()
            with self.assertRaises(ValueError):
                evaluation_pairs(root)

if __name__ == "__main__":
    unittest.main()
