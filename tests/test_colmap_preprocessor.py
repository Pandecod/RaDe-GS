from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.colmap_preprocessor import preprocess
from research_pipeline.scene_discovery import Scene, validation_errors
from test_research_pipeline import make_raw_scene


class ColmapPreprocessor(unittest.TestCase):
    def test_publishes_only_complete_output_and_then_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = make_raw_scene(Path(tmp), "togog")
            calls = []

            def fake_run(command):
                calls.append(command[1])
                if command[1] == "mapper":
                    sparse = Path(command[command.index("--output_path") + 1]) / "0"
                    sparse.mkdir(parents=True)
                    for name in ("cameras.bin", "images.bin", "points3D.bin"):
                        (sparse / name).write_bytes(b"fixture")
                elif command[1] == "image_undistorter":
                    output = Path(command[command.index("--output_path") + 1])
                    (output / "images").mkdir(parents=True)
                    (output / "images/00001.jpg").write_bytes(b"fixture")
                    (output / "sparse").mkdir()
                    for name in ("cameras.bin", "images.bin", "points3D.bin"):
                        (output / "sparse" / name).write_bytes(b"fixture")

            with patch("research_pipeline.colmap_preprocessor.shutil.which",
                       return_value="colmap"):
                self.assertEqual(preprocess(scene_path, run=fake_run), "completed")
                self.assertFalse(validation_errors(Scene("togog", scene_path)))
                self.assertTrue((scene_path / "input/00001.jpg").is_file())
                self.assertEqual(preprocess(scene_path, run=lambda _: self.fail()),
                                 "skipped")
            self.assertEqual(calls, ["feature_extractor", "exhaustive_matcher",
                                     "mapper", "image_undistorter"])

    def test_failure_keeps_attempt_but_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = make_raw_scene(Path(tmp), "togog")

            def fail_on_match(command):
                if command[1] == "exhaustive_matcher":
                    raise subprocess.CalledProcessError(1, command)

            with patch("research_pipeline.colmap_preprocessor.shutil.which",
                       return_value="colmap"), self.assertRaises(subprocess.CalledProcessError):
                preprocess(scene_path, run=fail_on_match)
            self.assertFalse((scene_path / "images").exists())
            self.assertFalse((scene_path / "sparse").exists())
            self.assertTrue(any((scene_path / ".pipeline_colmap").iterdir()))


if __name__ == "__main__":
    unittest.main()
