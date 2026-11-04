"""CPU-only checks for the native CPU image-storage recovery preset."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_pipeline.commands import commands
from research_pipeline.config import REPO, load_config, repository_parser
from research_pipeline.results import experiment_id
from research_pipeline.run_all import main
from test_research_pipeline import make_scene


class CpuImagePresetTests(unittest.TestCase):
    def test_preset_changes_only_native_photo_storage_and_records_a_new_identity(self):
        baseline = load_config(REPO / "configs/togog_bali.yaml")
        preset = load_config(REPO / "configs/togog_bali_detya_cpu_images.yaml")
        expected = copy.deepcopy(baseline)
        expected["training"]["extra_args"][1:1] = ["--data_device", "cpu"]
        self.assertEqual(preset, expected)
        parser = repository_parser("training")
        before = vars(parser.parse_args(baseline["training"]["extra_args"]))
        after = vars(parser.parse_args(preset["training"]["extra_args"]))
        self.assertEqual(before.pop("data_device"), "cuda")
        self.assertEqual(after.pop("data_device"), "cpu")
        self.assertEqual(before, after)
        self.assertNotEqual(experiment_id("detya", baseline), experiment_id("detya", preset))
        plan = commands(preset, Path("/dataset/detya"), Path("/output/detya"))
        self.assertEqual(plan["training"][plan["training"].index("--data_device") + 1], "cpu")
        self.assertEqual(plan["image_metrics"][1], "metric.py")
        self.assertEqual(plan["mesh_extraction"][1], "mesh_extract.py")

    def test_selected_scene_dry_run_never_imports_the_model_or_starts_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            make_scene(dataset, "detya")
            make_scene(dataset, "tualen")
            config = load_config(REPO / "configs/togog_bali_detya_cpu_images.yaml")
            config["dataset"]["root"] = str(dataset)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            path = root / "preset.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = sorted(str(p) for p in root.rglob("*"))
            with patch("subprocess.Popen", side_effect=AssertionError("No subprocess")), \
                 patch("research_pipeline.runner.run_batch", side_effect=AssertionError("No training")), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                status = main(["--config", str(path), "--scene", "detya",
                               "--start-from", "training", "--dry-run"])
            self.assertEqual(status, 0)
            text = out.getvalue()
            self.assertIn("Found 1 datasets.", text)
            self.assertIn("[1/1] detya", text)
            self.assertNotIn("[2/", text)
            self.assertIn("--data_device cpu", text)
            self.assertIn("--resolution 1920", text)
            self.assertIn("--enable_allocator_gc", text)
            self.assertIn("--eval", text)
            self.assertIn("python metric.py", text)
            self.assertIn("python mesh_extract.py", text)
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
