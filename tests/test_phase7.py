import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.config import DEFAULTS
from research_pipeline.results import (
    FIELDNAMES, ensure_master_csv, experiment_id, read_rows, successful_ids,
    upsert_result, write_summary,
)
from research_pipeline.runner import run_batch
from research_pipeline.scene_discovery import Scene


class Phase7(unittest.TestCase):
    def test_experiment_id_is_deterministic_and_configuration_sensitive(self):
        first = experiment_id("togog", copy.deepcopy(DEFAULTS))
        self.assertEqual(first, experiment_id("togog", copy.deepcopy(DEFAULTS)))
        changed = copy.deepcopy(DEFAULTS)
        changed["training"]["extra_args"] = ["--iterations", "10"]
        self.assertNotEqual(first, experiment_id("togog", changed))
        changed = copy.deepcopy(DEFAULTS)
        changed["dataset"]["root"] = "/different/dataset"
        self.assertNotEqual(first, experiment_id("togog", changed))
        changed = copy.deepcopy(DEFAULTS)
        changed["image_metrics"]["enabled"] = False
        self.assertNotEqual(first, experiment_id("togog", changed))
        changed = copy.deepcopy(DEFAULTS)
        changed["colmap"]["camera_model"] = "PINHOLE"
        self.assertNotEqual(first, experiment_id("togog", changed))
        self.assertNotEqual(first, experiment_id("other", DEFAULTS))

    def test_atomic_csv_upsert_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "results/experiments.csv"
            ensure_master_csv(csv_path)
            result = {
                "experiment_id": "abc", "timestamp": "now", "scene_name": "togog",
                "object_name": "togog", "dataset_path": "/data/togog",
                "output_path": str(root / "out/togog"), "status": "failed",
                "failed_stage": "training", "error_message": "fixture",
                "training": {}, "rendering": {}, "mesh_extraction": {}, "geometry": {},
                "reproducibility": {}, "warnings": [], "commands": {},
            }
            upsert_result(csv_path, result)
            self.assertFalse(successful_ids(csv_path))
            result["status"] = "success"
            result["failed_stage"] = None
            result["error_message"] = None
            upsert_result(csv_path, result)
            self.assertEqual(len(read_rows(csv_path)), 1)
            self.assertEqual(successful_ids(csv_path), {"abc"})
            with csv_path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.reader(stream)), FIELDNAMES)
            write_summary(result)
            self.assertTrue((root / "out/togog/research/summary.json").is_file())

    def test_schema_mismatch_is_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ensure_master_csv(path)

    def test_resume_skips_only_success_and_force_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["output"]["root"] = str(root / "out")
            config["results"]["csv"] = str(root / "results.csv")
            scene = Scene("togog", root / "data/togog")
            identity = experiment_id(scene.name, config)
            ensure_master_csv(Path(config["results"]["csv"]))
            prior = {
                "experiment_id": identity, "timestamp": "old", "scene_name": "togog",
                "object_name": "togog", "dataset_path": str(scene.path),
                "output_path": str(root / "old"), "status": "success",
                "failed_stage": None, "error_message": None, "training": {},
                "rendering": {}, "mesh_extraction": {}, "geometry": {},
                "reproducibility": {}, "warnings": [], "commands": {},
            }
            upsert_result(Path(config["results"]["csv"]), prior)
            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                    patch("research_pipeline.runner.execute_scene") as execute:
                self.assertEqual(run_batch(config, [scene], resume=True), 0)
                execute.assert_not_called()

            def success(_config, _scene, output):
                return {**prior, "output_path": str(output), "status": "success",
                        "warnings": []}

            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                    patch("research_pipeline.runner.execute_scene", side_effect=success) as execute:
                self.assertEqual(run_batch(config, [scene], resume=True, force=True), 0)
                execute.assert_called_once()
            self.assertEqual(len(read_rows(Path(config["results"]["csv"]))), 1)

    def test_batch_records_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["output"]["root"] = str(root / "out")
            config["results"]["csv"] = str(root / "results.csv")
            scenes = [Scene("a", root / "data/a"), Scene("b", root / "data/b")]
            seen = []

            def result(_config, scene, output):
                seen.append(scene.name)
                failed = scene.name == "a"
                return {"scene_name": scene.name, "object_name": scene.name,
                        "dataset_path": str(scene.path), "output_path": str(output),
                        "status": "failed" if failed else "success",
                        "failed_stage": "training" if failed else None,
                        "error_message": "fixture" if failed else None,
                        "training": {}, "rendering": {}, "mesh_extraction": {},
                        "geometry": {}, "warnings": [], "commands": {}}

            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                    patch("research_pipeline.runner.execute_scene", side_effect=result):
                self.assertEqual(run_batch(config, scenes), 1)
            self.assertEqual(seen, ["a", "b"])
            self.assertEqual(len(read_rows(Path(config["results"]["csv"]))), 2)


if __name__ == "__main__":
    unittest.main()
