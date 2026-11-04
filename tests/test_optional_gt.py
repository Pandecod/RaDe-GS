"""CPU-only checks for new output trees and optional GT geometry."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.commands import commands
from research_pipeline.config import DEFAULTS, REPO, load_config
from research_pipeline.geometry_stage import evaluate_geometry_stage
from research_pipeline.ground_truth import ground_truth_path
from research_pipeline.reporting import export, read_csv
from research_pipeline.results import csv_row, ensure_master_csv, upsert_result, write_summary
from research_pipeline.runner import ProcessResult, execute_scene
from research_pipeline.scene_discovery import Scene
from research_pipeline.utils import read_json, write_json
from test_research_pipeline import make_scene
from test_research_reporting import fixture as report_fixture


class OptionalGTTests(unittest.TestCase):
    def config(self, root):
        config = copy.deepcopy(DEFAULTS)
        config["geometry_evaluation"].update(enabled=True, gt_root=str(root / "gt"))
        config["resource_guard"]["enabled"] = False
        return config

    def command(self, config, output):
        return commands(config, output.parent / "data/patung", output)["geometry_evaluation"]

    def test_active_configs_use_fresh_outputs_and_separate_tables(self):
        for dataset in ("togog_bali", "tanks_temples"):
            config = load_config(REPO / "configs" / (dataset + ".yaml"))
            self.assertEqual(config["output"]["root"], f"output/hasil_radegs_v2/{dataset}")
            self.assertEqual(config["results"]["csv"],
                             f"research_results/hasil_radegs_v2/experiment_results_{dataset}.csv")
            self.assertTrue(config["geometry_evaluation"]["enabled"])
            self.assertTrue(config["geometry_evaluation"]["skip_missing_gt"])

    def test_missing_gt_records_skip_without_starting_geometry_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            output = root / "out"
            with patch("subprocess.Popen", side_effect=AssertionError("No process")), \
                 patch("research_pipeline.runner.run_process", side_effect=AssertionError("No executor")) as executor:
                result = evaluate_geometry_stage(config, "patung", output, self.command(config, output), executor)
            executor.assert_not_called()
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["skip_reason"], "missing_ground_truth")
            for key in ("chamfer_distance", "precision", "recall", "f1_score"):
                self.assertIsNone(result[key])
            self.assertEqual(read_json(output / "research/geometry_metrics.json")["status"], "skipped")
            self.assertIn("skipped", (output / "research/logs/geometry_evaluation.log").read_text())

    def test_missing_gt_skip_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            config["geometry_evaluation"]["skip_missing_gt"] = False
            result = evaluate_geometry_stage(config, "patung", root / "out",
                self.command(config, root / "out"), lambda *a: self.fail("No process"))
            self.assertEqual(result["status"], "failed")
            self.assertIn("Ground truth does not exist", result["error_message"])

    def test_existing_gt_runs_and_returns_real_geometry_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            gt = root / "gt/patung.ply"
            gt.parent.mkdir()
            gt.write_bytes(b"reference fixture")
            output = root / "out"
            called = []
            def executor(command, log, interval):
                called.append(command)
                write_json(output / "research/geometry_metrics.json", {
                    "chamfer_distance": .25, "precision": None, "recall": None,
                    "f1_score": None, "warnings": ["distance_threshold is null"]})
                return ProcessResult(0, 2.)
            result = evaluate_geometry_stage(config, "patung", output,
                                            self.command(config, output), executor)
            self.assertEqual(len(called), 1)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["chamfer_distance"], .25)
            self.assertEqual(result["duration_seconds"], 2.)
            self.assertEqual(result["ground_truth_path"], str(gt))
            self.assertEqual(result["evaluator"], "research_pipeline.geometry_evaluator")

    def test_gt_directory_or_bad_execution_is_failure_not_skip(self):
        for bad in ("directory", "nonzero", "incomplete"):
            with self.subTest(kind=bad), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.config(root)
                path = root / "gt/patung.ply"
                path.parent.mkdir()
                if bad == "directory":
                    path.mkdir()
                else:
                    path.write_bytes(b"reference")
                output = root / "out"
                def executor(command, log, interval):
                    if bad == "directory":
                        self.fail("Invalid GT must not execute")
                    if bad == "nonzero":
                        return ProcessResult(1, 1.)
                    write_json(output / "research/geometry_metrics.json", {"chamfer_distance": None})
                    return ProcessResult(0, 1.)
                result = evaluate_geometry_stage(config, "patung", output,
                                                self.command(config, output), executor)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["skip_reason"])
                self.assertTrue(result["error_message"])

    def test_nested_gt_supported_without_treating_colmap_as_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            nested = root / "gt/patung/patung.ply"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"reference")
            self.assertEqual(ground_truth_path(config, "patung"), nested)
            flat = root / "gt/patung.ply"
            flat.write_bytes(b"explicit reference")
            self.assertEqual(ground_truth_path(config, "patung"), flat)

    def test_missing_gt_keeps_scene_successful_and_preserves_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            config["rendering"]["enabled"] = False
            config["image_metrics"]["enabled"] = False
            scene = Scene("patung", make_scene(root / "data", "patung"))
            output = root / "out"
            seen = []
            def executor(command, log, interval):
                seen.append(command[1])
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("complete")
                if command[1] == "train.py":
                    model = output / "point_cloud/iteration_30000/point_cloud.ply"
                    model.parent.mkdir(parents=True)
                    model.write_bytes(b"Gaussian result")
                else:
                    self.assertEqual(command[1], "mesh_extract.py")
                    (output / "recon_post.ply").write_bytes(b"mesh result")
                return ProcessResult(0, 1.)
            with patch("research_pipeline.mesh_analyzer.analyze_mesh", return_value={"vertex_count": 3}):
                result = execute_scene(config, scene, output, executor)
            self.assertEqual(seen, ["train.py", "mesh_extract.py"])
            self.assertEqual(result["status"], "success")
            self.assertIsNone(result["failed_stage"])
            self.assertEqual(result["geometry"]["status"], "skipped")
            self.assertEqual((output / "recon_post.ply").read_bytes(), b"mesh result")
            self.assertEqual((output / "point_cloud/iteration_30000/point_cloud.ply").read_bytes(),
                             b"Gaussian result")

    def test_gt_skip_visible_in_summary_master_csv_and_research_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, summary = report_fixture(root, count=2)
            summary["geometry"] = {"status": "skipped", "skip_reason": "missing_ground_truth",
                                   "ground_truth_path": str(root / "gt/patung.ply"),
                                   "chamfer_distance": None,
                                   "evaluator": "research_pipeline.geometry_evaluator"}
            write_summary(summary)
            table = root / "tables/experiments.csv"
            ensure_master_csv(table)
            upsert_result(table, summary)
            with table.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["geometry_status"], "skipped")
            self.assertEqual(row["geometry_skip_reason"], "missing_ground_truth")
            self.assertEqual(row["chamfer_distance"], "")
            report = export([output], root / "report", pairs=1)
            row = read_csv(report / "tabel_evaluasi.csv")[0]
            self.assertEqual(row["Status GT 3D"], "skipped")
            self.assertEqual(row["Alasan GT dilewati"], "missing_ground_truth")
            self.assertEqual(row["Chamfer (satuan rekonstruksi)"], "")
            self.assertEqual(read_json(report / "detail_laporan.json")["experiments"][0]["geometry"]["status"], "skipped")

    def test_existing_geometry_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            output = root / "out"
            artifact = output / "research/geometry_metrics.json"
            write_json(artifact, {"existing": "preserve"})
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluate_geometry_stage(config, "patung", output,
                    self.command(config, output), lambda *a: self.fail("No executor"))
            self.assertEqual(read_json(artifact), {"existing": "preserve"})


if __name__ == "__main__":
    unittest.main()
