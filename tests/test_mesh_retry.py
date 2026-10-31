"""CPU checks for recorded, isolated mesh retries."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.mesh_retry import _prepare_model, retry_mesh
from research_pipeline.runner import ProcessResult


class MeshRetryTests(unittest.TestCase):
    def make_model(self, root):
        model = root / "dewa_brahma"
        ply = model / "point_cloud/iteration_30000/point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"original trained model")
        (model / "cfg_args").write_text("original config", encoding="utf-8")
        (model / "recon_post.ply").write_bytes(b"original mesh")
        summary = model / "research/summary.json"
        summary.parent.mkdir()
        summary.write_text(json.dumps({"status": "failed", "training": {
            "duration_seconds": 1234}, "rendering": {"psnr": 25.3}}), encoding="utf-8")
        return model

    def test_preparation_copies_only_config_and_links_existing_point_cloud(self):
        with patch("research_pipeline.mesh_retry.shutil.copy2") as copy, \
             patch.object(Path, "symlink_to") as link:
            _prepare_model(Path("model"), Path("attempt"))
        copy.assert_called_once_with(Path("model/cfg_args"), Path("attempt/cfg_args"))
        link.assert_called_once_with(Path("model/point_cloud"), target_is_directory=True)

    def test_success_records_real_retry_settings_and_preserves_original_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            originals = {p: p.read_bytes() for p in model.rglob("*") if p.is_file()}
            commands = []

            def execute(command, log, interval):
                commands.append(command)
                self.assertEqual(interval, 5)
                self.assertEqual(command[command.index("--max_depth") + 1], "20.0")
                self.assertNotIn("train.py", command)
                log.write_text("mesh completed", encoding="utf-8")
                (log.parent / "recon_post.ply").write_bytes(b"new mesh")
                return ProcessResult(0, 42, {
                    "peak_ram_mb": 1000, "peak_vram_mb": None,
                    "device_global": {"device_vram_used_mb": {"peak": 9000}}})

            def analyze(path, metrics, scene):
                self.assertNotEqual(path, model / "recon_post.ply")
                self.assertEqual(path.read_bytes(), b"new mesh")
                return {"vertex_count": 100, "face_count": 80, "file_size_mb": 1}

            with patch("research_pipeline.mesh_retry._prepare_model"):
                result = retry_mesh(model, max_depth=20.0, executor=execute, analyzer=analyze)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["source_summary_snapshot"]["training"]["duration_seconds"], 1234)
            attempt = Path(result["attempt_path"])
            self.assertEqual(json.loads((attempt / "result.json").read_text())["settings"]["script"], "mesh_extract.py")
            with (attempt / "result.csv").open(encoding="utf-8-sig", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["mesh_duration_minutes"], "0.7")
            self.assertEqual(row["mesh_peak_process_vram_mb"], "")
            self.assertEqual(row["mesh_peak_device_vram_sample_mb"], "9000")
            self.assertEqual(row["mesh_vertex_count"], "100")
            self.assertEqual(len(commands), 1)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_failed_extraction_keeps_exit_code_duration_and_log_without_analyzing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))

            def execute(command, log, interval):
                log.write_text("No block is touched in TSDF volume", encoding="utf-8")
                return ProcessResult(1, 19, {"peak_ram_mb": 222})

            with patch("research_pipeline.mesh_retry._prepare_model"), \
                 patch("research_pipeline.mesh_retry.analyze_mesh") as analyze:
                result = retry_mesh(model, max_depth=8.0, executor=execute, analyzer=analyze)
            analyze.assert_not_called()
            self.assertEqual((result["status"], result["exit_code"], result["duration_seconds"]), ("failed", 1, 19))
            self.assertEqual(result["failed_stage"], "mesh_extraction")
            self.assertTrue((Path(result["attempt_path"]) / "result.csv").is_file())

    def test_zero_exit_without_mesh_is_a_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            with patch("research_pipeline.mesh_retry._prepare_model"):
                result = retry_mesh(model, max_depth=20, executor=lambda *a: ProcessResult(0, 5))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_stage"], "mesh_validation")
            self.assertIn("Final mesh was not created", result["error_message"])

    def test_interrupt_and_preparation_error_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            with patch("research_pipeline.mesh_retry._prepare_model", side_effect=OSError("link unavailable")):
                failed = retry_mesh(model, max_depth=20)
            self.assertEqual(failed["failed_stage"], "preparation")
            with patch("research_pipeline.mesh_retry._prepare_model"):
                interrupted = retry_mesh(model, max_depth=20, executor=lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["exit_code"], 130)
            self.assertNotEqual(failed["attempt_path"], interrupted["attempt_path"])
            self.assertTrue((Path(interrupted["attempt_path"]) / "result.json").is_file())

    def test_invalid_settings_or_missing_model_do_not_create_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            before = list((model / "research").iterdir())
            for depth in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    retry_mesh(model, max_depth=depth)
            with self.assertRaises(ValueError):
                retry_mesh(model, max_depth=20, iteration=7000)
            self.assertEqual(list((model / "research").iterdir()), before)


if __name__ == "__main__":
    unittest.main()
