"""Exercise failed-attempt retention using real temporary outputs and reports."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_pipeline.attempt_retention import (
    cleanup_failed_attempts, finalize_retention, main, safe_path,
)
from research_pipeline.config import DEFAULTS
from research_pipeline.reporting import automatic_report, read_csv
from research_pipeline.results import (
    ensure_master_csv, experiment_id, read_rows, upsert_result, write_summary,
)
from research_pipeline.runner import run_batch
from research_pipeline.scene_discovery import Scene
from research_pipeline.utils import read_json, write_json
from test_research_reporting import fixture


class AttemptRetentionTests(unittest.TestCase):
    def config(self, root):
        config = copy.deepcopy(DEFAULTS)
        config["dataset"]["root"] = str(root / "data")
        config["output"]["root"] = str(root / "output")
        config["results"]["csv"] = str(root / "results.csv")
        config["rendering"]["enabled"] = False
        config["mesh"]["enabled"] = False
        return config

    def result(self, config, path, status="success", identity=None):
        _, result = fixture(path.parent, path.name, count=1)
        result.update(
            scene_name="patung", object_name="patung",
            dataset_path=str(Path(config["dataset"]["root"]) / "patung"),
            experiment_id=identity or experiment_id("patung", config),
            status=status, failed_stage="training" if status == "failed" else None,
            error_message="fixture failure" if status == "failed" else None,
            mesh_extraction={}, geometry={}, commands={}, reproducibility={},
        )
        return result

    def saved(self, config, path, status="success", identity=None):
        result = self.result(config, path, status, identity)
        write_summary(result)
        master = Path(config["results"]["csv"])
        ensure_master_csv(master)
        result["reporting"] = automatic_report(path, master)
        write_summary(result)
        upsert_result(master, result)
        return result

    def tables(self, root):
        return [read_csv(root / ("results" + suffix))
                for suffix in ("_laporan.csv", "_ringkasan.csv")]

    def test_success_removes_base_failure_and_failed_siblings_but_keeps_new_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            old = self.saved(config, base, "failed")
            sibling = self.saved(config, base / "runs/failed_again", "failed")
            winner = self.saved(config, base / "runs/winner")
            info = cleanup_failed_attempts(config, winner)
            self.assertEqual(set(info["outputs"]), {str(base), sibling["output_path"]})
            self.assertFalse((base / "research").exists())
            self.assertFalse((base / "point_cloud").exists())
            self.assertFalse(Path(sibling["output_path"]).exists())
            self.assertTrue((Path(winner["output_path"]) / "point_cloud/iteration_30000/point_cloud.ply").is_file())
            self.assertFalse(Path(old["reporting"]["directory"]).exists())
            self.assertFalse(Path(sibling["reporting"]["directory"]).exists())
            self.assertTrue(Path(winner["reporting"]["directory"]).is_dir())
            for rows in self.tables(root):
                self.assertEqual([(row["Status"], row["Folder output"]) for row in rows],
                                 [("success", winner["output_path"])])
            self.assertEqual(len(read_rows(root / "results.csv")), 1)
            from research_pipeline.report_collection import inspect_outputs
            chosen, history = inspect_outputs(root / "output")
            self.assertEqual(len(history), 1)
            self.assertEqual(chosen[0]["summary"]["output_path"], winner["output_path"])
            self.assertEqual(cleanup_failed_attempts(config, winner)["outputs"], [])

    def test_failed_retry_keeps_prior_outputs_until_a_success_exists(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            self.saved(config, base, "failed")
            retry = self.saved(config, base / "runs/retry", "failed")
            self.assertEqual(cleanup_failed_attempts(config, retry)["outputs"], [])
            self.assertTrue((base / "research/summary.json").is_file())
            self.assertTrue((Path(retry["output_path"]) / "research/summary.json").is_file())

    def test_existing_success_and_other_configuration_failures_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            previous = self.saved(config, base)
            other = self.saved(config, base / "runs/different_config", "failed", "different-id")
            failed = self.saved(config, base / "runs/failed", "failed")
            winner = self.saved(config, base / "runs/winner")
            cleanup_failed_attempts(config, winner)
            self.assertTrue((base / "research/summary.json").is_file())
            self.assertTrue(Path(previous["reporting"]["directory"]).is_dir())
            self.assertTrue(Path(other["output_path"]).is_dir())
            self.assertFalse(Path(failed["output_path"]).exists())
            self.assertEqual(len(self.tables(root)[0]), 3)

    def test_missing_success_record_model_mesh_or_report_prevents_deletion(self):
        for missing in ("master", "model", "mesh", "report"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                config = self.config(root)
                base = root / "output/patung"
                old = self.saved(config, base, "failed")
                winner = self.saved(config, base / "runs/winner")
                if missing == "master":
                    upsert_result(root / "results.csv", old)
                elif missing == "model":
                    (Path(winner["output_path"]) / "point_cloud/iteration_30000/point_cloud.ply").unlink()
                elif missing == "mesh":
                    config["mesh"]["enabled"] = True
                else:
                    winner["reporting"]["status"] = "failed"
                    write_summary(winner)
                before = [(root / name).read_bytes() for name in ("results_laporan.csv", "results_ringkasan.csv")]
                with self.assertRaises(ValueError):
                    cleanup_failed_attempts(config, winner)
                self.assertTrue((base / "point_cloud/iteration_30000/point_cloud.ply").is_file())
                self.assertTrue(Path(old["reporting"]["directory"]).is_dir())
                self.assertEqual(before, [(root / name).read_bytes() for name in ("results_laporan.csv", "results_ringkasan.csv")])

    def test_unidentified_failed_outputs_and_successful_mesh_recoveries_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            old = self.saved(config, base, "failed")
            write_json(base / "research/mesh_retry_fixed/result.json", {"status": "success"})
            unknown = base / "runs/unidentified"
            unknown.mkdir(parents=True)
            (unknown / "partial.ply").write_bytes(b"keep")
            damaged = base / "runs/damaged/research/summary.json"
            damaged.parent.mkdir(parents=True)
            damaged.write_text("{broken", encoding="utf-8")
            winner = self.saved(config, base / "runs/winner")
            info = cleanup_failed_attempts(config, winner)
            self.assertEqual(info["outputs"], [])
            self.assertTrue(info["warnings"])
            self.assertTrue((base / "research/summary.json").is_file())
            self.assertTrue((unknown / "partial.ply").is_file())
            self.assertTrue(damaged.is_file())
            self.assertTrue(Path(old["reporting"]["directory"]).is_dir())

    def test_csv_write_failure_keeps_old_model_and_new_success(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            self.saved(config, base, "failed")
            winner = self.saved(config, base / "runs/winner")
            with patch("research_pipeline.attempt_retention.atomic_write", side_effect=OSError("disk full")):
                info = finalize_retention(config, winner)
            self.assertEqual(info["status"], "incomplete")
            self.assertTrue((base / "point_cloud/iteration_30000/point_cloud.ply").is_file())
            self.assertEqual(read_rows(root / "results.csv")[0]["status"], "success")
            self.assertEqual(read_json(Path(winner["output_path"]) / "research/summary.json")["status"], "success")

    def test_preview_is_read_only_and_apply_cleans_existing_recorded_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            self.saved(config, base, "failed")
            self.saved(config, base / "runs/winner")
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(main(["--config", str(config_file), "--dry-run"]), 0)
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertEqual(main(["--config", str(config_file), "--apply"]), 0)
            self.assertFalse((base / "research").exists())
            self.assertEqual(len(self.tables(root)[0]), 1)

    def test_real_batch_cleans_failure_after_success_and_resume_can_finish_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            scene = Scene("patung", root / "data/patung")
            statuses = iter(("failed", "success"))
            def execute(_config, _scene, output):
                return self.result(config, output, next(statuses))
            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                 patch("research_pipeline.runner.execute_scene", side_effect=execute):
                self.assertEqual(run_batch(config, [scene]), 1)
                self.assertEqual(run_batch(config, [scene]), 0)
            rows = self.tables(root)[0]
            self.assertEqual([row["Status"] for row in rows], ["success"])
            winner = read_json(Path(rows[0]["Folder output"]) / "research/summary.json")
            base = root / "output/patung"
            old = self.saved(config, base / "runs/old_failure", "failed")
            upsert_result(root / "results.csv", winner)
            with patch("research_pipeline.runner.execute_scene") as executor:
                self.assertEqual(run_batch(config, [scene], resume=True), 0)
                executor.assert_not_called()
            self.assertFalse(Path(old["output_path"]).exists())
            self.assertEqual([row["Status"] for row in self.tables(root)[0]], ["success"])

    def test_cleanup_rejects_external_targets_and_linked_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            self.saved(config, base, "failed")
            winner = self.saved(config, base / "runs/winner")
            with self.assertRaises(ValueError):
                safe_path(root / "external", root / "output")
            actual = Path.is_symlink
            with patch.object(Path, "is_symlink", lambda path: path == base or actual(path)):
                with self.assertRaises(ValueError):
                    cleanup_failed_attempts(config, winner)
            self.assertTrue((base / "point_cloud/iteration_30000/point_cloud.ply").is_file())
            gaussian = Path(winner["output_path"]) / "point_cloud/iteration_30000/point_cloud.ply"
            with patch.object(Path, "is_symlink", lambda path: path == gaussian or actual(path)):
                with self.assertRaises(ValueError):
                    cleanup_failed_attempts(config, winner)
            self.assertTrue((base / "point_cloud/iteration_30000/point_cloud.ply").is_file())

    def test_copied_summary_with_foreign_output_path_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config(root)
            base = root / "output/patung"
            old = self.saved(config, base, "failed")
            old["output_path"] = str(root / "elsewhere")
            write_json(base / "research/summary.json", old)
            winner = self.saved(config, base / "runs/winner")
            self.assertEqual(cleanup_failed_attempts(config, winner)["outputs"], [])
            self.assertTrue((base / "point_cloud/iteration_30000/point_cloud.ply").is_file())


class MeshAttemptRetentionTests(unittest.TestCase):
    def fixture(self, root):
        from test_mesh_recovery_reporting import source_run, mesh_retry
        model, source = source_run(root, "patung")
        (model / "cfg_args").write_text("Namespace()", encoding="utf-8")
        winner, record = mesh_retry(model, source)
        return model, source, winner, record

    def failed_retry(self, model, record, name="mesh_retry_failed", **updates):
        path = model / "research" / name
        path.mkdir()
        failed = copy.deepcopy(record)
        failed.update(status="failed", exit_code=1, attempt_path=str(path),
                      failed_stage="mesh_extraction", **updates)
        write_json(path / "result.json", failed)
        (path / "recon_post.ply").write_bytes(b"partial failed mesh")
        return path

    def test_successful_mesh_retry_removes_only_identified_failed_siblings(self):
        from research_pipeline.attempt_retention import cleanup_mesh_retries
        with tempfile.TemporaryDirectory() as tmp:
            model, source, winner, record = self.fixture(Path(tmp))
            failed = self.failed_retry(model, record)
            running = self.failed_retry(model, record, name="mesh_retry_running")
            running_record = read_json(running / "result.json")
            running_record["status"] = "running"
            write_json(running / "result.json", running_record)
            different = self.failed_retry(model, record, name="mesh_retry_other_source")
            other_record = read_json(different / "result.json")
            other_record["source_summary_snapshot"]["experiment_id"] = "other-experiment"
            write_json(different / "result.json", other_record)
            before = {p: p.read_bytes() for p in model.rglob("*") if p.is_file()
                      and not p.is_relative_to(failed)}
            info = cleanup_mesh_retries(winner)
            self.assertEqual(info["outputs"], [str(failed)])
            self.assertFalse(failed.exists())
            self.assertTrue(running.is_dir())
            self.assertTrue(different.is_dir())
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_missing_or_invalid_successful_mesh_keeps_failed_retry(self):
        from research_pipeline.attempt_retention import cleanup_mesh_retries
        with tempfile.TemporaryDirectory() as tmp:
            model, _, winner, record = self.fixture(Path(tmp))
            failed = self.failed_retry(model, record)
            (winner / "recon_post.ply").unlink()
            with self.assertRaises(OSError):
                cleanup_mesh_retries(winner)
            self.assertTrue(failed.is_dir())
            self.assertTrue((model / "point_cloud/iteration_30000/point_cloud.ply").is_file())

    def test_retry_function_cleans_after_recording_a_success(self):
        from research_pipeline.mesh_retry import retry_mesh
        from research_pipeline.runner import ProcessResult
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            model, _, existing_success, record = self.fixture(Path(tmp))
            failed = self.failed_retry(model, record)
            mesh_bytes = (existing_success / "recon_post.ply").read_bytes()
            source_before = (model / "research/summary.json").read_bytes()
            def execute(command, log, interval):
                log.write_text("completed", encoding="utf-8")
                (log.parent / "recon_post.ply").write_bytes(mesh_bytes)
                return ProcessResult(0, 10, {})
            def analyze(path, metrics, scene):
                return {**record["mesh_metrics"], "mesh_path": str(path),
                        "file_size_mb": path.stat().st_size / 1024**2}
            with patch("research_pipeline.mesh_retry._prepare_model"):
                result = retry_mesh(model, max_depth=8, executor=execute, analyzer=analyze)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["retention"]["outputs"], [str(failed)])
            self.assertFalse(failed.exists())
            self.assertTrue(existing_success.is_dir())
            self.assertEqual((model / "research/summary.json").read_bytes(), source_before)
            saved = read_json(Path(result["attempt_path"]) / "result.json")
            self.assertEqual(saved["retention"]["status"], "complete")

    def test_export_replaces_failed_source_row_and_cleans_failed_retries(self):
        from research_pipeline.reporting import export, update_result_tables
        from research_pipeline.mesh_recovery import RECOVERED_STATUS
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            model, _, winner, record = self.fixture(root)
            failed = self.failed_retry(model, record)
            results = root / "results.csv"
            automatic_report(model, results)
            before = (model / "research/summary.json").read_bytes()
            package = export([], root / "recovered", pairs=1, mesh_retries=[winner])
            update_result_tables(package, results)
            for suffix in ("_laporan.csv", "_ringkasan.csv"):
                rows = read_csv(root / ("results" + suffix))
                self.assertEqual([row["Status"] for row in rows], [RECOVERED_STATUS])
                self.assertEqual(rows[0]["Folder output"], str(winner))
            self.assertFalse(failed.exists())
            self.assertEqual((model / "research/summary.json").read_bytes(), before)
            self.assertTrue((model / "point_cloud/iteration_30000/point_cloud.ply").is_file())

    def test_mesh_cleanup_cli_preview_does_not_delete_and_apply_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            model, _, winner, record = self.fixture(Path(tmp))
            failed = self.failed_retry(model, record)
            before = (model / "research/summary.json").read_bytes()
            self.assertEqual(main(["--mesh-retries", str(winner), "--dry-run"]), 0)
            self.assertTrue(failed.is_dir())
            self.assertEqual(main(["--mesh-retries", str(winner), "--apply"]), 0)
            self.assertFalse(failed.exists())
            self.assertEqual((model / "research/summary.json").read_bytes(), before)

    def test_directory_link_in_failed_retry_does_not_delete_shared_model(self):
        from research_pipeline.attempt_retention import cleanup_mesh_retries
        with tempfile.TemporaryDirectory() as tmp:
            model, _, winner, record = self.fixture(Path(tmp))
            failed = self.failed_retry(model, record)
            try:
                (failed / "point_cloud").symlink_to(model / "point_cloud", target_is_directory=True)
            except OSError:
                self.skipTest("Directory symlinks are unavailable in this Windows session")
            gaussian = model / "point_cloud/iteration_30000/point_cloud.ply"
            before = gaussian.read_bytes()
            cleanup_mesh_retries(winner)
            self.assertFalse(failed.exists())
            self.assertEqual(gaussian.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
