"""CPU checks for importing recorded mesh retries into research reports."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from research_pipeline.mesh_recovery import prepare_mesh_recovery, RECOVERED_STATUS
from research_pipeline.reporting import (
    export, main, read_csv, update_result_tables, automatic_report,
)
from research_pipeline.utils import read_json, write_json
from test_research_reporting import fixture


def source_run(root, name, *, status="failed"):
    model, summary = fixture(root, name=name, count=1)
    (model / "train").rename(model / "test")
    manifest = model / "test/ours_30000/render_manifest.json"
    data = read_json(manifest)
    data["split"] = "test"
    write_json(manifest, data)
    summary.update(status=status, failed_stage="mesh_extraction" if status == "failed" else None,
                   error_message="Mesh extraction exited with code -11" if status == "failed" else None)
    summary["training"].update(initial_point_count=5, camera_count=3,
                               training_camera_count=2, test_camera_count=1, peak_ram_mb=3000)
    summary["rendering"].update(split="test", psnr=25.123456789, duration_seconds=60)
    summary["mesh_extraction"] = {"duration_seconds": 900, "peak_ram_mb": 9999,
                                  "average_ram_mb": 8888, "voxel_size": .009, "max_depth": 8.}
    summary["geometry"] = {"status": "not_run", "sample_points": 100000}
    write_json(model / "research/summary.json", summary)
    return model, summary


def mesh_retry(model, summary):
    attempt = model / "research/mesh_retry_saved"
    attempt.mkdir()
    mesh = attempt / "recon_post.ply"
    mesh.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n", encoding="ascii")
    record = {
        "kind": "mesh_retry", "scene_name": model.name,
        "model_path": str(model), "attempt_path": str(attempt),
        "source_summary_path": str(model / "research/summary.json"),
        "source_summary_snapshot": copy.deepcopy(summary), "training_reused": True,
        "status": "success", "exit_code": 0, "duration_seconds": 30.,
        "finished_utc": "2026-11-04T23:04:46+00:00",
        "settings": {"script": "mesh_extract.py", "iteration": 30000,
                     "resolution": 2, "num_cluster": 8, "depth_ratio": .5,
                     "voxel_size": .009, "max_depth": 8., "block_resolution": 8},
        "process_metrics": {
            "peak_ram_mb": 512., "average_ram_mb": 256., "peak_vram_mb": None,
            "average_vram_mb": None,
            "device_global": {"device_vram_used_mb": {"peak": 8192., "average": 7000.}}},
        "mesh_metrics": {"vertex_count": 3, "face_count": 1,
                         "file_size_mb": mesh.stat().st_size / 1024**2, "mesh_path": str(mesh)},
        "native_runtime": {"block_resolution": 8, "active_voxel_blocks": 174622,
                           "mesh_structure_bytes": 174622 * 8**3 * 16},
        "command": ["python", "-m", "research_pipeline.native_mesh_runtime"],
        "warnings": ["Block layout recovery experiment"], "script_sha256": "recorded-native-hash",
        "native_runtime_sha256": "recorded-runtime-hash",
    }
    write_json(attempt / "result.json", record)
    return attempt, record


class MeshRecoveryReportingTests(unittest.TestCase):
    def test_combined_views_preserve_original_runs_and_use_retry_mesh_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "research_results/results.csv"
            detya, _ = source_run(root, "detya", status="success")
            tualen, old = source_run(root, "tualen")
            rangda, other = source_run(root, "rangda")
            for model in (detya, tualen, rangda):
                automatic_report(model, results)
            before_rows = read_csv(results.with_name("results_ringkasan.csv"))
            retry, _ = mesh_retry(tualen, old)
            second, _ = mesh_retry(rangda, other)
            source_files = {p: p.read_bytes() for model in (detya, tualen, rangda)
                            for p in model.rglob("*") if p.is_file()}
            results.write_bytes(b"original technical CSV is retained byte for byte")
            technical_before = results.read_bytes()
            destination = root / "research_results/recovery_report"
            with patch("subprocess.Popen", side_effect=AssertionError("No training or subprocess")):
                export([], destination, pairs=1, mesh_retries=[retry, second])
                update_result_tables(destination, results)
            rows = read_csv(results.with_name("results_ringkasan.csv"))
            self.assertEqual(len(rows), 5)
            for original in before_rows:
                self.assertIn(original, rows)
            recovered = [row for row in rows if row["Status"] == RECOVERED_STATUS]
            self.assertEqual({r["Folder"] for r in recovered}, {"tualen", "rangda"})
            for row in recovered:
                self.assertEqual(row["Training (menit)"], "2.0")
                self.assertEqual(row["PSNR (dB)"], "25.123457")
                self.assertEqual(row["Split evaluasi"], "test")
                self.assertEqual(row["Peak VRAM seluruh GPU (GiB)"], "2.0")
                self.assertEqual(row["Gaussian awal"], "5")
                self.assertEqual(row["Splats PLY akhir"], "17")
            recovery = read_csv(results.with_name("results_pemulihan_mesh.csv"))
            self.assertEqual(len(recovery), 2)
            self.assertEqual(recovery[0]["Block resolution"], "8")
            self.assertEqual(recovery[0]["Status run asal"], "failed")
            self.assertEqual(recovery[0]["Waktu selesai retry (UTC+08)"], "2026-11-05 07:04:46")
            self.assertEqual(recovery[0]["Durasi mesh (menit)"], "0.5")
            self.assertEqual(recovery[0]["RAM puncak mesh (MiB)"], "512.0")
            self.assertEqual(recovery[0]["VRAM puncak proses mesh (MiB)"], "")
            self.assertEqual(recovery[0]["Peak VRAM seluruh GPU saat mesh (GiB)"], "8.0")
            self.assertEqual(recovery[0]["Vertex mesh"], "3")
            resources = read_csv(destination / "rincian_proses.csv")
            meshes = [r for r in resources if r["Tahap"] == "Ekstraksi mesh"]
            self.assertEqual({r["RAM rata-rata proses (MiB)"] for r in meshes}, {"256.0"})
            self.assertEqual({r["Durasi (menit)"] for r in meshes}, {"0.5"})
            detail = read_json(destination / "detail_laporan.json")["experiments"][0]
            self.assertEqual(detail["recovery"]["source_experiment_id"], old["experiment_id"])
            self.assertEqual(detail["mesh_extraction"]["block_resolution"], 8)
            self.assertEqual(detail["recovery"]["model_path"], str(tualen))
            self.assertEqual(detail["comparison"]["status"], "complete")
            self.assertEqual(detail["geometry"]["status"], "not_run")
            with Image.open(detail["comparison"]["files"][0]) as picture:
                self.assertEqual(picture.getpixel((0, 48)), (0, 40, 60))
            for path, before in source_files.items():
                self.assertEqual(path.read_bytes(), before, str(path))
            self.assertEqual(results.read_bytes(), technical_before)
            again = root / "research_results/second_report"
            export([], again, pairs=1, mesh_retries=[retry])
            update_result_tables(again, results)
            self.assertEqual(len(read_csv(results.with_name("results_ringkasan.csv"))), 5)
            self.assertEqual(len(read_csv(results.with_name("results_pemulihan_mesh.csv"))), 2)

    def test_mixed_export_includes_detya_and_retry_as_separate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detya, _ = source_run(root, "detya", status="success")
            rangda, original = source_run(root, "rangda")
            retry, _ = mesh_retry(rangda, original)
            report = export([detya], root / "mixed", pairs=1, mesh_retries=[retry])
            rows = read_csv(report / "ringkasan_hasil.csv")
            self.assertEqual([r["Folder"] for r in rows], ["detya", "rangda"])
            self.assertEqual([r["Status"] for r in rows], ["success", RECOVERED_STATUS])
            self.assertEqual(rows[0]["Folder output"], str(detya))
            self.assertEqual(rows[1]["Folder output"], str(retry))

    def test_failed_running_changed_or_unverifiable_retries_never_publish_success(self):
        changes = {
            "failed": lambda r: r.update(status="failed", exit_code=-11),
            "running": lambda r: r.update(status="running"),
            "nonzero_exit": lambda r: r.update(exit_code=-11),
            "missing_snapshot": lambda r: r.pop("source_summary_snapshot"),
            "wrong_iteration": lambda r: r["settings"].update(iteration=7000),
            "wrong_mesh_path": lambda r: r["mesh_metrics"].update(mesh_path="other.ply"),
            "wrong_vertex_count": lambda r: r["mesh_metrics"].update(vertex_count=999),
            "missing_timezone": lambda r: r.update(finished_utc="2026-11-04T23:04:46"),
            "failed_other_stage": lambda r: r["source_summary_snapshot"].update(failed_stage="training"),
            "no_image_metric": lambda r: r["source_summary_snapshot"]["rendering"].update(psnr=None),
        }
        for name, change in changes.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model, summary = source_run(root, "tualen")
                retry, record = mesh_retry(model, summary)
                change(record)
                # Keep current/frozen source aligned for tests targeting stage/metric validation.
                if name in ("failed_other_stage", "no_image_metric"):
                    write_json(model / "research/summary.json", record["source_summary_snapshot"])
                write_json(retry / "result.json", record)
                destination = root / "bad_report"
                with self.assertRaises((ValueError, OSError)):
                    export([], destination, mesh_retries=[retry])
                self.assertFalse(destination.exists())

    def test_changed_source_and_truncated_mesh_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, summary = source_run(root, "rangda")
            retry, _ = mesh_retry(model, summary)
            changed = copy.deepcopy(summary)
            changed["rendering"]["psnr"] = 99.
            write_json(model / "research/summary.json", changed)
            with self.assertRaisesRegex(ValueError, "Source summary changed"):
                prepare_mesh_recovery(retry)
            write_json(model / "research/summary.json", summary)
            mesh = retry / "recon_post.ply"
            mesh.write_bytes(mesh.read_bytes()[:-5])
            with self.assertRaisesRegex(ValueError, "mesh size"):
                prepare_mesh_recovery(retry)

    def test_geometry_for_an_older_mesh_is_not_reused_or_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, summary = source_run(Path(tmp), "tualen", status="success")
            summary["geometry"] = {"status": "success", "chamfer_distance": .123,
                                   "precision": .9, "recall": .8, "f1_score": .85}
            write_json(model / "research/summary.json", summary)
            retry, _ = mesh_retry(model, summary)
            _, merged = prepare_mesh_recovery(retry)
            self.assertEqual(merged["geometry"]["status"], "not_run")
            self.assertEqual(merged["geometry"]["skip_reason"], "mesh_retry_not_evaluated")
            for key in ("chamfer_distance", "precision", "recall", "f1_score"):
                self.assertIsNone(merged["geometry"][key])

    def test_cli_imports_existing_results_to_results_folder_without_source_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, summary = source_run(root, "rangda")
            retry, _ = mesh_retry(model, summary)
            source_before = (model / "research/summary.json").read_bytes()
            retry_before = (retry / "result.json").read_bytes()
            results = root / "research_results/results.csv"
            argv = ["export_research_report.py", "--mesh-retries", str(retry / "result.json"),
                    "--results-csv", str(results), "--pairs", "1"]
            with patch("sys.argv", argv), patch("subprocess.Popen", side_effect=AssertionError("No process")):
                main()
            self.assertEqual(len(list(results.parent.glob("laporan_pemulihan_*"))), 1)
            self.assertTrue(results.with_name("results_ringkasan.csv").is_file())
            self.assertFalse(results.exists())
            self.assertEqual((model / "research/summary.json").read_bytes(), source_before)
            self.assertEqual((retry / "result.json").read_bytes(), retry_before)

    def test_incompatible_combined_recovery_table_preserves_other_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, summary = source_run(root, "tualen")
            results = root / "results.csv"
            automatic_report(model, results)
            retry, _ = mesh_retry(model, summary)
            report = export([], root / "recovery", mesh_retries=[retry])
            broken = results.with_name("results_pemulihan_mesh.csv")
            broken.write_text("unrelated,columns\n1,2\n")
            tables = [results.with_name("results_laporan.csv"), results.with_name("results_ringkasan.csv"), broken]
            before = [p.read_bytes() for p in tables]
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                update_result_tables(report, results)
            self.assertEqual([p.read_bytes() for p in tables], before)

    def test_duplicate_retry_arguments_create_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, summary = source_run(root, "tualen")
            retry, _ = mesh_retry(model, summary)
            destination = root / "duplicate"
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                export([], destination, mesh_retries=[retry, retry])
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
