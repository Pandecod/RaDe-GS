"""CPU tests for automatic research reports; no model or CUDA imports."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from research_pipeline.config import DEFAULTS
from research_pipeline.reporting import (
    export, read_csv, read_json, memory_info, ply_count, update_report_table,
)
from research_pipeline.runner import run_batch
from research_pipeline.scene_discovery import Scene
from research_pipeline.utils import write_json


def fixture(root, name="patung", count=7):
    output = root / name
    research = output / "research"
    research.mkdir(parents=True)
    summary = {
        "scene_name": name, "output_path": str(output), "experiment_id": name,
        "timestamp": "2026-10-29T04:18:45+00:00", "status": "success",
        "failed_stage": None, "error_message": None, "warnings": [],
        "training": {"duration_seconds": 120, "final_iteration": 30000,
                     "training_camera_count": count, "test_camera_count": 0},
        "rendering": {"split": "train", "method": "ours_30000",
                      "evaluation_image_count": count, "psnr": 25, "ssim": .8, "lpips": .1},
    }
    write_json(research / "summary.json", summary)
    folder = output / "train/ours_30000"
    for sub in ("renders", "gt"):
        (folder / sub).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (24, 12), (index * 10, 40, 60)).save(folder / sub / f"{index:05d}.png")
    write_json(folder / "render_manifest.json", {
        "split": "train", "iteration": 30000, "images": [
            {"file": f"{i:05d}.png", "camera_image_name": f"IMG_{99-i}",
             "width": 24, "height": 12} for i in range(count)]})
    ply = output / "point_cloud/iteration_30000/point_cloud.ply"
    ply.parent.mkdir(parents=True)
    props = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "f_dc_0"]
    ply.write_bytes(("ply\nformat binary_little_endian 1.0\nelement vertex 17\n" +
        "".join(f"property float {p}\n" for p in props) + "end_header\n").encode() + bytes(17*32))
    profiler = research / "profiling"
    profiler.mkdir()
    write_json(profiler / "manifest_run.json", {"run_id": "run"})
    write_json(profiler / "cameras_run.json", {"initial_active_gaussians": 3,
                                             "cameras": [{} for _ in range(count)]})
    (profiler / "gpu_run.csv").write_text(
        "device_status,gpu_uuid,device_vram_used_mb\navailable,GPU-one,1024\navailable,GPU-one,2048\n")
    (profiler / "training_run.csv").write_text(
        "run_id,cuda_peak_allocated_bytes,cuda_peak_reserved_bytes\nrun,1073741824,1610612736\n")
    return output, summary


class ReportingTests(unittest.TestCase):
    def test_counts_units_photo_mapping_and_unchanged_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, _ = fixture(root)
            destination = export([output], root / "report")
            row = read_csv(destination / "tabel_evaluasi.csv")[0]
            self.assertEqual(row["Waktu ringkasan (UTC+08)"], "2026-10-29 12:18:45")
            for field, value in {"Training (menit)": "2.0", "Peak VRAM seluruh GPU (GiB)": "2.0",
                                 "Jumlah foto terdaftar": "7", "Foto test": "0",
                                 "Gaussian awal": "3", "Splats PLY akhir": "17",
                                 "Iterasi PLY": "30000"}.items():
                self.assertEqual(row[field], value, field)
            photos = read_csv(destination / "01_patung_daftar_foto.csv")
            self.assertEqual(photos[0]["Nama foto kamera"], "IMG_99")
            self.assertEqual(photos[-1]["Nama foto kamera"], "IMG_93")
            pictures = sorted((destination / "01_patung").glob("*.png"))
            self.assertEqual([p.name.split("_")[0] for p in pictures],
                             ["00000", "00002", "00003", "00004", "00006"])
            with Image.open(pictures[-1]) as pair:
                self.assertEqual(pair.size, (48, 60))
                self.assertEqual(pair.getpixel((0, 48)), (60, 40, 60))
                self.assertEqual(pair.getpixel((24, 48)), (60, 40, 60))
            details = read_json(destination / "detail_laporan.json")["experiments"][0]
            self.assertEqual(details["memory"]["pytorch_allocated_peak_gib"], 1)
            with self.assertRaises(FileExistsError):
                export([output], destination)

    def test_missing_files_not_invented_and_no_mesh_count_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, summary = fixture(root)
            # A copied research bundle may lack all render images and the final Gaussian PLY.
            bundle = root / "bundle"
            (bundle / "research").mkdir(parents=True)
            write_json(bundle / "research/summary.json", summary)
            (bundle / "recon.ply").write_text("ply\nelement vertex 9999\nend_header\n")
            dest = export([bundle], root / "report")
            row = read_csv(dest / "tabel_evaluasi.csv")[0]
            self.assertEqual(row["Splats PLY akhir"], "")
            self.assertEqual(row["Peak VRAM seluruh GPU (GiB)"], "")
            details = read_json(dest / "detail_laporan.json")["experiments"][0]
            self.assertEqual(details["comparison"]["status"], "missing_images")
            self.assertEqual(read_csv(dest / "01_patung_daftar_foto.csv"), [])

    def test_reject_mismatched_manifest_and_missing_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, _ = fixture(root)
            path = output / "train/ours_30000/render_manifest.json"
            data = read_json(path)
            data["iteration"] = 7000
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "does not match"):
                export([output], root / "bad")
            data["iteration"] = 30000
            write_json(path, data)
            (output / "train/ours_30000/gt/00000.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing GT"):
                export([output], root / "missing")

    def test_gpu_scope_and_profiler_run_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = fixture(Path(tmp))
            profiler = output / "research/profiling"
            with (profiler / "gpu_run.csv").open("a") as stream:
                stream.write("available,GPU-other,5000\n")
            self.assertIsNone(memory_info(output)["gpu_total_peak_gib"])
            write_json(profiler / "manifest_other.json", {"run_id": "other"})
            with self.assertRaisesRegex(ValueError, "Multiple profiler"):
                memory_info(output)

    def test_pipeline_automatically_reports_and_retains_repeated_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            scene = Scene("patung", root / "data/patung")
            def execute(_config, _scene, output):
                _, summary = fixture(output.parent, output.name)
                summary["scene_name"] = scene.name
                return summary
            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                 patch("research_pipeline.runner.execute_scene", side_effect=execute):
                self.assertEqual(run_batch(config, [scene]), 0)
                self.assertEqual(run_batch(config, [scene]), 0)
            rows = read_csv(root / "results_laporan.csv")
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["Folder output"], rows[1]["Folder output"])
            for row in rows:
                summary = read_json(Path(row["Folder output"]) / "research/summary.json")
                self.assertEqual(summary["reporting"]["status"], "complete")
                self.assertTrue(Path(summary["reporting"]["directory"]).is_dir())
            update_report_table(root / "results_laporan.csv", [rows[0]])
            self.assertEqual(len(read_csv(root / "results_laporan.csv")), 2)

    def test_report_error_is_visible_and_keeps_training_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            def execute(_config, scene, output):
                return fixture(output.parent, output.name)[1]
            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                 patch("research_pipeline.runner.execute_scene", side_effect=execute), \
                 patch("research_pipeline.reporting.automatic_report", side_effect=OSError("disk full")):
                self.assertEqual(run_batch(config, [Scene("patung", root / "data")]), 1)
            summary = read_json(root / "output/patung/research/summary.json")
            self.assertEqual(summary["failed_stage"], "report_generation")
            self.assertEqual(summary["training"]["final_iteration"], 30000)
            self.assertEqual(summary["reporting"]["status"], "failed")

    def test_render_records_actual_order_without_changing_images(self):
        # Execute only render_set with CPU stand-ins; never import torch/render.py.
        import numpy as np
        from research_pipeline.utils import write_json
        tree = ast.parse(Path("render.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_set")
        def save(array, path):
            Image.fromarray(array.transpose(1, 2, 0)).save(path)
        scope = {"os": __import__("os"), "Path": Path, "write_json": write_json,
                 "makedirs": __import__("os").makedirs, "tqdm": lambda x, **k: x,
                 "render": lambda view, *args, **kwargs: {"render": view.original_image},
                 "torchvision": SimpleNamespace(utils=SimpleNamespace(save_image=save))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "render.py", "exec"), scope)
        with tempfile.TemporaryDirectory() as tmp:
            views = [SimpleNamespace(image_name=name, original_image=np.full((3, 12, 24), i, dtype=np.uint8))
                     for i, name in enumerate(["IMG_z", "IMG_a"])]
            scope["render_set"](tmp, "train", 30000, views, None, None, None, None)
            manifest = read_json(Path(tmp) / "train/ours_30000/render_manifest.json")
            self.assertEqual([r["camera_image_name"] for r in manifest["images"]], ["IMG_z", "IMG_a"])
            self.assertEqual(manifest["images"][1]["width"], 24)
            with Image.open(Path(tmp) / "train/ours_30000/gt/00001.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (1, 1, 1))


    def test_partial_report_failure_corrects_all_written_statuses(self):
        from research_pipeline import reporting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(DEFAULTS)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            original = reporting.update_report_table
            failed_once = False
            def fail_after_write(path, rows):
                nonlocal failed_once
                original(path, rows)
                if path.name == "results_laporan.csv" and not failed_once:
                    failed_once = True
                    raise OSError("simulated failure after report table write")
            def execute(_config, scene, output):
                return fixture(output.parent, output.name)[1]
            with patch("research_pipeline.results.reproducibility", return_value=({}, [])), \
                 patch("research_pipeline.runner.execute_scene", side_effect=execute), \
                 patch("research_pipeline.reporting.update_report_table", side_effect=fail_after_write):
                self.assertEqual(run_batch(config, [Scene("patung", root / "data")]), 1)
            summary = read_json(root / "output/patung/research/summary.json")
            self.assertEqual(summary["failed_stage"], "report_generation")
            folder = Path(summary["reporting"]["directory"])
            self.assertEqual(read_csv(folder / "tabel_evaluasi.csv")[0]["Status"], "failed")
            self.assertEqual(read_csv(root / "results_laporan.csv")[0]["Status"], "failed")
            self.assertEqual(read_csv(root / "results.csv")[0]["status"], "failed")
            detail = read_json(folder / "detail_laporan.json")["experiments"][0]
            self.assertEqual(detail["table_row"]["Status"], "failed")
            self.assertIn("simulated failure", detail["report_failure"])



if __name__ == "__main__":
    unittest.main()
