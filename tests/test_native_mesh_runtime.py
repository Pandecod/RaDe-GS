"""CPU checks for the opt-in native block layout experiment."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_pipeline.mesh_retry import main as retry_main, retry_mesh
from research_pipeline.native_mesh_runtime import (
    CheckedVolume, SIGNED_INDEX_BYTES, main as runtime_main, mesh_structure_bytes, run_native,
)
from research_pipeline.runner import ProcessResult


class NativeMeshRuntimeTests(unittest.TestCase):
    def test_observed_tualen_buffer_crosses_index_boundary(self):
        self.assertEqual(mesh_structure_bytes(41578, 16), 2724855808)
        self.assertEqual(mesh_structure_bytes(32768, 16), SIGNED_INDEX_BYTES)
        self.assertGreater(mesh_structure_bytes(32769, 16), SIGNED_INDEX_BYTES)
        self.assertEqual(mesh_structure_bytes(262144, 8), SIGNED_INDEX_BYTES)

    def test_guard_stops_before_cpp_and_records_observed_block_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = Mock()
            volume.hashmap.return_value.size.return_value = 41578
            volume.hashmap.return_value.capacity.return_value = 50000
            report = Path(tmp) / "runtime.json"
            checked = CheckedVolume(volume, 16, report, {})
            with self.assertRaisesRegex(RuntimeError, "stopped before"):
                checked.extract_triangle_mesh()
            volume.extract_triangle_mesh.assert_not_called()
            record = json.loads(report.read_text())
            self.assertEqual(record["active_voxel_blocks"], 41578)
            self.assertEqual(record["stage"], "buffer_guard_rejected")

    def test_smaller_layout_delegates_real_extraction_and_keeps_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = Mock()
            volume.hashmap.return_value.size.return_value = 200000
            volume.hashmap.return_value.capacity.return_value = 250000
            checked = CheckedVolume(volume, 8, Path(tmp) / "runtime.json", {})
            self.assertIs(checked.integrate, volume.integrate)
            self.assertIs(checked.extract_triangle_mesh(weight_threshold=3.0),
                          volume.extract_triangle_mesh.return_value)
            volume.extract_triangle_mesh.assert_called_once_with(weight_threshold=3.0)

    def test_size_eight_is_also_guarded_when_still_too_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = Mock()
            volume.hashmap.return_value.size.return_value = 262145
            volume.hashmap.return_value.capacity.return_value = 300000
            checked = CheckedVolume(volume, 8, Path(tmp) / "runtime.json", {})
            with self.assertRaises(RuntimeError):
                checked.extract_triangle_mesh()
            volume.extract_triangle_mesh.assert_not_called()

    def test_runtime_changes_only_block_layout_and_restores_process_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            constructor = Mock(return_value=Mock())
            geometry = SimpleNamespace(VoxelBlockGrid=constructor)
            before_argv = sys.argv
            options = dict(block_resolution=16, block_count=50000, voxel_size=.009,
                           attr_names=("tsdf", "weight", "color"), device="CPU:0")
            native_args = ["-m", "existing_attempt", "-r", "2", "--iteration", "30000"]

            def execute(script, run_name):
                self.assertTrue(script.endswith("mesh_extract.py"))
                self.assertEqual(run_name, "__main__")
                self.assertEqual(sys.argv[1:], native_args)
                volume = geometry.VoxelBlockGrid(**options)
                self.assertIsInstance(volume, CheckedVolume)
                raise RuntimeError("synthetic native failure")

            report = Path(tmp) / "runtime.json"
            with self.assertRaisesRegex(RuntimeError, "synthetic native"):
                run_native(native_args, block_resolution=8, runtime_report=report,
                           geometry=geometry, open3d_version="0.19.0", run_script=execute)
            constructor.assert_called_once_with(**{**options, "block_resolution": 8})
            self.assertIs(geometry.VoxelBlockGrid, constructor)
            self.assertIs(sys.argv, before_argv)
            self.assertEqual(json.loads(report.read_text())["voxel_size"], .009)

    def test_future_open3d_version_is_not_silently_assumed_compatible(self):
        geometry = SimpleNamespace(VoxelBlockGrid=Mock())
        with self.assertRaisesRegex(ValueError, "0.19.0"):
            run_native(["-m", "model"], block_resolution=8, runtime_report=Path("unused"),
                       geometry=geometry, open3d_version="0.20.0")
        geometry.VoxelBlockGrid.assert_not_called()

    def test_runtime_cli_keeps_native_model_and_resolution_arguments(self):
        native_args = ["-m", "attempt", "--iteration", "30000", "-r", "2",
                       "--num_cluster", "8", "--depth_ratio", "0.5"]
        with patch("research_pipeline.native_mesh_runtime.run_native") as run:
            runtime_main(["--block-resolution", "8",
                          "--runtime-report", "attempt/runtime.json", "--", *native_args])
        run.assert_called_once_with(
            native_args, block_resolution=8, runtime_report=Path("attempt/runtime.json"))

    def make_model(self, root):
        model = root / "tualen"
        ply = model / "point_cloud/iteration_30000/point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"original Gaussian model")
        (model / "cfg_args").write_text("original config")
        summary = model / "research/summary.json"
        summary.parent.mkdir()
        summary.write_text('{"status": "failed", "training": {"duration_seconds": 100}}')
        return model

    def test_retry_records_layout_runtime_and_preserves_original_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            originals = {p: p.read_bytes() for p in model.rglob("*") if p.is_file()}

            def execute(command, log, interval):
                self.assertIn("research_pipeline.native_mesh_runtime", command)
                self.assertIn("--block-resolution", command)
                self.assertEqual(command[command.index("--block-resolution") + 1], "8")
                self.assertNotIn("train.py", command)
                self.assertNotIn("--voxel_size", command)
                report = Path(command[command.index("--runtime-report") + 1])
                report.write_text(json.dumps({
                    "stage": "before_extraction", "active_voxel_blocks": 200000,
                    "mesh_structure_bytes": mesh_structure_bytes(200000, 8)}))
                log.write_text("synthetic crash")
                return ProcessResult(-11, 12, {"peak_ram_mb": 5000})

            with patch("research_pipeline.mesh_retry._prepare_model"):
                result = retry_mesh(model, max_depth=8, block_resolution=8, executor=execute)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], -11)
            self.assertEqual(result["settings"]["block_resolution"], 8)
            self.assertEqual(len(result["native_runtime_sha256"]), 64)
            self.assertEqual(result["native_runtime"]["active_voxel_blocks"], 200000)
            with (Path(result["attempt_path"]) / "result.csv").open(
                    encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["mesh_block_resolution"], "8")
            self.assertEqual(row["mesh_active_voxel_blocks"], "200000")
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_layout_is_native_only_and_invalid_settings_create_no_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            before = list((model / "research").iterdir())
            for value in (0, 4, 12):
                with self.assertRaises(ValueError):
                    retry_mesh(model, max_depth=8, block_resolution=value)
            with self.assertRaisesRegex(ValueError, "only"):
                retry_mesh(model, max_depth=8, block_resolution=8, script="mesh_extract_tnt.py")
            self.assertEqual(list((model / "research").iterdir()), before)

    def test_retry_cli_exposes_layout_and_preserves_failure_exit_status(self):
        with patch("research_pipeline.mesh_retry.retry_mesh", return_value={"status": "failed"}) as run:
            self.assertEqual(retry_main(["--model", "model", "--max-depth", "8.0",
                                         "--block-resolution", "8"]), 1)
            self.assertEqual(run.call_args.kwargs["block_resolution"], 8)


@unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D is not installed")
class Open3DLayoutIntegrationTests(unittest.TestCase):
    def test_both_layouts_extract_the_same_synthetic_depth_plane(self):
        import numpy as np
        import open3d as o3d
        if o3d.__version__ != "0.19.0":
            self.skipTest("Runtime experiment is scoped to Open3D 0.19.0")
        depth = o3d.t.geometry.Image(np.full((24, 32), 2, dtype=np.float32))
        color = o3d.t.geometry.Image(np.full((24, 32, 3), .5, dtype=np.float32))
        intrinsic = o3d.core.Tensor(
            [[28., 0, 15.5], [0, 28., 11.5], [0, 0, 1]], dtype=o3d.core.float64)
        extrinsic = o3d.core.Tensor(np.eye(4), dtype=o3d.core.float64)
        for resolution in (16, 8):
            volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3d.core.float32,) * 3, attr_channels=(1, 1, 3),
                voxel_size=.009, block_resolution=resolution, block_count=2048,
                device=o3d.core.Device("CPU:0"))
            blocks = volume.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, 1., 8.)
            for _ in range(4):
                volume.integrate(blocks, depth, color, intrinsic, extrinsic, 1., 8.)
            with tempfile.TemporaryDirectory() as tmp:
                checked = CheckedVolume(volume, resolution, Path(tmp) / "runtime.json", {})
                mesh = checked.extract_triangle_mesh()
            self.assertGreater(mesh.triangle.indices.shape[0], 0)
            self.assertTrue(np.allclose(mesh.vertex.positions.numpy()[:, 2], 2., atol=1e-4))


if __name__ == "__main__":
    unittest.main()
