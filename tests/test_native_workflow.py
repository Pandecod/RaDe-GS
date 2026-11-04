"""CPU-only checks for original source, held-out metrics and external reporting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_pipeline.config import DEFAULTS, REPO, validate
from research_pipeline.image_evaluator import prepare_native_evaluation, collect_native_metrics
from research_pipeline.reporting import memory_info
from research_pipeline.utils import write_json


class NativeWorkflowTests(unittest.TestCase):
    def fixture(self, output):
        (output / "cfg_args").write_text("Namespace(eval=True)")
        for kind in ("renders", "gt"):
            folder = output / "test/ours_30000" / kind
            folder.mkdir(parents=True)
            (folder / "00000.png").write_bytes(b"fixture")
        return {"PSNR": 25.1234567, "SSIM": .8765432, "LPIPS": .1234567}

    def outputs(self, output, values):
        write_json(output / "results.json", {"ours_30000": values})
        write_json(output / "per_view.json", {"ours_30000": {
            key: {"00000.png": value} for key, value in values.items()}})

    def test_exact_native_values_are_imported_without_recalculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            values = self.fixture(output)
            prepare_native_evaluation(output)
            self.outputs(output, values)
            result = collect_native_metrics(output, "scene")
            self.assertEqual({k: result[k.lower()] for k in values}, values)
            self.assertEqual((result["split"], result["evaluator"], result["image_count"]), ("test", "metric.py", 1))
            with self.assertRaisesRegex(ValueError, "already exist"):
                prepare_native_evaluation(output)

    def test_caught_native_error_without_outputs_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.fixture(output)
            with self.assertRaisesRegex(ValueError, "did not produce"):
                collect_native_metrics(output, "scene")

    def test_no_train_fallback_and_no_relabelled_training_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            values = self.fixture(output)
            (output / "test").rename(output / "train")
            with self.assertRaisesRegex(ValueError, "test images"):
                prepare_native_evaluation(output)
            (output / "train").rename(output / "test")
            (output / "cfg_args").write_text("Namespace(eval=False)")
            self.outputs(output, values)
            with self.assertRaisesRegex(ValueError, "eval=True"):
                collect_native_metrics(output, "scene")

    def test_incomplete_per_view_or_nan_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            values = self.fixture(output)
            self.outputs(output, values)
            write_json(output / "per_view.json", {"ours_30000": {"PSNR": {}}})
            with self.assertRaises(ValueError):
                collect_native_metrics(output, "scene")
            self.outputs(output, values)
            (output / "results.json").write_text(json.dumps({"ours_30000": {**values, "SSIM": float("nan")}}))
            with self.assertRaises(ValueError):
                collect_native_metrics(output, "scene")

    def test_configuration_requires_heldout_training_and_rendering(self):
        for stage, args in (("training", ["--resolution", "1920"]), ("rendering", ["--skip_test"])):
            config = copy.deepcopy(DEFAULTS)
            config[stage]["extra_args"] = args
            with self.assertRaises(ValueError):
                validate(config)

    def test_external_training_monitor_supplies_report_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            resource = output / "research/logs/training_resources_example.csv"
            resource.parent.mkdir(parents=True)
            resource.write_text("gpu_uuid,device_status,device_vram_used_mb,process_vram_mb\nGPU1,available,8192,\nGPU1,available,10240,\n")
            write_json(output / "research/summary.json", {"training": {"resource_csv": "/server/logs/" + resource.name}})
            result = memory_info(output)
            self.assertEqual(result["gpu_total_peak_gib"], 10)
            self.assertIsNone(result["pytorch_allocated_peak_gib"])
            self.assertEqual(result["gpu_samples"], 2)
            with resource.open("a") as stream:
                stream.write("GPU2,available,20480,\n")
            self.assertIsNone(memory_info(output)["gpu_total_peak_gib"])

    def test_restored_sources_match_recorded_baseline_with_voxel_exception(self):
        # Baseline d72f207; mesh only differs by user-requested voxel 0.009.
        expected = {'scene/__init__.py': 'c5a41e7361d2418950137daa419233ddc931ac3d84851cbd7b076605d4d92e30', 'utils/camera_utils.py': 'da5ae2a697cd135e22bcf5085bd365571b7de90a519801dea61f92e14f091725', 'mesh_extract.py': 'a403d7a0d4b2a172feca706ceee1afe2455d76570230c6959c12c5c480ab7e90', 'metric.py': '777e05016b645b71cebaf0ec4d404df8f193c1daad2fbaef643390f8982a2347'}
        for name, digest in expected.items():
            with self.subTest(file=name):
                content = (REPO / name).read_text(encoding="utf-8").replace("\r\n", "\n")
                self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), digest)
