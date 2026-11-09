"""CPU checks for the optional Scaffold anchor initialization experiment."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData, PlyElement
import yaml

from research_pipeline.commands import commands
from research_pipeline.config import DEFAULTS, REPO, validate
from research_pipeline.results import experiment_id
from research_pipeline.run_all import main as queue_main
from research_pipeline.runner import execute_scene, ProcessResult
from research_pipeline.scene_discovery import Scene
from research_pipeline.scaffold_initialization import (
    METHOD, export_anchors, locations, main, read_rgb_ply, split_manifest, verify_export,
)
from research_pipeline.reporting import resource_rows


def scene_fixture(root):
    source = root / "source/patung"
    sparse = source / "sparse/0"
    sparse.mkdir(parents=True)
    (source / "images").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 2 2 1 1 1 1\n")
    lines = []
    for index in range(9):
        name = f"{index:03d}.jpg"
        (source / "images" / name).write_bytes(b"photo fixture")
        lines.append(f"{index + 1} 1 0 0 0 0 0 1 1 {name}\n\n")
    (sparse / "images.txt").write_text("".join(lines))
    (sparse / "points3D.txt").write_text(
        "1 0 0 0 255 0 0 0\n2 10 0 0 0 0 255 0\n")
    return source


def anchor_fixture(path, xyz, text=False):
    vertex = np.zeros(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_anchor_feat_0", "f4"), ("f_offset_0", "f4"), ("opacity", "f4")])
    for index, key in enumerate(("x", "y", "z")):
        vertex[key] = np.asarray(xyz)[:, index]
    # Scaffold saves a placeholder opacity, not a learned Gaussian confidence.
    vertex["opacity"] = -100
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(str(path))
    return path


def enabled_config(root):
    config = copy.deepcopy(DEFAULTS)
    config["dataset"]["root"] = str(root / "source")
    config["output"]["root"] = str(root / "output")
    config["results"]["csv"] = str(root / "results.csv")
    config["scaffold_initialization"].update(
        enabled=True, repository=str(root / "Scaffold-GS"), python_executable=sys.executable,
        expected_commit="")
    return config


class AnchorExportTests(unittest.TestCase):
    def test_binary_anchor_export_preserves_coordinates_colors_cameras_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            before = {p.relative_to(source): p.read_bytes()
                      for p in source.rglob("*") if p.is_file()}
            anchors = anchor_fixture(root / "anchors.ply", [
                [.1, 0, 0], [9.9, 0, 0], [9.9, 0, 0], [float("nan"), 0, 0], [100, 0, 0]])
            output = root / "output/patung"
            report = export_anchors(source, anchors, output, max_color_distance=.2)
            self.assertEqual(report["point_count"], 2)
            self.assertEqual(report["removed_nonfinite"], 1)
            self.assertEqual(report["removed_duplicates"], 1)
            self.assertEqual(report["removed_by_color_distance"], 1)
            xyz, rgb = read_rgb_ply(Path(report["initial_ply"]))
            np.testing.assert_allclose(xyz, [[.1, 0, 0], [9.9, 0, 0]], atol=1e-6)
            np.testing.assert_array_equal(rgb, [[255, 0, 0], [0, 0, 255]])
            staged = Path(report["training_source"])
            for filename in ("cameras.txt", "images.txt", "points3D.txt"):
                self.assertEqual((staged / "sparse/0" / filename).read_bytes(),
                                 (source / "sparse/0" / filename).read_bytes())
            after = {p.relative_to(source): p.read_bytes()
                     for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(verify_export(output, source)["point_count"], 2)
            with Path(report["initial_ply"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_export(output, source)

    def test_ascii_anchor_export_and_default_distance_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            anchors = anchor_fixture(root / "anchors.ply", [[0, 0, 0], [100, 0, 0]], text=True)
            report = export_anchors(source, anchors, root / "output")
            self.assertEqual(report["point_count"], 2)
            self.assertEqual(report["removed_by_color_distance"], 0)
            with self.assertRaises(ValueError):
                export_anchors(source, anchors, root / "output")

    def test_invalid_clouds_thresholds_and_camera_models_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            one = anchor_fixture(root / "one.ply", [[0, 0, 0]])
            with self.assertRaisesRegex(ValueError, "At least two"):
                export_anchors(source, one, root / "output")
            valid = anchor_fixture(root / "valid.ply", [[.1, 0, 0], [10, 0, 0]])
            for threshold in (-1, float("nan"), float("inf")):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    export_anchors(source, valid, root / "output", threshold)
            with self.assertRaisesRegex(ValueError, "fewer than two"):
                export_anchors(source, valid, root / "output", .001)
            (source / "sparse/0/cameras.txt").write_text("1 OPENCV 2 2 1 1 1 1 0 0 0 0\n")
            with self.assertRaisesRegex(ValueError, "undistorted"):
                split_manifest(source)
            self.assertFalse((root / "output").exists())

    def test_regular_xyz_rgb_is_not_misidentified_as_a_scaffold_model(self):
        from research_pipeline.scaffold_initialization import write_rgb_ply
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            ordinary = root / "plain.ply"
            write_rgb_ply(ordinary, np.array([[0, 0, 0], [10, 0, 0]]),
                          np.array([[255, 0, 0], [0, 0, 255]]))
            with self.assertRaisesRegex(ValueError, "Scaffold anchor PLY"):
                export_anchors(source, ordinary, root / "output")


    def test_binary_colmap_model_and_cached_source_ply(self):
        import struct
        from research_pipeline.scaffold_initialization import write_rgb_ply
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            sparse = source / "sparse/0"
            with (sparse / "cameras.bin").open("wb") as stream:
                stream.write(struct.pack("<QiiQQdddd", 1, 1, 1, 2, 2, 1, 1, 1, 1))
            with (sparse / "images.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 9))
                for index in range(9):
                    stream.write(struct.pack("<idddddddi", index + 1, 1, 0, 0, 0, 0, 0, 1, 1))
                    stream.write(f"{index:03d}.jpg".encode() + b"\x00")
                    stream.write(struct.pack("<Q", 0))
            with (sparse / "points3D.bin").open("wb") as stream:
                stream.write(struct.pack("<Q", 2))
                for index, x in enumerate((0, 10)):
                    stream.write(struct.pack("<QdddBBBdQ", index + 1, x, 0, 0, 255, 0, 0, 0, 0))
            anchors = anchor_fixture(root / "anchors.ply", [[.1, 0, 0], [9.9, 0, 0]])
            report = export_anchors(source, anchors, root / "first")
            self.assertTrue(report["reference_cloud"].endswith("points3D.bin"))
            self.assertEqual(report["split"]["test"], ["000.jpg", "008.jpg"])
            # The baseline prefers an existing source PLY over bin/txt.
            write_rgb_ply(sparse / "points3D.ply", np.array([[0, 0, 0], [10, 0, 0]]),
                          np.array([[0, 255, 0], [0, 255, 0]]))
            report = export_anchors(source, anchors, root / "second")
            self.assertTrue(report["reference_cloud"].endswith("points3D.ply"))
            _, rgb = read_rgb_ply(Path(report["initial_ply"]))
            np.testing.assert_array_equal(rgb, [[0, 255, 0], [0, 255, 0]])



class ScaffoldQueueTests(unittest.TestCase):
    def test_commands_and_identity_keep_baseline_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            config = enabled_config(root)
            validate(config)
            output = root / "output/patung"
            built = commands(config, source, output)
            self.assertEqual(list(built)[:3], ["scaffold_training", "scaffold_export", "training"])
            self.assertEqual(built["training"][3], str(locations(output)["dataset"]))
            self.assertIn("--eval", built["scaffold_training"])
            baseline = copy.deepcopy(config)
            baseline["scaffold_initialization"]["enabled"] = False
            legacy = copy.deepcopy(baseline)
            legacy.pop("scaffold_initialization")
            self.assertEqual(experiment_id("patung", baseline), experiment_id("patung", legacy))
            self.assertNotEqual(experiment_id("patung", baseline), experiment_id("patung", config))
            changed = copy.deepcopy(config)
            changed["scaffold_initialization"]["voxel_size"] *= 2
            self.assertNotEqual(experiment_id("patung", config), experiment_id("patung", changed))
            self.assertEqual(commands(baseline, source, output)["training"][3], str(source))

    def test_configuration_rejects_missing_environment_and_test_leakage(self):
        for key, value in (("repository", ""), ("iterations", 0), ("iterations", 1.5),
                           ("voxel_size", 0), ("max_color_distance", -1), ("enabled", "yes")):
            with self.subTest(key=key):
                config = copy.deepcopy(DEFAULTS)
                config["scaffold_initialization"].update(enabled=True, repository="Scaffold-GS")
                config["scaffold_initialization"][key] = value
                with self.assertRaises(ValueError):
                    validate(config)
        config = copy.deepcopy(DEFAULTS)
        config["scaffold_initialization"].update(enabled=True, repository="Scaffold-GS")
        config["training"]["extra_args"] = []
        with self.assertRaisesRegex(ValueError, "--eval"):
            validate(config)

    def test_dry_run_lists_stages_without_processes_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_fixture(root)
            config = enabled_config(root)
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            before = sorted(str(p) for p in root.rglob("*"))
            with patch("subprocess.Popen", side_effect=AssertionError("No execution")), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(queue_main(["--config", str(path), "--dry-run"]), 0)
            self.assertIn("SCAFFOLD TRAIN:", printed.getvalue())
            self.assertIn("SCAFFOLD EXPORT:", printed.getvalue())
            self.assertLess(printed.getvalue().index("SCAFFOLD TRAIN:"),
                            printed.getvalue().index("SCAFFOLD EXPORT:"))
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))

    def test_pipeline_executes_export_before_radegs_and_records_extra_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            output = root / "output/patung"
            config = enabled_config(root)
            config["rendering"]["enabled"] = config["mesh"]["enabled"] = False
            seen = []
            anchors = anchor_fixture(root / "anchors.ply", [[.1, 0, 0], [9.9, 0, 0]])
            def executor(command, log_path, interval):
                seen.append(log_path.stem)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fixture\n")
                if log_path.stem == "scaffold_export":
                    export_anchors(source, anchors, output)
                if log_path.stem == "training":
                    xyz, _ = read_rgb_ply(Path(command[3]) / "sparse/0/points3D.ply")
                    np.testing.assert_allclose(xyz, [[.1, 0, 0], [9.9, 0, 0]], atol=1e-6)
                return ProcessResult(0, 1.5)
            result = execute_scene(config, Scene("patung", source), output, executor)
            self.assertEqual(result["status"], "success", result.get("error_message"))
            self.assertEqual(seen, ["scaffold_training", "scaffold_export", "training"])
            self.assertEqual(result["method"], METHOD)
            self.assertEqual(result["initialization"]["duration_seconds"], 3)
            rows = resource_rows(result, {"gpu_total_peak_gib": None})
            self.assertEqual(rows[0]["Tahap"], "Training Scaffold-GS")
            self.assertEqual(rows[0]["Metode"], METHOD)

    def test_failure_or_missing_export_blocks_radegs(self):
        for failing in ("scaffold_training", "scaffold_export", "missing_export"):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = scene_fixture(root)
                output = root / "output/patung"
                config = enabled_config(root)
                config["rendering"]["enabled"] = config["mesh"]["enabled"] = False
                seen = []
                def executor(command, log_path, interval):
                    seen.append(log_path.stem)
                    return ProcessResult(1 if failing == log_path.stem else 0, 1)
                result = execute_scene(config, Scene("patung", source), output, executor)
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("training", seen)
                self.assertEqual(result["failed_stage"],
                                 "scaffold_training" if failing == "scaffold_training"
                                 else "scaffold_export")

    def test_external_training_entry_point_and_export_with_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = scene_fixture(root)
            repository = root / "fake-scaffold"
            (repository / "arguments").mkdir(parents=True)
            (repository / "scene").mkdir()
            (repository / "arguments/__init__.py").write_text("")
            (repository / "scene/gaussian_model.py").write_text("")
            final = ("ply\nformat ascii 1.0\nelement vertex 2\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float f_anchor_feat_0\nproperty float f_offset_0\n"
                     "end_header\n0.1 0 0 0 0\n9.9 0 0 0 0\n")
            (repository / "torch.py").write_text("# CPU fixture; no GPU kernels are executed.\n")
            script = (
                "import os\nfrom pathlib import Path\nimport types\n"
                "raise RuntimeError('GPU auto-selection preamble must be skipped')\n"
                "import torch\nimport lpips\n"
                "lpips_fn = should_not_construct_a_vgg_model()\n"
                "class ModelParams:\n"
                " def __init__(self,p):\n"
                "  p.add_argument('-s',dest='source_path'); p.add_argument('-m',dest='model_path')\n"
                "  p.add_argument('--images'); p.add_argument('--resolution',type=int)\n"
                "  p.add_argument('--data_device'); p.add_argument('--voxel_size',type=float)\n"
                "  p.add_argument('--eval',action='store_true'); p.add_argument('--white_background',action='store_true')\n"
                " def extract(self,a): return a\n"
                "class OptimizationParams:\n"
                " def __init__(self,p): p.add_argument('--iterations',type=int)\n"
                " def extract(self,a): return a\n"
                "class PipelineParams:\n"
                " def __init__(self,p): pass\n"
                " def extract(self,a): return a\n"
                "network_gui=types.SimpleNamespace(conn=None,try_connect=lambda: None)\n"
                "def safe_state(quiet): assert quiet is False\n"
                "def get_logger(path): return None\n"
                "def training(dataset,opt,pipe,name,tests,saves,checks,checkpoint,debug,wandb=None,logger=None):\n"
                " assert tests == [] and saves == [opt.iterations]\n"
                " assert dataset.eval\n"
                " assert (Path(dataset.source_path)/'images/000.jpg').is_file()\n"
                " out=Path(dataset.model_path)/'point_cloud'/('iteration_'+str(opt.iterations))/'point_cloud.ply'\n"
                " out.parent.mkdir(parents=True)\n"
                f" out.write_text({final!r})\n"
                "if __name__ == '__main__': raise RuntimeError('Post-training evaluation must not execute')\n")
            (repository / "train.py").write_text(script)
            output = root / "output/patung"
            config = enabled_config(root)
            config["scaffold_initialization"].update(repository=str(repository), iterations=20)
            built = commands(config, source, output)
            for stage in ("scaffold_training", "scaffold_export"):
                process = subprocess.run(built[stage], cwd=REPO, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            report = verify_export(output, source)
            self.assertEqual(report["scaffold_training"]["status"], "success")
            self.assertEqual(report["scaffold_training"]["launcher"], "training_only")
            self.assertFalse(report["scaffold_training"]["post_training_evaluation"])
            self.assertEqual(report["split"], report["scaffold_training"]["split"])
            self.assertEqual(report["split"]["test"], ["000.jpg", "008.jpg"])
            self.assertEqual(len(report["split"]["train"]), 7)
            self.assertFalse((source / "sparse/0/points3D.ply").exists())



class ScaffoldPreparationTests(unittest.TestCase):
    def test_launcher_preserves_upstream_optimizer_and_is_python37_compatible(self):
        import ast
        from research_pipeline.scaffold_train import training_module
        source_path = REPO / "eksperimen/Scaffold-GS/train.py"
        if not source_path.is_file():
            self.skipTest("Initialize the official Scaffold submodule for source compatibility checks")
        source = source_path.read_text(encoding="utf-8")
        original = ast.parse(source)
        selected = training_module(source)
        original_training = next(node for node in original.body
                                 if isinstance(node, ast.FunctionDef) and node.name == "training")
        selected_training = next(node for node in selected.body
                                 if isinstance(node, ast.FunctionDef) and node.name == "training")
        self.assertEqual(ast.dump(original_training), ast.dump(selected_training))
        assigned = [target.id for node in selected.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)]
        self.assertNotIn("lpips_fn", assigned)
        self.assertFalse(any(isinstance(node, ast.Import) and any(
            alias.name == "lpips" for alias in node.names) for node in selected.body))
        launcher = (REPO / "research_pipeline/scaffold_train.py").read_text(encoding="utf-8")
        ast.parse(launcher, feature_version=(3, 7))

    def test_training_receipt_blocks_changed_inputs_and_failed_runs(self):
        from research_pipeline.scaffold_initialization import digest, input_fingerprint
        from research_pipeline.utils import write_json
        for issue in ("changed_photo", "changed_anchor", "failed_training"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = scene_fixture(root)
                anchors = anchor_fixture(root / "anchors.ply", [[0, 0, 0], [10, 0, 0]])
                output = root / "output"
                split = split_manifest(source)
                receipt = {"status": "success", "split": split,
                           "source_fingerprint": input_fingerprint(source, split),
                           "anchors_sha256": digest(anchors)}
                if issue == "changed_photo":
                    (source / "images/000.jpg").write_bytes(b"changed photograph")
                elif issue == "changed_anchor":
                    anchor_fixture(anchors, [[1, 0, 0], [10, 0, 0]])
                else:
                    receipt["status"] = "failed"
                write_json(locations(output)["training"], receipt)
                with self.assertRaisesRegex(ValueError, "receipt"):
                    export_anchors(source, anchors, output)
                self.assertFalse(locations(output)["dataset"].exists())

    def test_unset_interpreter_and_wrong_git_revision_fail_clearly(self):
        from types import SimpleNamespace
        from research_pipeline.scaffold_initialization import resolve_python, repository_state
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SCAFFOLD_PYTHON"):
                resolve_python("$" + "{SCAFFOLD_PYTHON}")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "arguments").mkdir()
            (repo / "scene").mkdir()
            for filename in ("train.py", "arguments/__init__.py", "scene/gaussian_model.py"):
                (repo / filename).write_text("")
            responses = [SimpleNamespace(stdout="a" * 40), SimpleNamespace(stdout="")]
            with patch("subprocess.run", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "commit mismatch"):
                    repository_state(repo, "b" * 40)

    def test_environment_probe_reports_missing_python_without_installing_anything(self):
        from research_pipeline.scaffold_environment import probe_python
        with patch("subprocess.run", side_effect=AssertionError("No process should be launched")):
            result = probe_python("nonexistent-scaffold-python-0123456", "scaffold")
        self.assertFalse(result["ready"])
        self.assertIn("executable not found", result["error"])



if __name__ == "__main__":
    unittest.main()
