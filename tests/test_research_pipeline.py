"""CPU-only tests: real subprocesses use tiny Python fixtures, never RaDe-GS."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml
from research_pipeline.config import DEFAULTS, validate
from research_pipeline.commands import commands
from research_pipeline.scene_discovery import (
    discover, start_validation_errors, validation_errors,
)
from research_pipeline.run_all import main

def make_scene(root: Path, name: str, suffix: str = ".bin") -> Path:
    scene = root / name
    (scene / "images").mkdir(parents=True)
    (scene / "sparse/0").mkdir(parents=True)
    for stem in ("cameras", "images", "points3D"):
        (scene / "sparse/0" / (stem + suffix)).write_bytes(b"fixture")
    return scene

def make_raw_scene(root: Path, name: str) -> Path:
    scene = root / name
    (scene / "input").mkdir(parents=True)
    (scene / "input/00001.jpg").write_bytes(b"fixture")
    return scene

class Phase1(unittest.TestCase):
    def test_discovery_validation_commands_and_read_only_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "Dataset Togog Bali"
            make_scene(data, "togog_ganesha")
            make_scene(data, "togog_arjuna", ".txt")
            config = copy.deepcopy(DEFAULTS)
            config["dataset"]["root"] = str(data)
            config["output"]["root"] = str(root / "outputs")
            config["results"]["csv"] = str(root / "results.csv")
            cfg = root / "config.yaml"
            cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = sorted(str(p) for p in root.rglob("*"))
            scenes = discover(data)
            self.assertEqual([s.name for s in scenes], ["togog_arjuna", "togog_ganesha"])
            self.assertFalse(validation_errors(scenes[0]))
            cmd = commands(config, scenes[0].path, root / "outputs")["mesh_extraction"]
            self.assertEqual(cmd[1:], ["mesh_extract_tnt.py", "-m", str(root / "outputs"),
                "-r", "2", "--num_cluster", "8", "--depth_ratio", "0.5", "--voxel_size", "0.009", "--max_depth", "8.0"])
            for mode in ("--list-scenes", "--dry-run"):
                with patch("subprocess.Popen", side_effect=AssertionError("No processes allowed")), contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(["--config", str(cfg), mode]), 0)
                self.assertIn("togog_arjuna", out.getvalue())
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))
            self.assertNotIn("torch", sys.modules)
            (data / "invalid").mkdir()
            self.assertTrue(validation_errors(discover(data, "invalid")[0]))
    def test_config_rejects_overrides_and_bad_values(self):
        for section, key, value in [
            ("mesh", "voxel_size", 0), ("mesh", "max_depth", float("nan")),
            ("mesh", "extra_args", ["--voxel_size=0.1"]),
            ("mesh", "extra_args", ["--max_depth", "20"]),
            ("mesh", "depth_ratio", 0.6), ("mesh", "extra_args", ["-r4"]),
            ("training", "extra_args", ["--model_path=elsewhere"]),
            ("training", "extra_args", ["--nonexistent"]),
            ("resource_monitor", "interval_seconds", 0),
            ("resource_guard", "enabled", "false"),
            ("colmap", "executable", ""),
            ("colmap", "use_gpu", "true")]:
            config = copy.deepcopy(DEFAULTS)
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate(config)

    def test_colmap_start_accepts_raw_input_and_dry_run_shows_both_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            raw = make_raw_scene(data, "togog")
            scene = discover(data)[0]
            self.assertTrue(start_validation_errors(scene, "training"))
            self.assertFalse(start_validation_errors(scene, "colmap"))

            config = copy.deepcopy(DEFAULTS)
            config["dataset"]["root"] = str(data)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            built = commands(config, raw, root / "output/togog", "colmap")
            self.assertEqual(list(built)[:2], ["colmap", "training"])
            self.assertIn("research_pipeline.colmap_preprocessor", built["colmap"])

            cfg = root / "config.yaml"
            cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("subprocess.Popen", side_effect=AssertionError("No processes allowed")), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["--config", str(cfg), "--start-from", "colmap",
                                       "--dry-run"]), 0)
            printed = out.getvalue()
            self.assertIn("COLMAP:", printed)
            self.assertIn("TRAIN:", printed)

    def test_shipped_gc_configs_dry_run_with_tnt_without_model_imports(self):
        from research_pipeline.config import REPO, load_config, repository_parser
        for name, width in (("togog_bali.yaml", 1920),
                            ("togog_bali_1600.yaml", 1600),
                            ("togog_bali_full_gc.yaml", 1024)):
            with self.subTest(config=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = load_config(REPO / "configs" / name)
                if name == "togog_bali.yaml":
                    reference = load_config(REPO / "configs/togog_bali_1600.yaml")
                    expected_training = copy.deepcopy(reference["training"])
                    resolution_index = expected_training["extra_args"].index("--resolution")
                    expected_training["extra_args"][resolution_index + 1] = "1920"
                    self.assertEqual(config["training"], expected_training)
                    self.assertEqual(config["mesh"], reference["mesh"])
                    self.assertEqual(config["output"]["root"], "output/togog_bali")
                    self.assertEqual(config["results"]["csv"],
                                     "/home/pande/projects/RaDe-GS/research_results/experiment_results.csv")
                    self.assertNotEqual(config["output"], reference["output"])
                    self.assertNotEqual(config["results"], reference["results"])
                scene = make_scene(root / "dataset", "togog_tualen")
                config["dataset"]["root"] = str(scene.parent)
                config["output"]["root"] = str(root / "output")
                config["results"]["csv"] = str(root / "results.csv")
                built = commands(config, scene, root / "output/togog_tualen")
                train = repository_parser("training").parse_args(built["training"][2:])
                self.assertTrue(train.enable_allocator_gc)
                self.assertEqual((train.iterations, train.resolution), (30000, width))
                mesh = repository_parser("mesh").parse_args(built["mesh_extraction"][2:])
                self.assertEqual(built["mesh_extraction"][1], "mesh_extract_tnt.py")
                self.assertEqual((mesh.resolution, mesh.num_cluster, mesh.depth_ratio,
                                  mesh.voxel_size, mesh.max_depth), (2, 8, 0.5, 0.009, 8.0))
                # Default -1 selects the latest saved model, including short custom runs.
                self.assertEqual(mesh.iteration, -1)
                cfg = root / "config.yaml"
                cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
                before = sorted(str(p) for p in root.rglob("*"))
                with patch("subprocess.Popen", side_effect=AssertionError("No processes allowed")), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(["--config", str(cfg), "--scene", "togog_tualen",
                                           "--start-from", "training", "--dry-run"]), 0)
                printed = out.getvalue()
                self.assertIn("--enable_allocator_gc", printed)
                self.assertIn("mesh_extract_tnt.py", printed)
                self.assertNotIn("python mesh_extract.py", printed)
                self.assertIn("--voxel_size 0.009 --max_depth 8.0", printed)
                self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))
                self.assertNotIn("torch", sys.modules)
                self.assertNotIn("open3d", sys.modules)

    def test_mutually_exclusive_flags(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--config", "missing", "--resume", "--force"])

    def test_keyboard_interrupt_returns_clean_shell_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            make_scene(data, "togog")
            config = copy.deepcopy(DEFAULTS)
            config["dataset"]["root"] = str(data)
            config["output"]["root"] = str(root / "output")
            config["results"]["csv"] = str(root / "results.csv")
            cfg = root / "config.yaml"
            cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("research_pipeline.runner.run_batch", side_effect=KeyboardInterrupt), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--config", str(cfg)]), 130)

if __name__ == "__main__":
    unittest.main()
