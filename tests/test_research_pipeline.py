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
from research_pipeline.scene_discovery import discover, validation_errors
from research_pipeline.run_all import main

def make_scene(root: Path, name: str, suffix: str = ".bin") -> Path:
    scene = root / name
    (scene / "images").mkdir(parents=True)
    (scene / "sparse/0").mkdir(parents=True)
    for stem in ("cameras", "images", "points3D"):
        (scene / "sparse/0" / (stem + suffix)).write_bytes(b"fixture")
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
            self.assertEqual(cmd[1:], ["mesh_extract.py", "-m", str(root / "outputs"),
                "-r", "2", "--num_cluster", "8", "--depth_ratio", "0.5"])
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
            ("mesh", "depth_ratio", 0.6), ("mesh", "extra_args", ["-r4"]),
            ("training", "extra_args", ["--model_path=elsewhere"]),
            ("training", "extra_args", ["--nonexistent"]),
            ("resource_monitor", "interval_seconds", 0),
            ("resource_guard", "enabled", "false")]:
            config = copy.deepcopy(DEFAULTS)
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate(config)
    def test_mutually_exclusive_flags(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--config", "missing", "--resume", "--force"])

if __name__ == "__main__":
    unittest.main()
