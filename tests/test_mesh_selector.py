"""CPU regressions for YAML-selected mesh_extract.py versus TNT."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from research_pipeline.commands import commands
from research_pipeline.config import DEFAULTS, REPO, load_config, repository_parser, validate
from research_pipeline.results import experiment_id
from research_pipeline.run_all import main
from research_pipeline.runner import ProcessResult, execute_scene
from research_pipeline.scene_discovery import Scene
from test_research_pipeline import make_scene


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)
    @property
    def T(self):
        return Array(self.data.T)
    def cpu(self):
        return self
    def numpy(self):
        return self.data
    def __getitem__(self, key):
        return Array(self.data[key])


class MeshSelectorTests(unittest.TestCase):
    def test_original_cli_and_tnt_cli_are_not_interchanged(self):
        config = copy.deepcopy(DEFAULTS)
        validate(config)
        command = commands(config, Path("data"), Path("out"))["mesh_extraction"]
        repository_parser("mesh", "mesh_extract.py").parse_args(command[2:])
        self.assertNotIn("--voxel_size", command)
        for key, value in (("voxel_size", .002), ("max_depth", 7.0)):
            bad = copy.deepcopy(config)
            bad["mesh"][key] = value
            with self.assertRaises(ValueError):
                validate(bad)
        config["mesh"].update(script="mesh_extract_tnt.py", voxel_size=.012, max_depth=6.25)
        validate(config)
        command = commands(config, Path("data"), Path("out"))["mesh_extraction"]
        args = repository_parser("mesh", "mesh_extract_tnt.py").parse_args(command[2:])
        self.assertEqual((args.voxel_size, args.max_depth), (.012, 6.25))

    def test_historical_tnt_id_is_preserved_but_vbg_is_a_distinct_experiment(self):
        config = copy.deepcopy(DEFAULTS)
        config["mesh"]["script"] = "mesh_extract_tnt.py"
        old_mesh = dict(config["mesh"])
        old_mesh.pop("script")
        payload = {"scene_name": "ngawasari", "dataset": config["dataset"],
                   "colmap": config["colmap"], "training": config["training"],
                   "rendering": config["rendering"], "image_metrics": config["image_metrics"],
                   "mesh": old_mesh, "geometry_evaluation": config["geometry_evaluation"]}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.assertEqual(experiment_id("ngawasari", config), expected)
        self.assertIn("script", config["mesh"])
        config["mesh"]["script"] = "mesh_extract.py"
        self.assertNotEqual(experiment_id("ngawasari", config), expected)

    def test_ngawasari_primary_config_dry_run_does_not_start_or_write_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(REPO / "configs/togog_bali.yaml")
            make_scene(root / "data", "ngawasari")
            make_scene(root / "data", "other_scene")
            config["dataset"]["root"] = str(root / "data")
            config["output"]["root"] = str(root / "out")
            config["results"]["csv"] = str(root / "results.csv")
            path = root / "test.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = sorted(str(p) for p in root.rglob("*"))
            with patch("subprocess.Popen", side_effect=AssertionError("No subprocesses")), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["--config", str(path), "--scene", "ngawasari",
                                       "--start-from", "training", "--dry-run"]), 0)
            printed = out.getvalue()
            self.assertIn("[1/1] ngawasari", printed)
            self.assertNotIn("other_scene", printed)
            self.assertIn("python mesh_extract.py", printed)
            self.assertNotIn("python mesh_extract_tnt.py", printed)
            self.assertIn("--resolution 1920", printed)
            self.assertNotIn("--max_depth", printed)
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))

    def test_pipeline_executes_selected_vbg_and_records_its_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("ngawasari", make_scene(root / "data", "ngawasari"))
            output = root / "out"
            config = copy.deepcopy(DEFAULTS)
            config["mesh"].update(script="mesh_extract.py", max_depth=8.)
            config["resource_guard"]["enabled"] = False
            config["rendering"]["enabled"] = False
            config["image_metrics"]["enabled"] = False
            seen = []
            def executor(command, log, interval):
                seen.append(command[1])
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("completed\n", encoding="utf-8")
                if command[1] == "mesh_extract.py":
                    (output / "recon_post.ply").write_bytes(b"fixture")
                return ProcessResult(0, 2.)
            with patch("research_pipeline.mesh_analyzer.analyze_mesh", return_value={"vertex_count": 3}):
                result = execute_scene(config, scene, output, executor)
            self.assertEqual(seen, ["train.py", "mesh_extract.py"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["mesh_extraction"]["script"], "mesh_extract.py")
            self.assertEqual(result["mesh_extraction"]["max_depth"], 8.)



if __name__ == "__main__":
    unittest.main()
