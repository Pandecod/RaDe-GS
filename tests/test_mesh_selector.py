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
    def test_each_script_gets_parameters_and_its_own_cli_validation(self):
        config = copy.deepcopy(DEFAULTS)
        for script in ("mesh_extract.py", "mesh_extract_tnt.py"):
            config["mesh"]["script"] = script
            config["mesh"]["voxel_size"] = .012
            config["mesh"]["max_depth"] = 6.25
            config["mesh"]["extra_args"] = []
            validate(config)
            command = commands(config, Path("data"), Path("output"))["mesh_extraction"]
            self.assertEqual(command[1], script)
            args = repository_parser("mesh", script).parse_args(command[2:])
            self.assertEqual((args.voxel_size, args.max_depth), (.012, 6.25))
        config["mesh"]["script"] = "mesh_extract.py"
        config["mesh"]["extra_args"] = ["--use_depth_filter"]
        with self.assertRaises(ValueError):
            validate(config)
        config["mesh"]["extra_args"] = []
        config["mesh"]["script"] = "../other.py"
        with self.assertRaises(ValueError):
            validate(config)

    def test_historical_tnt_id_is_preserved_but_vbg_is_a_distinct_experiment(self):
        config = copy.deepcopy(DEFAULTS)
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
            self.assertIn("--voxel_size 0.009 --max_depth 7.0", printed)
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))

    def test_pipeline_executes_selected_vbg_and_records_its_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("ngawasari", make_scene(root / "data", "ngawasari"))
            output = root / "out"
            config = copy.deepcopy(DEFAULTS)
            config["mesh"].update(script="mesh_extract.py", max_depth=7.)
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
            self.assertEqual(result["mesh_extraction"]["max_depth"], 7.)

    def test_vbg_function_really_uses_passed_voxel_and_both_depth_limits(self):
        tree = ast.parse((REPO / "mesh_extract.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_mesh")
        captures, calls, written = {}, [], []
        class Mesh:
            def to_legacy(self):
                return self
        class Volume:
            def __init__(self, **kwargs):
                captures.update(kwargs)
            def compute_unique_block_coordinates(self, *args):
                calls.append(("blocks", args[-1]))
                return []
            def integrate(self, *args):
                calls.append(("integrate", args[-1]))
            def extract_triangle_mesh(self, **kwargs):
                return Mesh()
        class Image:
            def __init__(self, value):
                pass
            def to(self, device):
                return self
        camera = SimpleNamespace(Fx=2., Fy=2., Cx=1., Cy=1., gt_mask=None,
                                 world_view_transform=Array(np.eye(4)))
        o3d = SimpleNamespace(
            core=SimpleNamespace(Device=lambda value: value, Tensor=lambda value: value),
            t=SimpleNamespace(geometry=SimpleNamespace(VoxelBlockGrid=Volume, Image=Image)),
            io=SimpleNamespace(write_triangle_mesh=lambda path, mesh: written.append(Path(path).name)))
        scope = {
            "math": math, "np": np, "os": __import__("os"),
            "o3d": o3d, "o3c": SimpleNamespace(float32=None),
            "torch": SimpleNamespace(float32=None, tensor=lambda *a, **k: None,
                                     clamp=lambda value, **k: value,
                                     cuda=SimpleNamespace(empty_cache=lambda: None)),
            "GaussianModel": lambda *a: None,
            "Scene": lambda *a, **k: SimpleNamespace(getTrainCameras=lambda: [camera]),
            "render": lambda *a, **k: {"render": Array(np.zeros((3, 2, 2))),
                                       "median_depth": Array(np.ones((1, 2, 2)))},
            "post_process_mesh": lambda mesh, count: mesh,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "mesh_extract.py", "exec"), scope)
        dataset = SimpleNamespace(sh_degree=3, kernel_size=0., depth_ratio=.5, model_path="output")
        scope["extract_mesh"](dataset, None, 30000, 8, .012, 6.25)
        self.assertEqual(captures["voxel_size"], .012)
        self.assertEqual(calls, [("blocks", 6.25), ("integrate", 6.25)])
        self.assertEqual(written, ["recon_raw.ply", "recon_post.ply"])
        with self.assertRaises(ValueError):
            scope["extract_mesh"](dataset, None, 30000, 8, .012, 0.)


if __name__ == "__main__":
    unittest.main()
