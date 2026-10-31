"""CPU-only ordered multi-config queue checks."""
import contextlib
import copy
import io
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_pipeline.config import DEFAULTS, REPO, load_config
from research_pipeline.run_all import main
from test_research_pipeline import make_scene


class ConfigQueueTests(unittest.TestCase):
    def make_configs(self, root):
        paths = []
        for name in ("togog", "tanks"):
            data = root / name / "dataset"
            make_scene(data, name + "_scene")
            config = copy.deepcopy(DEFAULTS)
            config["project"]["dataset"] = name
            config["dataset"]["root"] = str(data)
            config["output"]["root"] = str(root / name / "output")
            config["results"]["csv"] = str(root / name / "results.csv")
            path = root / (name + ".yaml")
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            paths.append(str(path))
        return paths

    def test_configs_run_in_order_even_when_first_has_failed_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_configs(Path(tmp))
            seen = []
            def run(config, scenes, **kwargs):
                seen.append((config["project"]["dataset"], scenes[0].name, kwargs))
                return 1 if len(seen) == 1 else 0
            with patch("research_pipeline.runner.run_batch", side_effect=run):
                self.assertEqual(main(["--config", *paths, "--resume"]), 1)
            self.assertEqual([s[0] for s in seen], ["togog", "tanks"])
            self.assertTrue(all(s[2]["resume"] for s in seen))
            self.assertTrue(all(s[2]["start_from"] == "training" for s in seen))

    def test_dry_run_both_configs_in_order_without_processes_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_configs(root)
            before = sorted(str(p) for p in root.rglob("*"))
            with patch("subprocess.Popen", side_effect=AssertionError("No subprocesses")), \
                 patch("research_pipeline.runner.run_batch", side_effect=AssertionError("No execution")), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["--config", *paths, "--dry-run"]), 0)
            printed = out.getvalue()
            self.assertLess(printed.index("togog_scene"), printed.index("tanks_scene"))
            self.assertEqual(printed.count("TRAIN:"), 2)
            self.assertEqual(printed.count("MESH:"), 2)
            self.assertEqual(before, sorted(str(p) for p in root.rglob("*")))

    def test_critical_error_or_interrupt_stops_before_next_config(self):
        for failure, code in ((RuntimeError("master CSV unavailable"), 2), (KeyboardInterrupt(), 130)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                paths = self.make_configs(Path(tmp))
                with patch("research_pipeline.runner.run_batch", side_effect=failure) as run:
                    self.assertEqual(main(["--config", *paths]), code)
                    run.assert_called_once()

    def test_invalid_second_config_keeps_error_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_configs(root)
            Path(paths[1]).write_text("invalid: true\n", encoding="utf-8")
            with patch("research_pipeline.runner.run_batch", return_value=0) as run:
                self.assertEqual(main(["--config", *paths]), 2)
                run.assert_called_once()

    def test_tanks_config_keeps_configurable_input_separate_from_radegs_outputs(self):
        config = load_config(REPO / "configs/tanks_temples.yaml")
        main_config = load_config(REPO / "configs/togog_bali.yaml")
        source = PurePosixPath(config["dataset"]["root"])
        self.assertTrue(source.is_absolute())
        self.assertEqual(source.name, "tanks_temples")
        self.assertEqual(config["output"]["root"], "output/tanks_temples")
        self.assertEqual(config["results"]["csv"], "research_results/experiment_results_tanks_temples.csv")
        self.assertEqual(config["training"], main_config["training"])
        expected_mesh = copy.deepcopy(main_config["mesh"])
        expected_mesh["max_depth"] = 8.0
        expected_mesh["script"] = "mesh_extract_tnt.py"
        self.assertEqual(config["mesh"], expected_mesh)
        self.assertEqual(main_config["mesh"]["max_depth"], 7.0)
        self.assertNotIn("wisnu", config["output"]["root"])
        self.assertNotIn("wisnu", config["results"]["csv"])


if __name__ == "__main__":
    unittest.main()
