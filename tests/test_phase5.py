import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest

from research_pipeline.config import DEFAULTS
from research_pipeline.mesh_analyzer import analyze_mesh, is_oom_failure
from research_pipeline.resource_guard import wait_until_ready


class Phase5(unittest.TestCase):
    def test_resource_guard_waits_then_passes(self):
        config = copy.deepcopy(DEFAULTS)
        config["resource_guard"]["cooldown_before_mesh_seconds"] = 0
        config["resource_guard"]["check_interval_seconds"] = 1
        config["resource_guard"]["max_wait_seconds"] = 2
        readings = iter([(3.0, 900.0), (5.0, 2048.0)])
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        ready, error = wait_until_ready(
            config, sleep=sleep, clock=lambda: now[0], probe=lambda: next(readings)
        )
        self.assertTrue(ready)
        self.assertIsNone(error)
        self.assertEqual(now[0], 1.0)

    def test_unverifiable_or_persistently_low_resources_fail(self):
        config = copy.deepcopy(DEFAULTS)
        config["resource_guard"]["cooldown_before_mesh_seconds"] = 0
        ready, error = wait_until_ready(
            config, probe=lambda: (_ for _ in ()).throw(RuntimeError("no nvidia-smi"))
        )
        self.assertFalse(ready)
        self.assertIn("safely", error)
        config["resource_guard"]["wait_if_memory_low"] = False
        ready, _ = wait_until_ready(config, probe=lambda: (1.0, 1.0))
        self.assertFalse(ready)

    def test_oom_markers_and_mesh_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "mesh.log"
            log.write_text("CUDA out of memory", encoding="utf-8")
            self.assertTrue(is_oom_failure(log, 1))
            log.write_text("ordinary error", encoding="utf-8")
            self.assertTrue(is_oom_failure(log, -9))
            self.assertFalse(is_oom_failure(log, 1))

            mesh_path = root / "recon_post.ply"
            mesh_path.write_bytes(b"mesh")
            fake_mesh = types.SimpleNamespace(vertices=[1, 2, 3], faces=[1])
            fake_module = types.SimpleNamespace(load=lambda *args, **kwargs: fake_mesh)
            old = sys.modules.get("trimesh")
            sys.modules["trimesh"] = fake_module
            try:
                result = analyze_mesh(mesh_path, {"duration_seconds": 2.0}, "togog")
            finally:
                if old is None:
                    del sys.modules["trimesh"]
                else:
                    sys.modules["trimesh"] = old
            self.assertEqual(result["vertex_count"], 3)
            self.assertEqual(result["face_count"], 1)
            self.assertTrue((root / "research/mesh_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
