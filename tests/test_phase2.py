import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from research_pipeline.config import DEFAULTS
from research_pipeline.runner import run_process, execute_scene, allocate_output
from research_pipeline.scene_discovery import Scene
from test_research_pipeline import make_scene

class Phase2(unittest.TestCase):
    def test_raw_stdout_stderr_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "training.log"
            result = run_process([sys.executable, "-c",
                "import os,sys; os.write(1,b'progress\\r10/10\\n'); os.write(2,b'error\\n'); sys.exit(7)"],
                log, 0.01)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(log.read_bytes(), b"progress\r10/10\nerror\n")
    def test_serial_failure_recovery_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = [Scene(name, make_scene(root / "data", name)) for name in ("a", "b")]
            finished = []
            def fixture(command, log, interval):
                name = log.parents[2].name
                code = "import sys; print('fixture'); sys.exit(" + ("1" if name == "a" else "0") + ")"
                result = run_process([sys.executable, "-c", code], log, 0.01)
                finished.append(name)
                return result
            config = copy.deepcopy(DEFAULTS)
            config["rendering"]["enabled"] = False
            config["mesh"]["enabled"] = False
            results = [execute_scene(config, s, root / "out" / s.name, fixture) for s in scenes]
            self.assertEqual(finished, ["a", "b"])
            self.assertEqual([r["status"] for r in results], ["failed", "success"])
            self.assertEqual(results[0]["failed_stage"], "training")
            new_output = allocate_output(root / "out", "a")
            self.assertNotEqual(new_output, root / "out/a")
            self.assertTrue((root / "out/a/research/logs/training.log").is_file())
    def test_interrupt_reaps_process(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            def interrupt(pid):
                seen.append(pid)
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                run_process([sys.executable, "-c", "import time; time.sleep(30)"],
                            Path(tmp) / "log", 0.01, sample=interrupt)
            self.assertFalse(psutil.pid_exists(seen[0]))

if __name__ == "__main__":
    unittest.main()
