from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research_pipeline.training_parser import parse_training
from research_pipeline.process_monitor import ProcessMonitor

class Phase3(unittest.TestCase):
    def test_tqdm_carriage_returns_multiline_and_split_metrics(self):
        text = ("Reading camera 8/8\rLoading Training Cameras: 7 .\nLoading Test Cameras: 1 .\n"
                "cameras extent: 6.1\nNumber of points at initialisation: 123\n"
                "7000/30000 [65.65it/s, Loss=0.2119, loss_normal=0.2527, loss_ncc=0.2098]\r"
                "[ITER 7000] Evaluating test: L1 0.9 PSNR 2\n"
                "[ITER 7000] Evaluating train:\nL1 0.053633\nPSNR 23.4509\n"
                "30000/30000 [2.0s/it, Loss=1.564e-1, loss_normal=0.0594]\r"
                "[ITER 30000] Evaluating train: L1 0.04 PSNR 25.7\n")
        with tempfile.TemporaryDirectory() as tmp:
            log, csv = Path(tmp) / "log", Path(tmp) / "metrics.csv"
            log.write_text(text)
            result = parse_training(log, csv, "togog")
            self.assertEqual(result["camera_count"], 8)
            self.assertEqual(result["initial_point_count"], 123)
            self.assertEqual(result["final_iteration"], 30000)
            self.assertEqual(result["final_psnr"], 25.7)
            self.assertEqual(result["final_it_per_sec"], 0.5)
            self.assertIsNone(result["final_loss_ncc"])
            self.assertIn("0.053633,23.4509,65.65,0.9,2.0", csv.read_text())
    def test_unknown_log_keeps_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            log.write_text("format changed")
            result = parse_training(log, Path(tmp) / "csv", "scene")
            self.assertTrue(all(value is None for value in result.values()))
    def test_monitor_failure_nonfatal_and_null_gpu(self):
        import os
        monitor = ProcessMonitor()
        with patch("research_pipeline.gpu_monitor.process_vram", side_effect=RuntimeError("unavailable")):
            monitor.sample(os.getpid())
        self.assertGreater(monitor.metrics()["peak_ram_mb"], 0)
        self.assertIsNone(monitor.metrics()["peak_vram_mb"])
        self.assertTrue(monitor.warnings)

    def test_monitor_retries_before_cuda_context_appears(self):
        import os
        monitor = ProcessMonitor()
        with patch("research_pipeline.gpu_monitor.process_vram", side_effect=[None, 123.0]):
            monitor.sample(os.getpid())
            monitor.sample(os.getpid())
        self.assertEqual(monitor.metrics()["peak_vram_mb"], 123.0)

if __name__ == "__main__":
    unittest.main()
