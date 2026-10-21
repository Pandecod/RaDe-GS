import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.config import DEFAULTS
from research_pipeline.runner import ProcessResult, execute_scene
from research_pipeline.scene_discovery import Scene
from research_pipeline.utils import write_json
from test_research_pipeline import make_scene


class PipelineFlow(unittest.TestCase):
    def test_full_enabled_flow_is_strictly_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("togog", make_scene(root / "data", "togog"))
            output = root / "out/togog"
            config = copy.deepcopy(DEFAULTS)
            config["resource_guard"]["enabled"] = False
            stages = []

            def fixture(command, log_path, interval):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fixture\n", encoding="utf-8")
                if command[1] == "train.py":
                    stage = "training"
                    log_path.write_text("30000/30000 [1it/s, Loss=0.1]\n", encoding="utf-8")
                elif command[1] == "render.py":
                    stage = "rendering"
                    for kind in ("renders", "gt"):
                        folder = output / "train/ours_30000" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "00000.png").write_bytes(b"fixture")
                elif command[1] == "mesh_extract.py":
                    stage = "mesh_extraction"
                    self.assertEqual(command[-6:], ["-r", "2", "--num_cluster", "8",
                                                    "--depth_ratio", "0.5"])
                    (output / "recon_post.ply").write_bytes(b"fixture")
                elif command[1:3] == ["-m", "research_pipeline.image_evaluator"]:
                    stage = "image_metrics"
                    write_json(output / "research/image_metrics.json", {
                        "ssim": 0.9, "psnr": 20.0, "lpips": 0.1,
                        "image_count": 1, "split": "train", "method": "ours_30000",
                        "warnings": [],
                    })
                else:
                    stage = "unknown"
                stages.append(stage)
                return ProcessResult(0, 1.0, {"peak_ram_mb": 10.0}, [])

            mesh_metrics = {"duration_seconds": 1.0, "peak_ram_mb": 10.0,
                            "vertex_count": 3, "face_count": 1, "file_size_mb": 0.1}
            with patch("research_pipeline.mesh_analyzer.analyze_mesh",
                       return_value=mesh_metrics):
                result = execute_scene(config, scene, output, fixture)
            self.assertEqual(stages, ["training", "rendering", "mesh_extraction",
                                      "image_metrics"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["training"]["final_iteration"], 30000)
            self.assertEqual(result["rendering"]["image_count"], 1)
            self.assertEqual(result["mesh_extraction"]["depth_ratio"], 0.5)

    def test_mesh_oom_is_not_retried_or_adjusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("togog", make_scene(root / "data", "togog"))
            output = root / "out/togog"
            config = copy.deepcopy(DEFAULTS)
            config["resource_guard"]["enabled"] = False
            commands_seen = []

            def fixture(command, log_path, interval):
                commands_seen.append(command)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if command[1] == "train.py":
                    log_path.write_text("Training complete.\n", encoding="utf-8")
                    return ProcessResult(0, 1.0)
                elif command[1] == "render.py":
                    # Rendering selalu jalan; buat folder renders agar render_statistics tidak error
                    (output / "train/ours_30000/renders").mkdir(parents=True, exist_ok=True)
                    (output / "train/ours_30000/gt").mkdir(parents=True, exist_ok=True)
                    log_path.write_text("Rendering complete.\n", encoding="utf-8")
                    return ProcessResult(0, 1.0)
                log_path.write_text("CUDA out of memory\n", encoding="utf-8")
                return ProcessResult(1, 1.0)

            result = execute_scene(config, scene, output, fixture)
            self.assertEqual(result["failed_stage"], "mesh_extraction_oom")
            self.assertEqual(len(commands_seen), 3)
            self.assertEqual(commands_seen[2][-6:], ["-r", "2", "--num_cluster", "8",
                                                     "--depth_ratio", "0.5"])

    def test_optional_geometry_runs_after_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("togog", make_scene(root / "data", "togog"))
            output = root / "out/togog"
            gt_root = root / "gt"
            gt_root.mkdir()
            (gt_root / "togog.ply").write_bytes(b"fixture")
            config = copy.deepcopy(DEFAULTS)
            config["resource_guard"]["enabled"] = False
            config["geometry_evaluation"].update(enabled=True, gt_root=str(gt_root))
            stages = []

            def fixture(command, log_path, interval):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fixture\n", encoding="utf-8")
                if command[1] == "train.py":
                    stages.append("training")
                elif command[1] == "render.py":
                    stages.append("rendering")
                    for kind in ("renders", "gt"):
                        folder = output / "train/ours_30000" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "00000.png").write_bytes(b"fixture")
                elif command[1] == "mesh_extract.py":
                    stages.append("mesh_extraction")
                    (output / "recon_post.ply").write_bytes(b"fixture")
                elif command[1:3] == ["-m", "research_pipeline.image_evaluator"]:
                    stages.append("image_metrics")
                    write_json(output / "research/image_metrics.json", {
                        "ssim": 0.9, "psnr": 20.0, "lpips": 0.1,
                        "image_count": 1, "split": "train", "method": "ours_30000",
                        "warnings": [],
                    })
                else:
                    stages.append("geometry_evaluation")
                    write_json(output / "research/geometry_metrics.json", {
                        "chamfer_distance": 0.1, "precision": None, "recall": None,
                        "f1_score": None, "sample_points": 100000,
                        "distance_threshold": None, "warnings": ["threshold null"],
                    })
                return ProcessResult(0, 1.0)

            with patch("research_pipeline.mesh_analyzer.analyze_mesh", return_value={
                    "vertex_count": 3, "face_count": 1, "file_size_mb": 0.1}):
                result = execute_scene(config, scene, output, fixture)
            self.assertEqual(stages, ["training", "rendering", "mesh_extraction", "image_metrics", "geometry_evaluation"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["geometry"]["chamfer_distance"], 0.1)


if __name__ == "__main__":
    unittest.main()
