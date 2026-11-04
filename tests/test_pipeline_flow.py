import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.config import DEFAULTS
from research_pipeline.runner import ProcessResult, execute_scene
from research_pipeline.scene_discovery import Scene
from research_pipeline.utils import write_json
from test_research_pipeline import make_raw_scene, make_scene



def write_native(output):
    values = {"SSIM": 0.9, "PSNR": 20.0, "LPIPS": 0.1}
    write_json(output / "results.json", {"ours_30000": values})
    write_json(output / "per_view.json", {"ours_30000": {
        key: {"00000.png": value} for key, value in values.items()}})

class PipelineFlow(unittest.TestCase):
    def test_colmap_start_prepares_dataset_then_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = Scene("togog", make_raw_scene(root / "data", "togog"))
            output = root / "out/togog"
            config = copy.deepcopy(DEFAULTS)
            config["rendering"]["enabled"] = False
            config["mesh"]["enabled"] = False
            stages = []

            def fixture(command, log_path, interval):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fixture\n", encoding="utf-8")
                if command[1:3] == ["-m", "research_pipeline.colmap_preprocessor"]:
                    stages.append("colmap")
                    make_scene(scene.path.parent, scene.name)
                elif command[1] == "train.py":
                    (output / "cfg_args").write_text("Namespace(eval=True)")
                    stages.append("training")
                    log_path.write_text("30000/30000 [1it/s, Loss=0.1]\n",
                                        encoding="utf-8")
                return ProcessResult(0, 1.0)

            result = execute_scene(config, scene, output, fixture,
                                   start_from="colmap")
            self.assertEqual(stages, ["colmap", "training"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["start_from"], "colmap")

    def test_image_metric_failure_does_not_prevent_mesh(self):
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
                    (output / "cfg_args").write_text("Namespace(eval=True)")
                    stages.append("training")
                elif command[1] == "render.py":
                    stages.append("rendering")
                    for kind in ("renders", "gt"):
                        folder = output / "test/ours_30000" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "00000.png").write_bytes(b"fixture")
                elif command[1] == "metric.py":
                    stages.append("image_metrics")
                    return ProcessResult(1, 1.0)
                else:
                    stages.append("mesh_extraction")
                    (output / "recon_post.ply").write_bytes(b"fixture")
                return ProcessResult(0, 1.0)

            with patch("research_pipeline.mesh_analyzer.analyze_mesh", return_value={}):
                result = execute_scene(config, scene, output, fixture)
            self.assertEqual(stages, ["training", "rendering", "image_metrics",
                                      "mesh_extraction"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_stage"], "image_metrics")

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
                    (output / "cfg_args").write_text("Namespace(eval=True)")
                    stage = "training"
                    log_path.write_text("30000/30000 [1it/s, Loss=0.1]\n", encoding="utf-8")
                elif command[1] == "render.py":
                    stage = "rendering"
                    for kind in ("renders", "gt"):
                        folder = output / "test/ours_30000" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "00000.png").write_bytes(b"fixture")
                elif command[1] == "mesh_extract.py":
                    stage = "mesh_extraction"
                    self.assertEqual(command[-6:], ["-r", "2", "--num_cluster", "8",
                                                    "--depth_ratio", "0.5"])
                    (output / "recon_post.ply").write_bytes(b"fixture")
                elif command[1] == "metric.py":
                    stage = "image_metrics"
                    write_native(output)
                else:
                    stage = "unknown"
                stages.append(stage)
                return ProcessResult(0, 1.0, {"peak_ram_mb": 10.0}, [])

            mesh_metrics = {"duration_seconds": 1.0, "peak_ram_mb": 10.0,
                            "vertex_count": 3, "face_count": 1, "file_size_mb": 0.1}
            with patch("research_pipeline.mesh_analyzer.analyze_mesh",
                       return_value=mesh_metrics):
                result = execute_scene(config, scene, output, fixture)
            self.assertEqual(stages, ["training", "rendering", "image_metrics",
                                      "mesh_extraction"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["training"]["final_iteration"], 30000)
            self.assertEqual(result["rendering"]["image_count"], 1)
            self.assertEqual(result["mesh_extraction"]["depth_ratio"], 0.5)
            self.assertEqual(result["mesh_extraction"]["script"], "mesh_extract.py")
            self.assertEqual(result["mesh_extraction"]["voxel_size"], 0.009)
            self.assertEqual(result["mesh_extraction"]["max_depth"], 8.0)

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
                    (output / "cfg_args").write_text("Namespace(eval=True)")
                    log_path.write_text("Training complete.\n", encoding="utf-8")
                    return ProcessResult(0, 1.0)
                elif command[1] == "render.py":
                    # Rendering selalu jalan; buat folder renders agar render_statistics tidak error
                    (output / "test/ours_30000/renders").mkdir(parents=True, exist_ok=True)
                    (output / "test/ours_30000/gt").mkdir(parents=True, exist_ok=True)
                    for kind in ("renders", "gt"):
                        (output / "test/ours_30000" / kind / "00000.png").write_bytes(b"fixture")
                    log_path.write_text("Rendering complete.\n", encoding="utf-8")
                    return ProcessResult(0, 1.0)
                elif command[1] == "metric.py":
                    write_native(output)
                    return ProcessResult(0, 1.0)
                log_path.write_text("CUDA out of memory\n", encoding="utf-8")
                return ProcessResult(1, 1.0)

            result = execute_scene(config, scene, output, fixture)
            self.assertEqual(result["failed_stage"], "mesh_extraction_oom")
            self.assertEqual(result["mesh_extraction"]["duration_seconds"], 1.0)
            self.assertEqual(len(commands_seen), 4)
            self.assertEqual(commands_seen[3][-6:], ["-r", "2", "--num_cluster", "8",
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
                    (output / "cfg_args").write_text("Namespace(eval=True)")
                    stages.append("training")
                elif command[1] == "render.py":
                    stages.append("rendering")
                    for kind in ("renders", "gt"):
                        folder = output / "test/ours_30000" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "00000.png").write_bytes(b"fixture")
                elif command[1] == "mesh_extract.py":
                    stages.append("mesh_extraction")
                    (output / "recon_post.ply").write_bytes(b"fixture")
                elif command[1] == "metric.py":
                    stages.append("image_metrics")
                    write_native(output)
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
            self.assertEqual(stages, ["training", "rendering", "image_metrics", "mesh_extraction", "geometry_evaluation"])
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["geometry"]["chamfer_distance"], 0.1)


if __name__ == "__main__":
    unittest.main()
