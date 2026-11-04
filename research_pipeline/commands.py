"""Build argument lists against the inspected repository CLI."""
import shlex
import sys
from pathlib import Path

def commands(config: dict, dataset: Path, output: Path,
             start_from: str = "training") -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if start_from == "colmap":
        colmap = config["colmap"]
        result["colmap"] = [
            sys.executable, "-m", "research_pipeline.colmap_preprocessor",
            "--scene", str(dataset),
            "--colmap-executable", colmap["executable"],
            "--camera-model", colmap["camera_model"],
            "--use-gpu", "1" if colmap["use_gpu"] else "0",
        ]
    elif start_from != "training":
        raise ValueError(f"Unsupported start stage: {start_from}")
    result["training"] = [sys.executable, "train.py", "-s", str(dataset), "-m", str(output),
                          *config["training"]["extra_args"]]
    if config["rendering"]["enabled"]:
        result["rendering"] = [sys.executable, "render.py", "-m", str(output),
                               *config["rendering"]["extra_args"]]
        if config["image_metrics"]["enabled"]:
            result["image_metrics"] = [sys.executable, "metric.py", "-m", str(output)]
    if config["mesh"]["enabled"]:
        mesh = config["mesh"]
        script = mesh.get("script", "mesh_extract.py")
        if script == "mesh_extract.py" and (mesh["voxel_size"], mesh["max_depth"]) != (0.009, 8.0):
            raise ValueError("Restored mesh_extract.py uses fixed voxel_size=0.009 and max_depth=8.0")
        command = [sys.executable, script, "-m", str(output),
                   "-r", str(mesh["resolution"]), "--num_cluster", str(mesh["num_cluster"]),
                   "--depth_ratio", str(mesh["depth_ratio"])]
        if script == "mesh_extract_tnt.py":
            command.extend(["--voxel_size", str(mesh["voxel_size"]),
                            "--max_depth", str(mesh["max_depth"])])
        result["mesh_extraction"] = [*command, *mesh["extra_args"]]
    if config["geometry_evaluation"]["enabled"]:
        geometry = config["geometry_evaluation"]
        from research_pipeline.ground_truth import ground_truth_path
        gt_path = ground_truth_path(config, dataset.name)
        command = [sys.executable, "-m", "research_pipeline.geometry_evaluator",
                   "--reconstruction", str(output / "recon_post.ply"),
                   "--ground-truth", str(gt_path),
                   "--sample-points", str(geometry["sample_points"]),
                   "--seed", str(geometry["seed"]), "--scene-name", dataset.name,
                   "--output", str(output)]
        if geometry["distance_threshold"] is not None:
            command.extend(["--distance-threshold", str(geometry["distance_threshold"])])
        result["geometry_evaluation"] = command
    return result

def display(command: list[str]) -> str:
    return shlex.join(["python", *command[1:]])
