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
            result["image_metrics"] = [sys.executable, "-m", "research_pipeline.image_evaluator",
                                        "--output", str(output), "--scene-name", dataset.name]
    if config["mesh"]["enabled"]:
        result["mesh_extraction"] = [sys.executable, "mesh_extract_tnt.py", "-m", str(output),
                                     "-r", "2", "--num_cluster", "8", "--depth_ratio", "0.5",
                                     "--voxel_size", str(config["mesh"]["voxel_size"]),
                                     "--max_depth", str(config["mesh"]["max_depth"]),
                                     *config["mesh"]["extra_args"]]
    if config["geometry_evaluation"]["enabled"]:
        geometry = config["geometry_evaluation"]
        gt_root = Path(geometry["gt_root"]).expanduser()
        if not gt_root.is_absolute():
            gt_root = Path(__file__).resolve().parents[1] / gt_root
        command = [sys.executable, "-m", "research_pipeline.geometry_evaluator",
                   "--reconstruction", str(output / "recon_post.ply"),
                   "--ground-truth", str(gt_root.resolve() / f"{dataset.name}.ply"),
                   "--sample-points", str(geometry["sample_points"]),
                   "--seed", str(geometry["seed"]), "--scene-name", dataset.name,
                   "--output", str(output)]
        if geometry["distance_threshold"] is not None:
            command.extend(["--distance-threshold", str(geometry["distance_threshold"])])
        result["geometry_evaluation"] = command
    return result

def display(command: list[str]) -> str:
    return shlex.join(["python", *command[1:]])
