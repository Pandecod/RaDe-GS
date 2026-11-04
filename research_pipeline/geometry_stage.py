"""Run optional GT geometry after mesh extraction; missing references are explicit skips."""
import math
from pathlib import Path
import stat

from research_pipeline.utils import read_json, write_json


def evaluate_geometry_stage(config, scene_name, output, command, executor):
    settings = config["geometry_evaluation"]
    gt = Path(command[command.index("--ground-truth") + 1])
    log = output / "research/logs/geometry_evaluation.log"
    artifact = output / "research/geometry_metrics.json"
    if artifact.exists():
        raise ValueError("Geometry output already exists; refuse stale results")
    result = {
        "scene_name": scene_name, "object_name": scene_name,
        "status": "failed", "skip_reason": None, "error_message": None,
        "ground_truth_path": str(gt), "evaluator": "research_pipeline.geometry_evaluator",
        "chamfer_distance": None, "precision": None, "recall": None, "f1_score": None,
        "sample_points": settings["sample_points"], "distance_threshold": settings["distance_threshold"],
        "duration_seconds": None, "warnings": [],
        "coordinate_note": "Requires reconstruction and reference already aligned in the same scale and coordinate frame; not the official TNT benchmark protocol.",
    }
    try:
        try:
            reference_stat = gt.stat()
        except FileNotFoundError:
            if not settings["skip_missing_gt"]:
                raise ValueError(f"Ground truth does not exist: {gt}")
            result.update(status="skipped", skip_reason="missing_ground_truth",
                          message=f"GT 3D unavailable; geometry evaluation skipped: {gt}")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x", encoding="utf-8") as stream:
                stream.write(result["message"] + "\n")
            print("[GEOMETRY] SKIPPED missing_ground_truth: " + str(gt), flush=True)
        else:
            if not stat.S_ISREG(reference_stat.st_mode):
                raise ValueError(f"Ground truth must be a regular geometry file: {gt}")
            process = executor(command, log, config["resource_monitor"]["interval_seconds"])
            result.update(duration_seconds=process.duration_seconds, **process.metrics)
            result["warnings"].extend(process.warnings)
            if process.returncode:
                raise RuntimeError(f"Geometry evaluation exited with code {process.returncode}")
            values = read_json(artifact)
            if not isinstance(values, dict):
                raise ValueError("Geometry output must contain a JSON object")
            for key in ("chamfer_distance", "precision", "recall", "f1_score"):
                value = values.get(key)
                if key != "chamfer_distance" and settings["distance_threshold"] is None:
                    if value is not None:
                        raise ValueError(f"Unexpected {key} without distance_threshold")
                elif (type(value) not in (int, float) or not math.isfinite(value)
                      or value < 0 or (key != "chamfer_distance" and value > 1)):
                    raise ValueError(f"Incomplete or invalid geometry metric: {key}")
            result.update({key: values.get(key) for key in (
                "chamfer_distance", "precision", "recall", "f1_score", "seed")})
            result["warnings"].extend(values.get("warnings", []))
            result["status"] = "success"
    except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
        result.update(status="failed", error_message=str(exc))
    write_json(artifact, result)
    return result
