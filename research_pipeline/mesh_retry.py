"""Record an isolated mesh retry using an existing trained Gaussian model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import uuid

from research_pipeline.config import REPO
from research_pipeline.mesh_analyzer import analyze_mesh, is_oom_failure
from research_pipeline.runner import run_process
from research_pipeline.utils import atomic_write, read_json, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def _prepare_model(model: Path, attempt: Path):
    # Scene writes multi_view.json into model_path even when loading a trained PLY.
    # Use a small copied config and a directory link so original run files survive.
    shutil.copy2(model / "cfg_args", attempt / "cfg_args")
    (attempt / "point_cloud").symlink_to(model / "point_cloud", target_is_directory=True)


def _write_report(attempt: Path, result: dict):
    write_json(attempt / "result.json", result)
    metrics = result.get("process_metrics", {})
    device = metrics.get("device_global", {}).get("device_vram_used_mb", {})
    duration = result.get("duration_seconds")
    settings = result["settings"]
    mesh = result.get("mesh_metrics", {})
    row = {
        "timestamp_utc": result["finished_utc"],
        "scene_name": result["scene_name"], "method": "RaDe-GS",
        "status": result["status"], "failed_stage": result.get("failed_stage"),
        "error_message": result.get("error_message"),
        "script": settings["script"], "iteration": settings["iteration"],
        "resolution": settings["resolution"], "num_cluster": settings["num_cluster"],
        "depth_ratio": settings["depth_ratio"],
        "voxel_size": settings["voxel_size"], "max_depth": settings["max_depth"],
        "mesh_duration_seconds": duration,
        "mesh_duration_minutes": duration / 60 if duration is not None else None,
        "mesh_peak_ram_mb": metrics.get("peak_ram_mb"),
        "mesh_peak_process_vram_mb": metrics.get("peak_vram_mb"),
        "mesh_peak_device_vram_sample_mb": device.get("peak"),
        "mesh_vertex_count": mesh.get("vertex_count"),
        "mesh_face_count": mesh.get("face_count"),
        "mesh_file_size_mb": mesh.get("file_size_mb"),
        "model_path": result["model_path"], "attempt_path": str(attempt),
        "exit_code": result.get("exit_code"),
    }

    def write_csv(stream):
        stream.write("\ufeff")
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    atomic_write(attempt / "result.csv", write_csv)


def retry_mesh(model_path, *, max_depth, script="mesh_extract.py", iteration=30000,
               resolution=2, num_cluster=8, depth_ratio=0.5, voxel_size=0.009,
               interval=5.0, executor=run_process, analyzer=analyze_mesh):
    if script not in ("mesh_extract.py", "mesh_extract_tnt.py"):
        raise ValueError("Choose mesh_extract.py or mesh_extract_tnt.py")
    for name, value in (("max_depth", max_depth), ("voxel_size", voxel_size),
                        ("interval", interval)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if iteration <= 0 or resolution <= 0 or num_cluster <= 0:
        raise ValueError("iteration, resolution and num_cluster must be positive")
    if not math.isfinite(depth_ratio) or not 0 <= depth_ratio <= 1:
        raise ValueError("depth_ratio must be finite and between 0 and 1")
    model = Path(model_path).resolve()
    for required in (model / "cfg_args",
                     model / f"point_cloud/iteration_{iteration}/point_cloud.ply"):
        if not required.is_file():
            raise ValueError(f"Required existing model file not found: {required}")

    settings = dict(script=script, iteration=iteration, resolution=resolution,
                    num_cluster=num_cluster, depth_ratio=depth_ratio,
                    voxel_size=voxel_size, max_depth=max_depth)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt = model / "research" / ("mesh_retry_" + run_id)
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-B", str(REPO / script),
               "-m", str(attempt), "--iteration", str(iteration),
               "-r", str(resolution), "--num_cluster", str(num_cluster),
               "--depth_ratio", str(depth_ratio), "--voxel_size", str(voxel_size),
               "--max_depth", str(max_depth)]
    result = {
        "kind": "mesh_retry", "scene_name": model.name, "model_path": str(model),
        "attempt_path": str(attempt), "started_utc": utc(), "status": "running",
        "settings": settings, "command": command, "training_reused": True,
        "source_summary_path": str(model / "research/summary.json"),
        "warnings": [], "exit_code": None, "duration_seconds": None,
        "telemetry_note": "Process VRAM and device-global VRAM are separate sampled values; unavailable values stay null. Device usage includes other workloads.",
    }
    print(f"[MESH RETRY] Output: {attempt}", flush=True)
    started = perf_counter()
    stage = "preparation"
    try:
        result["script_sha256"] = hashlib.sha256((REPO / script).read_bytes()).hexdigest()
        if (model / "research/summary.json").is_file():
            try:
                result["source_summary_snapshot"] = read_json(model / "research/summary.json")
            except (OSError, ValueError) as exc:
                result["warnings"].append(f"Original summary could not be read: {exc}")
        write_json(attempt / "result.json", result)
        _prepare_model(model, attempt)
        stage = "mesh_extraction"
        process = executor(command, attempt / "mesh.log", interval)
        result.update(exit_code=process.returncode, duration_seconds=process.duration_seconds,
                      duration_scope="mesh_subprocess_including_loading_and_saving",
                      process_metrics=process.metrics)
        result["warnings"].extend(process.warnings)
        if process.returncode:
            if (attempt / "mesh.log").is_file() and is_oom_failure(attempt / "mesh.log", process.returncode):
                stage = "mesh_extraction_oom"
            raise RuntimeError(f"Mesh extraction exited with code {process.returncode}")
        stage = "mesh_validation"
        result["mesh_metrics"] = analyzer(
            attempt / "recon_post.ply",
            {"duration_seconds": process.duration_seconds, **process.metrics}, model.name)
        result["status"] = "success"
    except KeyboardInterrupt:
        result.update(status="interrupted", failed_stage=stage,
                      error_message="Interrupted by user", exit_code=130)
    except Exception as exc:
        result.update(status="failed", failed_stage=stage,
                      error_message=f"{type(exc).__name__}: {exc}")
    finally:
        result["finished_utc"] = utc()
        result["retry_wall_seconds"] = perf_counter() - started
        _write_report(attempt, result)
    print(f"[MESH RETRY] Status: {result['status']}", flush=True)
    print(f"[MESH RETRY] Report: {attempt / 'result.csv'}", flush=True)
    if result.get("error_message"):
        print(f"[MESH RETRY] {result['error_message']}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Existing trained output folder, not the dataset folder")
    parser.add_argument("--script", choices=("mesh_extract.py", "mesh_extract_tnt.py"),
                        default="mesh_extract.py")
    parser.add_argument("--max-depth", required=True, type=float)
    parser.add_argument("--voxel-size", default=0.009, type=float)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--resolution", default=2, type=int)
    parser.add_argument("--num-cluster", default=8, type=int)
    parser.add_argument("--depth-ratio", default=0.5, type=float)
    parser.add_argument("--interval", default=5.0, type=float)
    options = vars(parser.parse_args(argv))
    model = options.pop("model")
    try:
        result = retry_mesh(model, **options)
    except (OSError, ValueError) as exc:
        print(f"[MESH RETRY] {exc}", file=sys.stderr)
        return 2
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
