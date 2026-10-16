"""Synchronous subprocess execution with raw logs and process-tree cleanup."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable
from research_pipeline.commands import commands
from research_pipeline.config import REPO, resolve_path
from research_pipeline.scene_discovery import Scene, validation_errors

@dataclass
class ProcessResult:
    returncode: int
    duration_seconds: float
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

class StageFailure(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)

def cleanup_process(process: subprocess.Popen, children: list | None = None) -> None:
    """Reap the direct child and terminate descendants before another stage."""
    import psutil

    children = children or []
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child in reversed(children):
        try:
            child.terminate()
        except (OSError, RuntimeError, psutil.Error):
            pass
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(children, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError("Process descendants could not be stopped; refusing next stage")

def run_process(command: list[str], log_path: Path, interval: float = 1.0,
                sample: Callable | None = None) -> ProcessResult:
    import psutil
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique attempt output directories mean logs are never overwritten.
    from research_pipeline.process_monitor import ProcessMonitor
    monitor = ProcessMonitor()
    started = time.monotonic()
    children: dict[int, psutil.Process] = {}
    process = None
    with log_path.open("xb") as log:
        try:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=(os.name == "posix"))
            while process.poll() is None:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        children[child.pid] = child
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if sample is not None:
                    sample(process.pid)
                else:
                    monitor.sample(process.pid)
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            result = ProcessResult(process.returncode, time.monotonic() - started,
                                   monitor.metrics(), monitor.warnings)
        finally:
            if process is not None:
                cleanup_process(process, list(children.values()))
            log.flush()
            os.fsync(log.fileno())
    return result

def allocate_output(root: Path, name: str) -> Path:
    from datetime import datetime, timezone
    import uuid
    base = root / name
    if not base.exists() or not any(base.iterdir()):
        return base
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    return base / "runs" / attempt

def execute_scene(config: dict, scene: Scene, output: Path,
                  executor: Callable = run_process) -> dict:
    mesh_config = {
        "resolution": config["mesh"]["resolution"],
        "num_cluster": config["mesh"]["num_cluster"],
        "depth_ratio": config["mesh"]["depth_ratio"],
    }
    geometry_config = {
        "chamfer_distance": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "sample_points": config["geometry_evaluation"]["sample_points"],
        "distance_threshold": config["geometry_evaluation"]["distance_threshold"],
    }
    result = {"scene_name": scene.name, "object_name": scene.name,
              "dataset_path": str(scene.path), "output_path": str(output),
              "status": "failed", "failed_stage": None, "error_message": None,
              "training": {}, "rendering": {}, "mesh_extraction": mesh_config,
              "geometry": geometry_config,
              "warnings": [], "commands": commands(config, scene.path, output)}
    stage = "dataset_validation"
    try:
        errors = validation_errors(scene)
        if errors:
            raise StageFailure(stage, "; ".join(errors))
        stage = "training"
        process = executor(result["commands"][stage], output / "research/logs/training.log",
                           config["resource_monitor"]["interval_seconds"])
        result["training"] = {"duration_seconds": process.duration_seconds,
                              **process.metrics}
        result["warnings"].extend(process.warnings)
        from research_pipeline.training_parser import parse_training
        try:
            result["training"].update(parse_training(output / "research/logs/training.log",
                output / "research/training_metrics.csv", scene.name))
        except (OSError, ValueError) as exc:
            result["warnings"].append(f"Training metrics could not be parsed: {exc}")
        if process.returncode != 0:
            raise StageFailure(stage, f"Training exited with code {process.returncode}")
        if config["rendering"]["enabled"]:
            from research_pipeline.image_evaluator import render_statistics
            from research_pipeline.utils import read_json
            stage = "rendering"
            process = executor(result["commands"][stage], output / "research/logs/rendering.log",
                               config["resource_monitor"]["interval_seconds"])
            result["rendering"] = {"duration_seconds": process.duration_seconds, **process.metrics,
                                  **render_statistics(output, process.duration_seconds)}
            result["warnings"].extend(process.warnings)
            if process.returncode:
                raise StageFailure(stage, f"Rendering exited with code {process.returncode}")
            stage = "image_metrics"
            process = executor(result["commands"][stage], output / "research/logs/image_metrics.log",
                               config["resource_monitor"]["interval_seconds"])
            if process.returncode:
                raise StageFailure(stage, f"Image evaluation exited with code {process.returncode}")
            image_metrics = read_json(output / "research/image_metrics.json")
            if image_metrics.get("image_count", 0) <= 0:
                raise StageFailure(stage, "Image evaluation produced no image metrics")
            result["rendering"].update({k: image_metrics.get(k) for k in ("ssim", "psnr", "lpips", "split", "method")})
            result["rendering"]["evaluation_image_count"] = image_metrics["image_count"]
            result["warnings"].extend(image_metrics.get("warnings", []))
        if config["mesh"]["enabled"]:
            from research_pipeline.mesh_analyzer import analyze_mesh, is_oom_failure
            from research_pipeline.resource_guard import wait_until_ready

            stage = "mesh_resource_check"
            ready, error = wait_until_ready(config)
            if not ready:
                raise StageFailure(stage, error or "Mesh resource check failed")
            stage = "mesh_extraction"
            mesh_log = output / "research/logs/mesh_extraction.log"
            process = executor(result["commands"][stage], mesh_log,
                               config["resource_monitor"]["interval_seconds"])
            mesh_process_metrics = {"duration_seconds": process.duration_seconds,
                                    **process.metrics}
            result["warnings"].extend(process.warnings)
            if process.returncode:
                failed_stage = "mesh_extraction_oom" if is_oom_failure(
                    mesh_log, process.returncode) else stage
                raise StageFailure(failed_stage,
                                   f"Mesh extraction exited with code {process.returncode}")
            result["mesh_extraction"] = {**mesh_config, **analyze_mesh(
                output / "recon_post.ply", mesh_process_metrics, scene.name
            )}
        if config["geometry_evaluation"]["enabled"]:
            from research_pipeline.utils import read_json

            stage = "geometry_evaluation"
            gt_path = Path(result["commands"][stage][
                result["commands"][stage].index("--ground-truth") + 1
            ])
            if not gt_path.is_file():
                raise StageFailure(stage, f"Ground truth does not exist: {gt_path}")
            process = executor(
                result["commands"][stage],
                output / "research/logs/geometry_evaluation.log",
                config["resource_monitor"]["interval_seconds"],
            )
            if process.returncode:
                raise StageFailure(stage,
                                   f"Geometry evaluation exited with code {process.returncode}")
            result["geometry"] = {
                **geometry_config,
                **read_json(output / "research/geometry_metrics.json"),
            }
            result["warnings"].extend(result["geometry"].get("warnings", []))
        result["status"] = "success"
    except StageFailure as exc:
        result.update(failed_stage=exc.stage, error_message=str(exc))
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        result.update(failed_stage=stage, error_message=str(exc))
    return result

def run_batch(config: dict, scenes: list[Scene], resume: bool = False, force: bool = False) -> int:
    from research_pipeline.results import (
        ensure_master_csv, experiment_id, reproducibility, successful_ids,
        timestamp, upsert_result, write_summary,
    )

    csv_path = resolve_path(config["results"]["csv"])
    try:
        ensure_master_csv(csv_path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"CRITICAL: master CSV is not writable: {exc}") from exc
    completed = successful_ids(csv_path) if resume and not force else set()
    environment, environment_warnings = reproducibility()
    failed = False
    for index, scene in enumerate(scenes, 1):
        logging.info("[%s/%s] %s", index, len(scenes), scene.name)
        identity = experiment_id(scene.name, config)
        if resume and not force and identity in completed:
            logging.info("SKIPPED: matching successful experiment %s", identity)
            continue
        output = allocate_output(resolve_path(config["output"]["root"]), scene.name)
        result = execute_scene(config, scene, output)
        result["experiment_id"] = identity
        result["timestamp"] = timestamp()
        result["reproducibility"] = environment
        result["warnings"].extend(environment_warnings)
        try:
            write_summary(result)
        except (OSError, ValueError) as exc:
            result.update(status="failed", failed_stage="result_recording",
                          error_message=f"Unable to write summary.json: {exc}")
        try:
            upsert_result(csv_path, result)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"CRITICAL: master CSV write failed: {exc}") from exc
        failed |= result["status"] != "success"
        logging.info("%s: %s", result["status"], result["error_message"] or scene.name)
    return 1 if failed else 0
