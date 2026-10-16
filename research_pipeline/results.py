"""Deterministic experiment identity, summaries, and atomic master CSV."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from research_pipeline.config import REPO
from research_pipeline import gpu_monitor
from research_pipeline.utils import atomic_write, write_json


FIELDNAMES = [
    "experiment_id", "timestamp", "scene_name", "object_name", "dataset_path",
    "output_path", "status", "failed_stage", "error_message", "camera_count",
    "training_camera_count", "test_camera_count", "initial_point_count", "camera_extent",
    "final_iteration", "final_loss", "final_loss_normal", "final_loss_ncc", "final_l1",
    "final_training_psnr", "final_it_per_second", "training_duration_seconds",
    "training_peak_vram_mb", "training_average_vram_mb", "training_peak_ram_mb",
    "training_average_ram_mb", "render_duration_seconds", "render_peak_vram_mb",
    "render_average_vram_mb", "render_peak_ram_mb", "render_average_ram_mb",
    "render_image_count", "render_seconds_per_image", "render_fps", "ssim", "psnr",
    "lpips", "mesh_resolution", "mesh_num_cluster", "mesh_depth_ratio",
    "mesh_extraction_duration_seconds", "mesh_peak_vram_mb", "mesh_average_vram_mb",
    "mesh_peak_ram_mb", "mesh_average_ram_mb", "mesh_vertex_count", "mesh_face_count",
    "mesh_file_size_mb", "chamfer_distance", "precision", "recall", "f1_score",
    "geometry_sample_points", "distance_threshold", "git_commit", "git_dirty",
    "python_version", "pytorch_version", "cuda_version", "gpu_name", "driver_version",
]


def experiment_id(scene_name: str, config: dict) -> str:
    payload = {
        "scene_name": scene_name,
        "training": config["training"],
        "rendering": config["rendering"],
        "mesh": config["mesh"],
        "geometry_evaluation": config["geometry_evaluation"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(["git", *arguments], cwd=REPO, capture_output=True,
                                text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def reproducibility() -> tuple[dict, list[str]]:
    warnings = []
    commit = _git("rev-parse", "HEAD")
    dirty_output = _git("status", "--porcelain", "--untracked-files=normal")
    metadata = {
        "git_commit": commit,
        "git_dirty": None if dirty_output is None else bool(dirty_output),
        "python_version": platform.python_version(),
        "pytorch_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "driver_version": None,
    }
    try:
        code = ("import json, torch; print(json.dumps({"
                "'pytorch_version': torch.__version__, "
                "'cuda_version': torch.version.cuda}))")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True, timeout=30, check=True)
        metadata.update(json.loads(result.stdout.strip().splitlines()[-1]))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        warnings.append(f"PyTorch/CUDA version unavailable: {exc}")
    try:
        rows = gpu_monitor.query(["--query-gpu=name,driver_version"])
        metadata["gpu_name"] = "; ".join(dict.fromkeys(row[0] for row in rows)) or None
        metadata["driver_version"] = "; ".join(
            dict.fromkeys(row[1] for row in rows if len(row) > 1)
        ) or None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        warnings.append(f"GPU/driver identity unavailable: {exc}")
    return metadata, warnings


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_row(result: dict) -> dict:
    training = result.get("training", {})
    rendering = result.get("rendering", {})
    mesh = result.get("mesh_extraction", {})
    geometry = result.get("geometry", {})
    reproducible = result.get("reproducibility", {})
    row = {
        "experiment_id": result.get("experiment_id"),
        "timestamp": result.get("timestamp"),
        "scene_name": result.get("scene_name"),
        "object_name": result.get("object_name"),
        "dataset_path": result.get("dataset_path"),
        "output_path": result.get("output_path"),
        "status": result.get("status"),
        "failed_stage": result.get("failed_stage"),
        "error_message": result.get("error_message"),
        "camera_count": training.get("camera_count"),
        "training_camera_count": training.get("training_camera_count"),
        "test_camera_count": training.get("test_camera_count"),
        "initial_point_count": training.get("initial_point_count"),
        "camera_extent": training.get("camera_extent"),
        "final_iteration": training.get("final_iteration"),
        "final_loss": training.get("final_loss"),
        "final_loss_normal": training.get("final_loss_normal"),
        "final_loss_ncc": training.get("final_loss_ncc"),
        "final_l1": training.get("final_l1"),
        "final_training_psnr": training.get("final_psnr"),
        "final_it_per_second": training.get("final_it_per_sec"),
        "training_duration_seconds": training.get("duration_seconds"),
        "training_peak_vram_mb": training.get("peak_vram_mb"),
        "training_average_vram_mb": training.get("average_vram_mb"),
        "training_peak_ram_mb": training.get("peak_ram_mb"),
        "training_average_ram_mb": training.get("average_ram_mb"),
        "render_duration_seconds": rendering.get("duration_seconds"),
        "render_peak_vram_mb": rendering.get("peak_vram_mb"),
        "render_average_vram_mb": rendering.get("average_vram_mb"),
        "render_peak_ram_mb": rendering.get("peak_ram_mb"),
        "render_average_ram_mb": rendering.get("average_ram_mb"),
        "render_image_count": rendering.get("image_count"),
        "render_seconds_per_image": rendering.get("seconds_per_image"),
        "render_fps": rendering.get("fps"),
        "ssim": rendering.get("ssim"),
        "psnr": rendering.get("psnr"),
        "lpips": rendering.get("lpips"),
        "mesh_resolution": mesh.get("resolution"),
        "mesh_num_cluster": mesh.get("num_cluster"),
        "mesh_depth_ratio": mesh.get("depth_ratio"),
        "mesh_extraction_duration_seconds": mesh.get("duration_seconds"),
        "mesh_peak_vram_mb": mesh.get("peak_vram_mb"),
        "mesh_average_vram_mb": mesh.get("average_vram_mb"),
        "mesh_peak_ram_mb": mesh.get("peak_ram_mb"),
        "mesh_average_ram_mb": mesh.get("average_ram_mb"),
        "mesh_vertex_count": mesh.get("vertex_count"),
        "mesh_face_count": mesh.get("face_count"),
        "mesh_file_size_mb": mesh.get("file_size_mb"),
        "chamfer_distance": geometry.get("chamfer_distance"),
        "precision": geometry.get("precision"),
        "recall": geometry.get("recall"),
        "f1_score": geometry.get("f1_score"),
        "geometry_sample_points": geometry.get("sample_points"),
        "distance_threshold": geometry.get("distance_threshold"),
        **{key: reproducible.get(key) for key in (
            "git_commit", "git_dirty", "python_version", "pytorch_version",
            "cuda_version", "gpu_name", "driver_version")},
    }
    return {key: row.get(key) for key in FIELDNAMES}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(f"Master CSV has an unexpected schema: {path}")
        return list(reader)


def _write_rows(path: Path, rows: list[dict]) -> None:
    def write(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, write)


def ensure_master_csv(path: Path) -> None:
    if path.exists():
        read_rows(path)
    else:
        _write_rows(path, [])


def upsert_result(path: Path, result: dict) -> None:
    rows = read_rows(path)
    replacement = csv_row(result)
    updated = False
    for index, row in enumerate(rows):
        if row["experiment_id"] == replacement["experiment_id"]:
            rows[index] = replacement
            updated = True
            break
    if not updated:
        rows.append(replacement)
    _write_rows(path, rows)


def successful_ids(path: Path) -> set[str]:
    return {row["experiment_id"] for row in read_rows(path)
            if row["status"] == "success"}


def write_summary(result: dict) -> None:
    path = Path(result["output_path"]) / "research/summary.json"
    write_json(path, result)

