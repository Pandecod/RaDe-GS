"""Read the verified final RaDe-GS mesh and record structural metrics."""
from __future__ import annotations

import math
from pathlib import Path

from research_pipeline.utils import write_json


OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "std::bad_alloc",
    "cannot allocate memory",
    "killed",
)


def cluster_triangle_threshold(cluster_sizes, cluster_to_keep: int,
                               minimum_triangles: int = 50) -> int:
    """Return a safe cutoff when fewer clusters exist than requested."""
    sizes = sorted(int(size) for size in cluster_sizes)
    if not sizes:
        raise ValueError("Extracted mesh has no connected triangle clusters")
    keep = min(max(int(cluster_to_keep), 1), len(sizes))
    return max(sizes[-keep], int(minimum_triangles))


def is_oom_failure(log_path: Path, returncode: int) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    # A direct subprocess reports SIGKILL as -9; wrappers sometimes report 137.
    return any(marker in text for marker in OOM_MARKERS) or returncode in (-9, 137)


def analyze_mesh(mesh_path: Path, process_metrics: dict, scene_name: str) -> dict:
    if not mesh_path.is_file():
        raise ValueError(f"Final mesh was not created: {mesh_path}")
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    if vertex_count <= 0 or face_count <= 0:
        raise ValueError(f"Final mesh is empty or invalid: {mesh_path}")
    result = {
        "scene_name": scene_name,
        "object_name": scene_name,
        **process_metrics,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "file_size_mb": mesh_path.stat().st_size / 1024**2,
        "mesh_path": str(mesh_path),
    }
    for value in result.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Mesh metrics contain a non-finite number")
    write_json(mesh_path.parent / "research/mesh_metrics.json", result)
    return result

