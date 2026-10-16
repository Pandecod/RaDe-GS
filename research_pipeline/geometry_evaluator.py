"""Optional deterministic mesh/point-cloud geometry evaluation."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from research_pipeline.utils import write_json


def sample_geometry(path: Path, count: int, seed: int) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"Geometry file does not exist: {path}")
    import trimesh

    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        meshes = [item for item in geometry.geometry.values()
                  if hasattr(item, "faces") and len(item.faces)]
        if meshes:
            geometry = trimesh.util.concatenate(meshes)
        else:
            vertices = [np.asarray(item.vertices) for item in geometry.geometry.values()
                        if hasattr(item, "vertices") and len(item.vertices)]
            if not vertices:
                raise ValueError(f"Geometry has no vertices: {path}")
            points = np.concatenate(vertices, axis=0)
            rng = np.random.default_rng(seed)
            return points[rng.choice(len(points), count, replace=len(points) < count)]
    if hasattr(geometry, "faces") and len(geometry.faces):
        try:
            points, _ = trimesh.sample.sample_surface(geometry, count, seed=seed)
        except TypeError:
            state = np.random.get_state()
            try:
                np.random.seed(seed)
                points, _ = trimesh.sample.sample_surface(geometry, count)
            finally:
                np.random.set_state(state)
        return np.asarray(points, dtype=np.float64)
    if not hasattr(geometry, "vertices") or not len(geometry.vertices):
        raise ValueError(f"Geometry has no vertices: {path}")
    points = np.asarray(geometry.vertices, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), count, replace=len(points) < count)]


def threshold_metrics(reconstruction_distances: np.ndarray,
                      gt_distances: np.ndarray,
                      threshold: float | None) -> dict:
    chamfer = float(np.mean(reconstruction_distances) + np.mean(gt_distances))
    if threshold is None:
        return {"chamfer_distance": chamfer, "precision": None,
                "recall": None, "f1_score": None}
    precision = float(np.mean(reconstruction_distances < threshold))
    recall = float(np.mean(gt_distances < threshold))
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return {"chamfer_distance": chamfer, "precision": precision,
            "recall": recall, "f1_score": f1}


def evaluate(reconstruction: Path, ground_truth: Path, sample_points: int,
             distance_threshold: float | None, seed: int, scene_name: str,
             output: Path) -> dict:
    from scipy.spatial import cKDTree

    reconstructed = sample_geometry(reconstruction, sample_points, seed)
    gt = sample_geometry(ground_truth, sample_points, seed + 1)
    if not np.isfinite(reconstructed).all() or not np.isfinite(gt).all():
        raise ValueError("Geometry contains non-finite coordinates")
    reconstruction_distances = cKDTree(gt).query(reconstructed, k=1, workers=1)[0]
    gt_distances = cKDTree(reconstructed).query(gt, k=1, workers=1)[0]
    values = threshold_metrics(reconstruction_distances, gt_distances,
                               distance_threshold)
    if not math.isfinite(values["chamfer_distance"]):
        raise ValueError("Chamfer distance is non-finite")
    warnings = []
    if distance_threshold is None:
        warnings.append("distance_threshold is null; precision, recall, and F1 are null")
    result = {
        "scene_name": scene_name,
        "object_name": scene_name,
        **values,
        "sample_points": sample_points,
        "distance_threshold": distance_threshold,
        "seed": seed,
        "ground_truth_path": str(ground_truth),
        "warnings": warnings,
    }
    write_json(output / "research/geometry_metrics.json", result)
    print(f"Geometry metrics: {values}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--sample-points", required=True, type=int)
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.reconstruction, args.ground_truth, args.sample_points,
             args.distance_threshold, args.seed, args.scene_name, args.output)


if __name__ == "__main__":
    main()

