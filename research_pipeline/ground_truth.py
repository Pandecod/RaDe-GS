"""Resolve explicitly configured GT geometry; no GPU/model imports."""
from pathlib import Path
from research_pipeline.config import resolve_path


def ground_truth_path(config: dict, scene_name: str) -> Path:
    root = resolve_path(config["geometry_evaluation"]["gt_root"])
    flat = root / f"{scene_name}.ply"
    nested = root / scene_name / f"{scene_name}.ply"
    # The flat layout has priority; nested scene folders are also supported.
    return nested if not flat.exists() and nested.exists() else flat
