"""Transactional COLMAP preprocessing for raw scene input directories."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid
from typing import Callable

from research_pipeline.scene_discovery import Scene, validation_errors


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def raw_input_errors(scene: Scene) -> list[str]:
    """Accept an already processed scene or a raw scene containing input images."""
    if not validation_errors(scene):
        return []
    errors = []
    input_path = scene.path / "input"
    if not input_path.is_dir():
        errors.append("Missing input/ directory for COLMAP")
    elif not any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                 for path in input_path.iterdir()):
        errors.append("input/ contains no supported image files")
    if (scene.path / "images").exists() or (scene.path / "sparse").exists():
        errors.append(
            "Incomplete images/ or sparse/ output exists; move it aside before COLMAP"
        )
    return errors


def _run(command: list[str]) -> None:
    print("[COLMAP] " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def _complete_model(path: Path) -> bool:
    return all((path / name).is_file()
               for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _find_model(sparse_root: Path) -> Path:
    candidates = [path for path in sparse_root.iterdir()
                  if path.is_dir() and _complete_model(path)]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no complete sparse model")
    candidates.sort(key=lambda path: (0, int(path.name)) if path.name.isdigit()
                    else (1, path.name))
    return candidates[0]


def _normalize_sparse(staging: Path) -> None:
    sparse = staging / "sparse"
    zero = sparse / "0"
    if _complete_model(zero):
        return
    if not _complete_model(sparse):
        raise RuntimeError("COLMAP undistorter produced no complete sparse model")
    files = [path for path in sparse.iterdir() if path.is_file()]
    zero.mkdir()
    for path in files:
        path.rename(zero / path.name)


def preprocess(scene_path: Path, colmap_executable: str = "colmap",
               camera_model: str = "OPENCV", use_gpu: bool = True,
               *, run: Callable[[list[str]], None] = _run) -> str:
    """Create images/ and sparse/0 only after every COLMAP command succeeds."""
    scene_path = scene_path.resolve()
    scene = Scene(scene_path.name, scene_path)
    if not validation_errors(scene):
        print(f"[COLMAP] {scene.name} is already processed; skipping.", flush=True)
        return "skipped"
    errors = raw_input_errors(scene)
    if errors:
        raise ValueError("; ".join(errors))

    executable = shutil.which(colmap_executable)
    if executable is None:
        candidate = Path(colmap_executable).expanduser()
        if not candidate.is_file():
            raise ValueError(f"COLMAP executable was not found: {colmap_executable}")
        executable = str(candidate.resolve())

    attempt = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
               + "_" + uuid.uuid4().hex[:10])
    work_parent = scene_path / ".pipeline_colmap"
    work = work_parent / attempt
    sparse_work = work / "sparse"
    staging = work / "undistorted"
    sparse_work.mkdir(parents=True)
    staging.mkdir()
    database = work / "database.db"
    input_path = scene_path / "input"
    gpu = "1" if use_gpu else "0"

    try:
        run([
            executable, "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(input_path),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.use_gpu", gpu,
        ])
        run([
            executable, "exhaustive_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", gpu,
        ])
        run([
            executable, "mapper",
            "--database_path", str(database),
            "--image_path", str(input_path),
            "--output_path", str(sparse_work),
            "--Mapper.ba_global_function_tolerance", "0.000001",
        ])
        model = _find_model(sparse_work)
        run([
            executable, "image_undistorter",
            "--image_path", str(input_path),
            "--input_path", str(model),
            "--output_path", str(staging),
            "--output_type", "COLMAP",
        ])
        _normalize_sparse(staging)
        staged_scene = Scene(scene.name, staging)
        staged_errors = validation_errors(staged_scene)
        if staged_errors:
            raise RuntimeError("Invalid staged COLMAP output: " + "; ".join(staged_errors))
        if (scene_path / "images").exists() or (scene_path / "sparse").exists():
            raise RuntimeError("Final COLMAP output appeared while preprocessing was running")
        (staging / "images").rename(scene_path / "images")
        try:
            (staging / "sparse").rename(scene_path / "sparse")
        except BaseException:
            # Restore the staging state so an interrupted publication never
            # leaves a scene that looks ready for training.
            (scene_path / "images").rename(staging / "images")
            raise
        shutil.rmtree(work)
        print(f"[COLMAP] Completed scene {scene.name}", flush=True)
        return "completed"
    except BaseException:
        print(f"[COLMAP] Failed attempt retained for diagnosis: {work}",
              file=sys.stderr, flush=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare one raw scene with COLMAP")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--colmap-executable", default="colmap")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--use-gpu", choices=("0", "1"), default="1")
    args = parser.parse_args(argv)
    try:
        preprocess(args.scene, args.colmap_executable, args.camera_model,
                   args.use_gpu == "1")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"COLMAP preprocessing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
