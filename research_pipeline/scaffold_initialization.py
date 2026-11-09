"""Scaffold-GS anchors as an optional RaDe-GS point-cloud initializer.

No torch/Scaffold imports in this process. Scaffold training runs in its own
Python environment; exporting anchors needs numpy, scipy and plyfile.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

from research_pipeline.config import REPO
from research_pipeline.scene_discovery import Scene, validation_errors
from research_pipeline.utils import read_json, write_json

VERSION = 2
METHOD = "RaDe-GS + Scaffold anchor initialization"


def locations(output: Path) -> dict[str, Path]:
    base = output / "scaffold_initialization"
    return {"base": base, "source": base / "source", "model": base / "model",
            "dataset": base / "radegs_input", "report": base / "initialization.json",
            "training": base / "scaffold_training.json"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()



def resolve_python(value: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError("Set SCAFFOLD_PYTHON to the Python executable in the Scaffold environment")
    executable = shutil.which(expanded)
    if executable is None:
        raise ValueError(f"Scaffold Python executable not found: {expanded}")
    return executable


def repository_state(repository: Path, expected_commit: str = "") -> dict:
    for filename in ("train.py", "arguments/__init__.py", "scene/gaussian_model.py"):
        if not (repository / filename).is_file():
            raise ValueError(f"Scaffold-GS repository is incomplete: {repository / filename}")
    state = {"repository": str(repository.resolve()), "git_commit": None, "git_dirty": None}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                  capture_output=True, text=True, timeout=10, check=True)
        state["git_commit"] = revision.stdout.strip()
        dirty = subprocess.run(["git", "diff", "--name-only", "HEAD", "--"],
                               cwd=repository, capture_output=True, text=True,
                               timeout=10, check=True)
        changed = [name for name in dirty.stdout.splitlines() if name]
        state["git_dirty"] = bool(changed)
        state["modified_source_files"] = [
            name for name in changed if Path(name).suffix.lower() in
            (".py", ".cu", ".cuh", ".cpp", ".h", ".yml") or name.endswith("CMakeLists.txt")]
    except (OSError, subprocess.SubprocessError):
        if expected_commit:
            raise ValueError("Cannot verify the pinned Scaffold checkout with Git")
    if expected_commit and state["git_commit"] != expected_commit:
        raise ValueError(f"Scaffold commit mismatch: expected {expected_commit}, got {state['git_commit']}")
    if expected_commit and state.get("modified_source_files"):
        raise ValueError("Pinned Scaffold checkout has modified tracked files")
    state["code_sha256"] = {name: digest(repository / name) for name in (
        "train.py", "arguments/__init__.py", "scene/gaussian_model.py")}
    return state


def input_fingerprint(source: Path, split: dict, images: str = "images") -> str:
    # Hash registered photographs and reconstruction inputs. Ignore timestamps.
    paths = [source / images / name for name in sorted(split["train"] + split["test"])]
    sparse = source / "sparse/0"
    paths.extend(path for path in sorted(sparse.iterdir())
                 if path.name in {"cameras.bin", "cameras.txt", "images.bin", "images.txt",
                                  "points3D.bin", "points3D.txt", "points3D.ply"})
    value = hashlib.sha256()
    for path in paths:
        value.update(path.relative_to(source).as_posix().encode("utf-8"))
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def colmap_reader():
    # Import the existing CPU-only reader without importing scene/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "_scaffold_init_colmap", REPO / "scene/colmap_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_manifest(source: Path, images: str = "images") -> dict:
    reader = colmap_reader()
    sparse = source / "sparse/0"
    try:
        cameras = reader.read_intrinsics_binary(sparse / "cameras.bin")
        registered = reader.read_extrinsics_binary(sparse / "images.bin")
    except (OSError, ValueError, EOFError, struct.error):
        try:
            cameras = reader.read_intrinsics_text(sparse / "cameras.txt")
            registered = reader.read_extrinsics_text(sparse / "images.txt")
        except AssertionError as exc:
            raise ValueError("Use undistorted PINHOLE COLMAP text input") from exc
    if not cameras or not registered:
        raise ValueError("COLMAP must contain registered cameras and images")
    if any(c.model not in ("PINHOLE", "SIMPLE_PINHOLE") for c in cameras.values()):
        raise ValueError("Use undistorted PINHOLE/SIMPLE_PINHOLE COLMAP input")
    # Match both official loaders: basename, split at first '.', lexicographic sort.
    names = sorted((Path(item.name).name for item in registered.values()),
                   key=lambda name: name.split(".")[0])
    stems = [name.split(".")[0] for name in names]
    if len(set(stems)) != len(stems):
        raise ValueError("Registered photo names must have unique loader image names")
    for name in names:
        if not (source / images / name).is_file():
            raise ValueError(f"Missing registered photo: {source / images / name}")
    if len(names) < 2:
        raise ValueError("Held-out training requires at least two registered images")
    return {"policy": "sorted image_name; index % 8 == 0 is test",
            "train": [name for i, name in enumerate(names) if i % 8 != 0],
            "test": [name for i, name in enumerate(names) if i % 8 == 0]}


def prepare_dataset(source: Path, destination: Path, images: str = "images") -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Staged dataset must be outside the source dataset tree")
    errors = validation_errors(Scene(source.name, source))
    if errors:
        raise ValueError("; ".join(errors))
    if destination.exists():
        raise ValueError(f"Refusing to reuse an existing staged dataset: {destination}")
    if Path(images).is_absolute() or ".." in Path(images).parts:
        raise ValueError("images must be a relative folder inside the scene")
    image_source = source / images
    if not image_source.is_dir():
        raise ValueError(f"Missing photo directory: {image_source}")
    sparse = destination / "sparse/0"
    sparse.mkdir(parents=True)

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return str(dst)

    # Photo files are never written by the training entry points. Metadata is
    # copied, so creating/replacing points3D.ply cannot change the source cloud.
    shutil.copytree(image_source, destination / images, copy_function=link_or_copy)
    for stem in ("cameras", "images", "points3D"):
        for suffix in (".bin", ".txt"):
            path = source / "sparse/0" / (stem + suffix)
            if path.is_file():
                shutil.copy2(path, sparse / path.name)
    cached = source / "sparse/0/points3D.ply"
    if cached.is_file():
        shutil.copy2(cached, sparse / cached.name)


def read_rgb_ply(path: Path):
    import numpy as np
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"]
    required = ("x", "y", "z", "red", "green", "blue")
    if not set(required).issubset(vertex.data.dtype.names or ()):
        raise ValueError(f"Expected XYZ/RGB point cloud: {path}")
    xyz = np.column_stack([vertex[key] for key in required[:3]])
    rgb = np.column_stack([vertex[key] for key in required[3:]])
    return xyz, rgb


def source_cloud(source: Path):
    import numpy as np
    sparse = source / "sparse/0"
    cached = sparse / "points3D.ply"
    if cached.is_file():
        xyz, rgb = read_rgb_ply(cached)
        path = cached
    else:
        reader = colmap_reader()
        path = sparse / "points3D.bin"
        try:
            xyz, rgb, _ = reader.read_points3D_binary(path)
        except (OSError, ValueError, EOFError, struct.error):
            path = sparse / "points3D.txt"
            xyz, rgb, _ = reader.read_points3D_text(path)
    xyz, rgb = np.asarray(xyz), np.asarray(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("Source COLMAP cloud is empty or malformed")
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    valid &= ((rgb >= 0) & (rgb <= 255)).all(axis=1)
    if not valid.any():
        raise ValueError("Source COLMAP cloud has no finite XYZ/RGB points")
    return xyz[valid], rgb[valid].astype(np.uint8), path


def write_rgb_ply(path: Path, xyz, rgb) -> None:
    import numpy as np
    from plyfile import PlyData, PlyElement
    fields = [(key, "f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(key, "u1") for key in ("red", "green", "blue")]
    vertex = np.zeros(len(xyz), dtype=fields)
    for index, key in enumerate(("x", "y", "z")):
        vertex[key] = xyz[:, index]
    for index, key in enumerate(("red", "green", "blue")):
        vertex[key] = rgb[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def export_anchors(source: Path, anchors_ply: Path, output: Path,
                   max_color_distance: float = 0.0, images: str = "images") -> dict:
    import numpy as np
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    if not math.isfinite(max_color_distance) or max_color_distance < 0:
        raise ValueError("max_color_distance must be finite and nonnegative")
    paths = locations(output)
    if paths["report"].exists():
        raise ValueError("Initialization report already exists; use a fresh output")
    vertex = PlyData.read(str(anchors_ply))["vertex"]
    fields = vertex.data.dtype.names or ()
    if (not {"x", "y", "z"}.issubset(fields)
            or not any(name.startswith("f_anchor_feat_") for name in fields)
            or not any(name.startswith("f_offset_") for name in fields)):
        raise ValueError("Expected an official Scaffold anchor PLY, not an XYZ/RGB or 3DGS PLY")
    xyz = np.column_stack([vertex[key] for key in ("x", "y", "z")]).astype(np.float32)
    original_count = len(xyz)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    finite_count = len(xyz)
    _, first = np.unique(xyz, axis=0, return_index=True)
    xyz = xyz[np.sort(first)]
    unique_count = len(xyz)
    if len(xyz) < 2:
        raise ValueError("At least two finite, distinct anchors are required")
    reference_xyz, reference_rgb, reference_path = source_cloud(source)
    distances, nearest = cKDTree(reference_xyz).query(xyz, workers=1)
    if max_color_distance > 0:
        keep = distances <= max_color_distance
        xyz, nearest, distances = xyz[keep], nearest[keep], distances[keep]
    if len(xyz) < 2:
        raise ValueError("Color-distance filtering left fewer than two anchors")
    split = split_manifest(source, images)
    training_receipt = None
    if paths["training"].is_file():
        training_receipt = read_json(paths["training"])
        if (training_receipt.get("status") != "success"
                or training_receipt.get("split") != split
                or training_receipt.get("source_fingerprint") != input_fingerprint(source, split, images)
                or training_receipt.get("anchors_sha256") != digest(anchors_ply)):
            raise ValueError("Scaffold training receipt does not match the source or anchor PLY")
    prepare_dataset(source, paths["dataset"], images)
    target = paths["dataset"] / "sparse/0/points3D.ply"
    write_rgb_ply(target, xyz, reference_rgb[nearest])
    report = {
        "version": VERSION, "method": METHOD, "status": "success",
        "source_scene": str(source.resolve()),
        "training_source": str(paths["dataset"].resolve()),
        "anchors_ply": str(anchors_ply.resolve()), "anchors_sha256": digest(anchors_ply),
        "point_selection": "Scaffold anchor XYZ; neural Gaussian offsets are not exported",
        "color_initialization": "nearest source COLMAP point RGB",
        "reference_cloud": str(reference_path.resolve()),
        "reference_cloud_sha256": digest(reference_path),
        "coordinate_transform": "none; original COLMAP coordinates",
        "input_anchor_count": original_count,
        "removed_nonfinite": original_count - finite_count,
        "removed_duplicates": finite_count - unique_count,
        "removed_by_color_distance": unique_count - len(xyz),
        "max_color_distance": max_color_distance,
        "color_distance_mean": float(distances.mean()),
        "color_distance_max": float(distances.max()),
        "point_count": len(xyz), "initial_ply": str(target.resolve()),
        "initial_ply_sha256": digest(target), "split": split,
    }
    if training_receipt is not None:
        report["scaffold_training"] = training_receipt
    write_json(paths["report"], report)
    return report


def verify_export(output: Path, source: Path) -> dict:
    paths = locations(output)
    report = read_json(paths["report"])
    expected = paths["dataset"] / "sparse/0/points3D.ply"
    if (report.get("version") != VERSION or report.get("status") != "success"
            or report.get("point_count", 0) < 2
            or Path(report["source_scene"]).resolve() != source.resolve()
            or Path(report["initial_ply"]).resolve() != expected.resolve()
            or Path(report["training_source"]).resolve() != paths["dataset"].resolve()
            or digest(expected) != report.get("initial_ply_sha256")):
        raise ValueError("Scaffold initialization output is incomplete or does not match this scene")
    return report


def stage_commands(config: dict, source: Path, output: Path):
    from research_pipeline.config import repository_parser, resolve_path
    options = config["scaffold_initialization"]
    training = repository_parser("training").parse_args(config["training"]["extra_args"])
    paths = locations(output)
    common = [sys.executable, "-m", "research_pipeline.scaffold_initialization"]
    train = [*common, "train", "--scene", str(source), "--output", str(output),
             "--repository", str(resolve_path(options["repository"])),
             "--python-executable", options["python_executable"],
             "--iterations", str(options["iterations"]), "--voxel-size", str(options["voxel_size"]),
             "--resolution", str(training.resolution), "--images", training.images,
             "--data-device", training.data_device,
             "--expected-commit", options.get("expected_commit", "")]
    if training.eval:
        train.append("--eval")
    if training.white_background:
        train.append("--white-background")
    export = [*common, "export", "--scene", str(source), "--output", str(output),
              "--anchors-ply", str(paths["model"] / "point_cloud"
                                   / f"iteration_{options['iterations']}" / "point_cloud.ply"),
              "--max-color-distance", str(options["max_color_distance"]),
              "--images", training.images]
    return {"scaffold_training": train, "scaffold_export": export}, paths["dataset"]


def train_scaffold(args) -> int:
    if not args.eval:
        raise ValueError("Scaffold initialization requires --eval to preserve held-out evaluation")
    repository = args.repository.resolve()
    state = repository_state(repository, args.expected_commit)
    executable = resolve_python(args.python_executable)
    paths = locations(args.output)
    if paths["model"].exists() or paths["training"].exists():
        raise ValueError("Refusing to reuse a Scaffold model; use a fresh attempt output")
    split = split_manifest(args.scene, args.images)
    fingerprint = input_fingerprint(args.scene, split, args.images)
    prepare_dataset(args.scene, paths["source"], args.images)
    if input_fingerprint(paths["source"], split, args.images) != fingerprint:
        raise ValueError("Staged Scaffold input does not match the source dataset")
    command = [executable, "-B", str(REPO / "research_pipeline/scaffold_train.py"),
               "--repository", str(repository),
               "-s", str(paths["source"].resolve()), "-m", str(paths["model"].resolve()),
               "--images", args.images, "--resolution", str(args.resolution),
               "--data_device", args.data_device,
               "--iterations", str(args.iterations), "--voxel_size", str(args.voxel_size),
               "--eval"]
    if args.white_background:
        command.append("--white_background")
    metadata = {"version": VERSION, "status": "running", "command": command,
                **state, "expected_commit": args.expected_commit,
                "source_fingerprint": fingerprint, "iterations": args.iterations,
                "voxel_size": args.voxel_size, "split": split,
                "seed": 0, "seed_policy": "official Scaffold safe_state default",
                "launcher": "training_only",
                "launcher_sha256": digest(REPO / "research_pipeline/scaffold_train.py"),
                "post_training_evaluation": False, "viewer": False,
                "gpu_selection": "inherited CUDA_VISIBLE_DEVICES"}
    write_json(paths["training"], metadata)
    started = time.monotonic()
    try:
        process = subprocess.run(command, cwd=repository, check=False)
        metadata.update(returncode=process.returncode,
                        status="failed" if process.returncode else "success")
        final = paths["model"] / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
        if process.returncode == 0:
            if not final.is_file():
                raise RuntimeError(f"Scaffold did not produce the final anchor PLY: {final}")
            if input_fingerprint(args.scene, split, args.images) != fingerprint:
                raise RuntimeError("Source dataset changed while Scaffold was training")
            metadata["anchors_sha256"] = digest(final)
        if process.returncode:
            metadata["error"] = f"Scaffold exited with code {process.returncode}"
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=str(exc) or type(exc).__name__)
        raise
    finally:
        metadata["duration_seconds"] = time.monotonic() - started
        write_json(paths["training"], metadata)
    if metadata["status"] != "success":
        raise RuntimeError(metadata["error"])
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("train", "export"):
        sub = phases.add_parser(phase)
        sub.add_argument("--scene", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--images", default="images")
        if phase == "train":
            sub.add_argument("--repository", type=Path, required=True)
            sub.add_argument("--python-executable", required=True)
            sub.add_argument("--expected-commit", default="")
            sub.add_argument("--data-device", default="cuda")
            sub.add_argument("--iterations", type=int, default=30000)
            sub.add_argument("--voxel-size", type=float, default=0.001)
            sub.add_argument("--resolution", type=int, default=1920)
            sub.add_argument("--eval", action="store_true")
            sub.add_argument("--white-background", action="store_true")
        else:
            sub.add_argument("--anchors-ply", type=Path, required=True)
            sub.add_argument("--max-color-distance", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if args.phase == "train":
            if args.iterations <= 0 or not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
                raise ValueError("iterations and voxel-size must be positive")
            return train_scaffold(args)
        report = export_anchors(args.scene, args.anchors_ply, args.output,
                                args.max_color_distance, args.images)
        print(json.dumps(report, indent=2))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Scaffold initialization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
