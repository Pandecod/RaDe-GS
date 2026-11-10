"""Read-only checks before running a Scaffold initialization experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from research_pipeline.config import REPO, load_config, resolve_path
from research_pipeline.scaffold_initialization import repository_state, resolve_python

PROBE = """
import importlib, json, sys
repository, names = sys.argv[1], json.loads(sys.argv[2])
if repository:
    sys.path.insert(0, repository)
result = {"python": sys.executable, "modules": {}, "cuda_available": False}
for name in names:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {"available": True, "path": getattr(module, "__file__", None)}
        if name == "torch":
            result["pytorch"] = module.__version__
            result["cuda_runtime"] = module.version.cuda
            result["cuda_available"] = bool(module.cuda.is_available())
            result["gpus"] = [module.cuda.get_device_name(i) for i in range(module.cuda.device_count())]
    except Exception as error:
        result["modules"][name] = {"available": False, "error": str(error)}
result["ready"] = result["cuda_available"] and all(item["available"] for item in result["modules"].values())
print(json.dumps(result))
"""


def probe_python(executable: str, role: str, repository: Path | None = None) -> dict:
    required = ["torch", "numpy", "plyfile", "diff_gaussian_rasterization", "simple_knn._C"]
    if role == "scaffold":
        required += ["torchvision", "torch_scatter", "einops", "wandb"]
    else:
        required += ["scipy", "yaml", "psutil", "open3d", "warp_patch_ncc"]
    try:
        executable = resolve_python(executable)
        process = subprocess.run(
            [executable, "-B", "-c", PROBE, str(repository) if repository else "",
             json.dumps(required)], capture_output=True, text=True, timeout=45)
        if process.returncode:
            return {"ready": False, "error": process.stderr.strip() or
                    f"Python probe exited with code {process.returncode}"}
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return {"ready": False, "error": str(exc)}


def probe_training_imports(executable: str, repository: Path) -> dict:
    """Exercise the actual script entry point; --help exits before training/output."""
    try:
        process = subprocess.run(
            [resolve_python(executable), "-B", str(REPO / "research_pipeline/scaffold_train.py"),
             "--repository", str(repository.resolve()), "--help"],
            cwd=repository, capture_output=True, text=True, timeout=45)
        result = {"ready": process.returncode == 0, "returncode": process.returncode}
        if process.returncode:
            result["error"] = process.stderr.strip() or process.stdout.strip()
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ready": False, "error": str(exc)}


def inspect(config_path: Path, scene_name: str | None = None) -> dict:
    from research_pipeline.scene_discovery import discover, validation_errors
    from research_pipeline.scaffold_initialization import split_manifest
    config = load_config(config_path)
    options = config["scaffold_initialization"]
    report = {"ready": False, "config": str(config_path.resolve()),
              "experiment_enabled": options["enabled"]}
    repository = resolve_path(options["repository"]) if options["repository"] else None
    try:
        if repository is None:
            raise ValueError("No Scaffold repository configured")
        report["repository"] = repository_state(repository, options["expected_commit"])
        extension_files = [
            "submodules/diff-gaussian-rasterization/setup.py",
            "submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp",
            "submodules/simple-knn/setup.py"]
        missing = [name for name in extension_files if not (repository / name).is_file()]
        if missing:
            report["repository"]["ready"] = False
            report["repository"]["error"] = "Missing extension sources: " + ", ".join(missing)
        else:
            report["repository"]["ready"] = True
    except (OSError, ValueError) as exc:
        report["repository"] = {"ready": False, "error": str(exc)}
    report["radegs_environment"] = probe_python(sys.executable, "radegs")
    report["scaffold_environment"] = probe_python(
        options["python_executable"], "scaffold", repository)
    report["training_imports"] = {"ready": False, "error": "Repository or Scaffold environment is not ready"}
    if report["repository"].get("ready") and report["scaffold_environment"].get("ready"):
        report["training_imports"] = probe_training_imports(options["python_executable"], repository)
    try:
        scenes = discover(resolve_path(config["dataset"]["root"]), scene_name)
        checked = []
        for scene in scenes:
            errors = validation_errors(scene)
            entry = {"scene": scene.name, "path": str(scene.path), "errors": errors}
            if not errors:
                try:
                    split = split_manifest(scene.path)
                    entry.update(train_images=len(split["train"]), test_images=len(split["test"]))
                except (ImportError, OSError, ValueError, KeyError) as exc:
                    entry["errors"].append(str(exc))
            checked.append(entry)
        report["dataset"] = {"ready": all(not item["errors"] for item in checked),
                             "scenes": checked}
    except (OSError, ValueError) as exc:
        report["dataset"] = {"ready": False, "error": str(exc)}
    report["ready"] = options["enabled"] and all(report[key].get("ready", False) for key in (
        "repository", "radegs_environment", "scaffold_environment", "training_imports", "dataset"))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.config, args.scene)
    except (ImportError, OSError, ValueError) as exc:
        report = {"ready": False, "error": str(exc)}
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
