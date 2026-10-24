"""Configuration validation; no CUDA or model imports."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import math
from pathlib import Path
from typing import Any
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "project": {"name": "radegs_togog_bali", "dataset": "Dataset Togog Bali"},
    "dataset": {"root": "/home/pande/dataset_togog_bali"},
    "output": {"root": "output/togog_bali"},
    "results": {"csv": "research_results/experiment_results.csv"},
    "colmap": {"executable": "colmap", "camera_model": "OPENCV", "use_gpu": True},
    "training": {"extra_args": []},
    "rendering": {"enabled": True, "extra_args": []},
    "image_metrics": {"enabled": True},
    "mesh": {"enabled": True, "resolution": 2, "num_cluster": 8,
             "depth_ratio": 0.5, "extra_args": []},
    "resource_monitor": {"interval_seconds": 5.0},
    "resource_guard": {"enabled": True, "cooldown_before_mesh_seconds": 5,
                       "minimum_free_ram_gb": 4, "minimum_free_vram_mb": 1024,
                       "wait_if_memory_low": True, "check_interval_seconds": 10,
                       "max_wait_seconds": 300},
    "geometry_evaluation": {"enabled": False, "gt_root": None,
                            "sample_points": 100000, "distance_threshold": None,
                            "seed": 0},
}

def repository_parser(stage: str) -> argparse.ArgumentParser:
    # arguments/__init__.py imports only stdlib. Never import train/render/mesh.
    spec = importlib.util.spec_from_file_location("radegs_cli_arguments", REPO / "arguments/__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    module.ModelParams(parser)
    module.PipelineParams(parser)
    if stage == "training":
        module.OptimizationParams(parser)
    script = {"training": "train.py", "rendering": "render.py", "mesh": "mesh_extract.py"}[stage]
    tree = ast.parse((REPO / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "parser" and node.func.attr == "add_argument"):
            args = [ast.literal_eval(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg == "type" and isinstance(kw.value, ast.Name):
                    kwargs[kw.arg] = {"int": int, "float": float, "str": str}[kw.value.id]
                else:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
            parser.add_argument(*args, **kwargs)
    return parser

def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()

def validate(config: dict[str, Any]) -> dict[str, Any]:
    for section, defaults in DEFAULTS.items():
        values = config[section]
        for key, default in defaults.items():
            value = values[key]
            if isinstance(default, bool) and type(value) is not bool:
                raise ValueError(f"{section}.{key} must be boolean")
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{section}.{key} must be finite numeric")
                if value < 0:
                    raise ValueError(f"{section}.{key} must not be negative")
    for section, key in (("dataset", "root"), ("output", "root"), ("results", "csv")):
        value = config[section][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} must be a nonempty path")
    for key in ("name", "dataset"):
        if not isinstance(config["project"][key], str):
            raise ValueError(f"project.{key} must be text")
    for key in ("executable", "camera_model"):
        value = config["colmap"][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"colmap.{key} must be nonempty text")
    for section, key in (("resource_monitor", "interval_seconds"), ("resource_guard", "check_interval_seconds"),
                         ("geometry_evaluation", "sample_points")):
        if config[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    geometry = config["geometry_evaluation"]
    for key in ("sample_points", "seed"):
        if type(geometry[key]) is not int:
            raise ValueError(f"geometry_evaluation.{key} must be an integer")
    threshold = geometry["distance_threshold"]
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("distance_threshold must be null or positive and finite")
    if geometry["enabled"] and (not isinstance(geometry["gt_root"], str) or not geometry["gt_root"].strip()):
        raise ValueError("Geometry evaluation requires gt_root")
    if geometry["enabled"] and not config["mesh"]["enabled"]:
        raise ValueError("Geometry evaluation requires mesh.enabled")
    mesh = config["mesh"]
    if (mesh["resolution"], mesh["num_cluster"], mesh["depth_ratio"]) != (2, 8, 0.5):
        raise ValueError("Verified mesh parameters are fixed: -r 2 --num_cluster 8 --depth_ratio 0.5")
    for stage in ("training", "rendering", "mesh"):
        extras = config[stage]["extra_args"]
        if not isinstance(extras, list) or any(not isinstance(a, str) for a in extras):
            raise ValueError(f"{stage}.extra_args must be a list of strings")
        parser = repository_parser(stage)
        try:
            parsed, unknown = parser.parse_known_args(extras)
        except (argparse.ArgumentError, SystemExit) as exc:
            raise ValueError(f"Invalid {stage}.extra_args: {exc}") from exc
        if unknown:
            raise ValueError(f"Unknown {stage} arguments: {unknown}")
        # Reject overrides even when an attached short form or abbreviation is used.
        for action in parser._actions:
            forbidden = {"source_path", "model_path"}
            if stage == "mesh":
                forbidden |= {"resolution", "num_cluster", "depth_ratio"}
            if action.dest in forbidden:
                for token in extras:
                    if any(token == opt or token.startswith(opt + "=")
                           or (len(opt) == 2 and token.startswith(opt) and not token.startswith("--"))
                           for opt in action.option_strings):
                        raise ValueError(f"{stage}.extra_args may not override {action.dest}")
        if stage == "rendering" and parsed.skip_train and parsed.skip_test and config[stage]["enabled"]:
            raise ValueError("Rendering cannot skip both train and test")
    dataset, output = resolve_path(config["dataset"]["root"]), resolve_path(config["output"]["root"])
    results = resolve_path(config["results"]["csv"])
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Dataset and output roots must be separate directory trees")
    if results == dataset or dataset in results.parents:
        raise ValueError("Master CSV must not be stored inside the dataset root")
    return config

def load_config(path: Path) -> dict[str, Any]:
    import copy
    supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("Config must be a YAML mapping")
    config = copy.deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section: {section}")
        unknown = set(values) - set(config[section])
        if unknown:
            raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
        config[section].update(values)
    return validate(config)
