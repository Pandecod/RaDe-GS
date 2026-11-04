"""Validate and import original metric.py results; no metric recomputation."""
from __future__ import annotations
import argparse
import math
import ast
import json
from pathlib import Path
from research_pipeline.utils import write_json

def rendered_sets(output: Path) -> dict[str, tuple[Path, list[Path]]]:
    result = {}
    for split in ("train", "test"):
        candidates = []
        for folder in (output / split).glob("ours_*"):
            try:
                iteration = int(folder.name.removeprefix("ours_"))
            except ValueError:
                continue
            if (folder / "renders").is_dir():
                candidates.append((iteration, folder))
        if candidates:
            folder = max(candidates)[1]
            images = sorted((folder / "renders").glob("*.png"))
            result[split] = (folder, images)
    return result

def render_statistics(output: Path, duration: float) -> dict:
    count = sum(len(images) for _, images in rendered_sets(output).values())
    return {"image_count": count, "seconds_per_image": duration / count if count else None,
            "fps": count / duration if count and duration > 0 else None}

def evaluation_pairs(output: Path) -> tuple[str, Path, list[tuple[Path, Path]]]:
    sets = rendered_sets(output)
    if "test" not in sets or not sets["test"][1]:
        raise ValueError("No rendered test images. Train a new model with --eval; train metrics are not test metrics.")
    folder, images = sets["test"]
    pairs = []
    for rendered in images:
        gt = folder / "gt" / rendered.name
        if not gt.is_file():
            raise ValueError(f"Missing ground truth image: {gt}")
        pairs.append((rendered, gt))
    if {p.name for p in (folder / "gt").glob("*.png")} != {p.name for p in images}:
        raise ValueError("Test render and ground truth filenames do not match")
    return "test", folder, pairs


def require_eval_training(output: Path) -> None:
    # Inspect Namespace(...) as syntax, never execute the saved configuration.
    try:
        node = ast.parse((output / "cfg_args").read_text(encoding="utf-8").strip(), mode="eval").body
        enabled = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "Namespace"
                   and any(k.arg == "eval" and isinstance(k.value, ast.Constant)
                           and k.value.value is True for k in node.keywords))
    except (OSError, SyntaxError) as exc:
        raise ValueError("Cannot verify the saved training split in cfg_args") from exc
    if not enabled:
        raise ValueError("cfg_args must record eval=True from training. Do not relabel an all-training model.")


def prepare_native_evaluation(output: Path) -> None:
    require_eval_training(output)
    evaluation_pairs(output)
    # metric.py catches its own errors, so stale output must never count as success.
    if any((output / name).exists() for name in ("results.json", "per_view.json")):
        raise ValueError("Native metric outputs already exist; use an isolated run to avoid accepting stale results.")


def collect_native_metrics(output: Path, scene_name: str) -> dict:
    require_eval_training(output)
    split, folder, pairs = evaluation_pairs(output)
    try:
        aggregate = json.loads((output / "results.json").read_text(encoding="utf-8"))
        per_view = json.loads((output / "per_view.json").read_text(encoding="utf-8"))
        values = aggregate[folder.name]
        views = per_view[folder.name]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("metric.py did not produce complete results.json and per_view.json for the test iteration") from exc
    names = {a.name for a, _ in pairs}
    metrics, warnings = {}, []
    def valid(value, name):
        return (type(value) in (int, float) and not math.isnan(value)
                and (math.isfinite(value) or (name == "PSNR" and value > 0)))
    for name in ("PSNR", "SSIM", "LPIPS"):
        value = values.get(name) if isinstance(values, dict) else None
        images = views.get(name) if isinstance(views, dict) else None
        if (not valid(value, name) or not isinstance(images, dict)
                or set(images) != names or not all(valid(v, name) for v in images.values())):
            raise ValueError(f"Incomplete or invalid native {name} results")
        metrics[name.lower()] = value if math.isfinite(value) else None
        if not math.isfinite(value):
            warnings.append("Native PSNR is positive infinity; stored as null in the report.")
    result = {"scene_name": scene_name, "object_name": scene_name, **metrics,
              "image_count": len(pairs), "split": split, "method": folder.name,
              "evaluator": "metric.py", "source_results": str(output / "results.json"),
              "source_per_view": str(output / "per_view.json"), "warnings": warnings}
    write_json(output / "research/image_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-name", required=True)
    args = parser.parse_args()
    collect_native_metrics(args.output, args.scene_name)


if __name__ == "__main__":
    main()
