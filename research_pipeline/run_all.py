"""CLI entry point. Planning never imports GPU libraries or starts subprocesses."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.config import load_config, resolve_path
from research_pipeline.scene_discovery import discover, start_validation_errors
from research_pipeline.commands import commands, display

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serial RaDe-GS Research Pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--start-from", choices=("training", "colmap"),
                        default="training",
                        help="First stage to run (default: training)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-scenes", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--resume", action="store_true")
    rerun.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        scenes = discover(resolve_path(config["dataset"]["root"]), args.scene)
        if args.list_scenes:
            print(f"Discovered {len(scenes)} scenes:")
            for index, scene in enumerate(scenes, 1):
                errors = start_validation_errors(scene, args.start_from)
                suffix = " [INVALID: " + "; ".join(errors) + "]" if errors else ""
                print(f"{index}. {scene.name}{suffix}")
            return 0
        if args.dry_run:
            print("RaDe-GS Research Pipeline")
            print(f"Dataset: {config['project']['dataset']}")
            print(f"Start from: {args.start_from}")
            print(f"Found {len(scenes)} datasets. Sequential execution.")
            invalid = False
            for index, scene in enumerate(scenes, 1):
                print(f"\n[{index}/{len(scenes)}] {scene.name}")
                errors = start_validation_errors(scene, args.start_from)
                if errors:
                    invalid = True
                    print("FAILED dataset_validation: " + "; ".join(errors))
                    continue
                output = resolve_path(config["output"]["root"]) / scene.name
                for stage, command in commands(
                        config, scene.path, output, args.start_from).items():
                    if stage == "mesh_extraction":
                        print("RESOURCE GUARD: " + ("enabled" if config["resource_guard"]["enabled"] else "disabled"))
                    label = {
                        "colmap": "COLMAP",
                        "training": "TRAIN",
                        "rendering": "RENDER",
                        "image_metrics": "IMAGE METRICS",
                        "mesh_extraction": "MESH",
                        "geometry_evaluation": "GEOMETRY",
                    }[stage]
                    print(label + ":\n" + display(command))
            print("\nDry-run only: no subprocesses, CUDA, output files, or results written.")
            return 1 if invalid else 0
        from research_pipeline.runner import run_batch
        return run_batch(config, scenes, resume=args.resume, force=args.force,
                         start_from=args.start_from)
    except KeyboardInterrupt:
        logging.error("\nPipeline stopped by user. Child processes have been cleaned up.")
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
