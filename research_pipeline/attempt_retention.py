"""Remove recorded failed attempts only after a replacement success is durable."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import shutil

from research_pipeline.config import load_config, resolve_path
from research_pipeline.results import read_rows, write_summary
from research_pipeline.utils import atomic_write, read_json


def safe_path(path: Path, root: Path) -> Path:
    """Reject paths outside the configured tree and redirected directories."""
    path, root = Path(path).absolute(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Cleanup target is outside its storage root: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"Cleanup target uses a link or junction: {part}")
    if path.resolve() != path:
        raise ValueError(f"Cleanup target is redirected: {path}")
    return path


def same_path(value, path: Path) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).resolve() == path


def failed_summary(path: Path, result: dict) -> dict | None:
    file = path / "research/summary.json"
    if not file.is_file():
        return None
    summary = read_json(file)
    if (summary.get("status") != "failed"
            or summary.get("experiment_id") != result["experiment_id"]
            or summary.get("scene_name") != result["scene_name"]
            or summary.get("method", "RaDe-GS") != result.get("method", "RaDe-GS")
            or not same_path(summary.get("output_path"), path)):
        return None
    # A successful mesh recovery still depends on the original model and images.
    for retry in (path / "research").glob("mesh_retry_*/result.json"):
        if read_json(retry).get("status") == "success":
            return None
    return summary


def verify_success(config: dict, result: dict, current: Path) -> None:
    saved = read_json(current / "research/summary.json")
    if (saved.get("status") != "success" or saved.get("failed_stage") is not None
            or saved.get("experiment_id") != result["experiment_id"]
            or not same_path(saved.get("output_path"), current)
            or saved.get("reporting", {}).get("status") != "complete"):
        raise ValueError("Replacement success and its report have not been saved completely")
    rows = read_rows(resolve_path(config["results"]["csv"]))
    if not any(row["status"] == "success"
               and row["experiment_id"] == result["experiment_id"]
               and same_path(row["output_path"], current) for row in rows):
        raise ValueError("Replacement success has not been recorded in the master CSV")
    iteration = saved.get("training", {}).get("final_iteration")
    if type(iteration) is not int or iteration <= 0:
        raise ValueError("Replacement success has no recorded final training iteration")
    required = [current / f"point_cloud/iteration_{iteration}/point_cloud.ply"]
    if config["mesh"]["enabled"]:
        required.append(current / "recon_post.ply")
    for path in required:
        safe_path(path, current)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError("Replacement model or mesh is missing; failed attempts are retained")


def cleanup_plan(config: dict, result: dict) -> dict:
    plan = {"outputs": [], "reports": [], "tables": [], "warnings": []}
    if result.get("status") != "success":
        return plan
    name = result["scene_name"]
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("Invalid scene name for cleanup")
    root = resolve_path(config["output"]["root"])
    base = safe_path(root / name, root)
    current = safe_path(Path(result["output_path"]), root)
    if current != base and not (current.parent == base / "runs"):
        raise ValueError("Replacement output is outside this scene's attempt folders")
    candidates = [base]
    runs = base / "runs"
    if runs.exists():
        safe_path(runs, root)
        candidates.extend(sorted(path for path in runs.iterdir() if path.is_dir()))
    for candidate in candidates:
        if candidate == current:
            continue
        try:
            candidate = safe_path(candidate, root)
            safe_path(candidate / "research", root)
            safe_path(candidate / "research/summary.json", root)
            if failed_summary(candidate, result) is not None:
                plan["outputs"].append(candidate)
        except (OSError, ValueError, TypeError) as exc:
            plan["warnings"].append(f"Attempt retained: {candidate}: {exc}")
    if not plan["outputs"]:
        return plan
    verify_success(config, result, current)
    master = resolve_path(config["results"]["csv"])
    from research_pipeline.reporting import FIELDS, SUMMARY_FIELDS, read_report_table
    for suffix, fields in (("_laporan.csv", FIELDS), ("_ringkasan.csv", SUMMARY_FIELDS)):
        table = master.with_name(master.stem + suffix)
        if not table.exists():
            continue
        previous = read_report_table(table, fields)
        retained = [row for row in previous if not (
            row.get("Status") == "failed"
            and any(same_path(row.get("Folder output"), old) for old in plan["outputs"]))]
        if len(retained) != len(previous):
            plan["tables"].append((table, fields, retained))
    report_root = master.parent / "laporan"
    # Standalone failed packages can be removed. Mixed exported collections are
    # independent snapshots and may contain successes belonging to other scenes.
    for metadata in sorted(report_root.glob("*/*/detail_laporan.json")):
        try:
            directory = safe_path(metadata.parent, report_root)
            details = read_json(metadata).get("experiments")
            if isinstance(details, list) and details and all(
                item.get("experiment_id") == result["experiment_id"]
                and item.get("table_row", {}).get("Status") == "failed"
                and any(same_path(item.get("original_output_path"), old)
                        for old in plan["outputs"])
                for item in details
            ):
                plan["reports"].append(directory)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            plan["warnings"].append(f"Report retained: {metadata.parent}: {exc}")
    return plan


def remove_attempt(path: Path, root: Path) -> None:
    """Keep nested runs and remove the failed summary after its payload."""
    path = safe_path(path, root)
    research = path / "research"
    for child in list(path.iterdir()):
        if child.name in ("runs", "research"):
            continue
        child = safe_path(child, root)
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    if research.exists():
        safe_path(research, root)
        for child in list(research.iterdir()):
            if child.name == "summary.json":
                continue
            child = safe_path(child, root)
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        (research / "summary.json").unlink()
        research.rmdir()
    if not any(path.iterdir()):
        path.rmdir()


def cleanup_failed_attempts(config: dict, result: dict, *, dry_run=False) -> dict:
    plan = cleanup_plan(config, result)
    removed = {"status": "dry_run" if dry_run else "complete",
               "outputs": [str(path) for path in plan["outputs"]],
               "reports": [str(path) for path in plan["reports"]],
               "warnings": plan["warnings"]}
    if dry_run or not plan["outputs"]:
        return removed
    root = resolve_path(config["output"]["root"])
    current = safe_path(Path(result["output_path"]), root)
    verify_success(config, result, current)
    # Recheck every candidate before changing any table or removing a directory.
    for path in plan["outputs"]:
        if failed_summary(safe_path(path, root), result) is None:
            raise ValueError(f"Attempt changed during cleanup: {path}")
    for table, fields, rows in plan["tables"]:
        def write(stream, fields=fields, rows=rows):
            stream.write("\ufeff")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        atomic_write(table, write)
    report_root = resolve_path(config["results"]["csv"]).parent / "laporan"
    for path in plan["reports"]:
        shutil.rmtree(safe_path(path, report_root))
    for path in sorted(plan["outputs"], key=lambda value: len(value.parts), reverse=True):
        remove_attempt(path, root)
        logging.info("Removed superseded failed attempt: %s", path)
    if plan["reports"]:
        from research_pipeline.report_index import refresh_catalog
        refresh_catalog(report_root.parent)
    return removed


def finalize_retention(config: dict, result: dict) -> dict:
    """Keep a valid successful experiment even if housekeeping cannot finish."""
    try:
        result["retention"] = cleanup_failed_attempts(config, result)
        result["warnings"].extend(result["retention"]["warnings"])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result["retention"] = {"status": "incomplete", "error": str(exc)}
        result["warnings"].append(f"Failed-attempt cleanup incomplete: {exc}")
        logging.warning("Failed-attempt cleanup incomplete: %s", exc)
    try:
        write_summary(result)
    except (OSError, ValueError) as exc:
        logging.warning("Could not save cleanup status: %s", exc)
    return result["retention"]


def cleanup_mesh_retries(attempt: Path, *, dry_run=False) -> dict:
    """Remove failed sibling retries while retaining their shared source model."""
    attempt = Path(attempt).absolute()
    model = attempt.parent.parent
    safe_path(attempt, model)
    if attempt.parent.name != "research" or not attempt.name.startswith("mesh_retry_"):
        raise ValueError("Expected a recorded mesh_retry folder")
    current = read_json(attempt / "result.json")
    info = {"status": "dry_run" if dry_run else "complete", "outputs": [], "warnings": []}
    if current.get("status") != "success":
        return info
    frozen = current.get("source_summary_snapshot", {})
    source_id = frozen.get("experiment_id")
    candidates = []
    for path in sorted(attempt.parent.glob("mesh_retry_*")):
        if path == attempt or not path.is_dir():
            continue
        try:
            path = safe_path(path, model)
            safe_path(path / "result.json", model)
            if not (path / "result.json").is_file():
                continue
            old = read_json(path / "result.json")
            snapshot = old.get("source_summary_snapshot", {})
            if (old.get("status") in ("failed", "interrupted")
                    and old.get("kind") == "mesh_retry"
                    and source_id and snapshot.get("experiment_id") == source_id
                    and snapshot.get("training") == frozen.get("training")
                    and snapshot.get("rendering") == frozen.get("rendering")
                    and same_path(old.get("model_path"), model)
                    and same_path(old.get("attempt_path"), path)):
                candidates.append(path)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            info["warnings"].append(f"Mesh retry retained: {path}: {exc}")
    if not candidates:
        return info
    from research_pipeline.mesh_recovery import prepare_mesh_recovery
    source, _ = prepare_mesh_recovery(attempt)
    if source != model:
        raise ValueError("Successful mesh recovery does not belong to this source model")
    safe_path(attempt / "recon_post.ply", attempt)
    info["outputs"] = [str(path) for path in candidates]
    if not dry_run:
        for path in candidates:
            # rmtree unlinks the retry's point_cloud link, preserving its target.
            shutil.rmtree(safe_path(path, model))
            logging.info("Removed failed mesh retry: %s", path)
    return info


def finalize_mesh_retention(attempt: Path, result: dict) -> dict:
    try:
        info = cleanup_mesh_retries(attempt)
        result["warnings"].extend(info["warnings"])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        info = {"status": "incomplete", "error": str(exc)}
        result["warnings"].append(f"Failed mesh retry cleanup incomplete: {exc}")
        logging.warning("Failed mesh retry cleanup incomplete: %s", exc)
    result["retention"] = info
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=[], type=Path)
    parser.add_argument("--mesh-retries", nargs="+", default=[], type=Path,
                        help="Successful retry folders; retain their shared source models")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Remove verified failed attempts")
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    args = parser.parse_args(argv)
    if not args.config and not args.mesh_retries:
        parser.error("Supply --config or --mesh-retries")
    incomplete = False
    try:
        for config_path in args.config:
            config = load_config(config_path)
            master = resolve_path(config["results"]["csv"])
            for row in read_rows(master):
                if row["status"] != "success":
                    continue
                result = read_json(Path(row["output_path"]) / "research/summary.json")
                if (result.get("status") != "success"
                        or result.get("experiment_id") != row["experiment_id"]):
                    raise ValueError("Master CSV and saved successful experiment do not match")
                info = (finalize_retention(config, result) if args.apply else
                        cleanup_failed_attempts(config, result, dry_run=True))
                incomplete |= info["status"] == "incomplete" or bool(info.get("warnings"))
                print(f"{config_path.name}: {result['scene_name']}: {info}")
        retries = list(args.mesh_retries)
        for config_path in args.config:
            config = load_config(config_path)
            root = resolve_path(config["output"]["root"])
            for pattern in ("*/research/mesh_retry_*/result.json",
                            "*/runs/*/research/mesh_retry_*/result.json"):
                for path in root.glob(pattern):
                    if read_json(path).get("status") == "success":
                        retries.append(path.parent)
        for retry in dict.fromkeys(retries):
            info = cleanup_mesh_retries(retry, dry_run=not args.apply)
            incomplete |= bool(info.get("warnings"))
            print(f"Mesh retry {retry}: {info}")
        return 1 if incomplete else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Failed-attempt cleanup stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
