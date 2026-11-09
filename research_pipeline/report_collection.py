"""Collect existing experiment reports, including nested runs and successful mesh retries."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from research_pipeline.config import load_config, resolve_path
from research_pipeline.mesh_recovery import prepare_mesh_recovery
from research_pipeline.report_index import central_directory, refresh_catalog, write_report_index
from research_pipeline.reporting import export, read_json, update_result_tables, write_table
from research_pipeline.utils import write_json

HISTORY_FIELDS = ["Folder", "Metode", "Status", "Jenis", "Waktu sumber", "Waktu hasil",
                  "Split evaluasi", "Tahap gagal", "Pesan", "File sumber", "Folder output", "Terpilih"]
SUCCESS = {"success", "success_mesh_recovered"}


def stamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp harus berupa string berzona waktu")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return result


def inspect_outputs(root):
    """Inspect known model locations only; never walk heavy render/PLY payload trees."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Folder hasil belum ada: {root}")
    summaries = sorted(set(root.glob("*/research/summary.json")) |
                       set(root.glob("*/runs/*/research/summary.json")))
    candidates, history = [], []

    def record(path, kind, summary=None, problem=None):
        summary = summary or {}
        source = summary.get("recovery", {})
        row = {
            "Folder": summary.get("scene_name", path.parent.parent.name),
            "Metode": summary.get("method", "RaDe-GS"),
            "Status": summary.get("status", "tidak_dapat_dibaca"),
            "Jenis": kind, "Waktu sumber": source.get("source_timestamp", summary.get("timestamp")),
            "Waktu hasil": summary.get("timestamp"),
            "Split evaluasi": summary.get("rendering", {}).get("split"),
            "Tahap gagal": summary.get("failed_stage"),
            "Pesan": problem or summary.get("error_message"), "File sumber": str(path),
            "Folder output": summary.get("output_path"), "Terpilih": False,
        }
        history.append(row)
        return row

    for path in summaries:
        model = path.parent.parent
        try:
            summary = read_json(path)
            if not isinstance(summary, dict) or not summary.get("scene_name") or not summary.get("output_path"):
                raise ValueError("Ringkasan tidak memiliki scene_name/output_path")
            source_time = stamp(summary["timestamp"])
            row = record(path, "pipeline", summary)
            if summary.get("status") in ("success", "failed"):
                candidates.append({"summary":summary,"output":model,"retry":None,"history":row,
                                   "source_time":source_time,"finished":source_time})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record(path, "pipeline", problem=str(exc))
            continue
        for result_file in sorted((model / "research").glob("mesh_retry_*/result.json")):
            try:
                raw = read_json(result_file)
                if not isinstance(raw, dict):
                    raise ValueError("Catatan retry harus berupa objek JSON")
                # Incomplete/failed retries are history, never replacement successes.
                if raw.get("status") != "success":
                    record(result_file, "retry mesh", {
                        "scene_name":summary["scene_name"],"method":summary.get("method", "RaDe-GS"),
                        "status":raw.get("status", "unknown"), "timestamp":raw.get("finished_utc"),
                        "output_path":raw.get("attempt_path"), "failed_stage":raw.get("failed_stage"),
                        "error_message":raw.get("error_message")})
                    continue
                asset_root, recovered = prepare_mesh_recovery(result_file)
                retry_row = record(result_file, "retry mesh", recovered)
                retry_row["Waktu sumber"] = summary["timestamp"]
                candidates.append({"summary":recovered,"output":asset_root,"retry":result_file.parent,
                                   "history":retry_row,"source_time":source_time,
                                   "finished":stamp(recovered["timestamp"])})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record(result_file, "retry ditolak", {
                    "scene_name":summary["scene_name"],"method":summary.get("method", "RaDe-GS"),
                    "status":"tidak_dapat_diverifikasi"}, str(exc))
    if not candidates:
        raise ValueError(f"Tidak ada summary run selesai yang dapat diekspor dari {root}")
    grouped = {}
    for item in candidates:
        summary = item["summary"]
        key = (summary["scene_name"], summary.get("method", "RaDe-GS"))
        grouped.setdefault(key, []).append(item)
    chosen = []
    for key, options in sorted(grouped.items()):
        successful = [item for item in options if item["summary"].get("status") in SUCCESS]
        pool = successful or options
        selected = max(pool, key=lambda item: (item["source_time"], item["finished"],
                                               str(item["summary"].get("output_path"))))
        selected["history"]["Terpilih"] = True
        chosen.append(selected)
    return chosen, history


def collect(config, *, pairs=5, dry_run=False):
    if pairs < 1:
        raise ValueError("--pairs harus minimal 1")
    output_root = resolve_path(config["output"]["root"])
    results = resolve_path(config["results"]["csv"])
    selected, history = inspect_outputs(output_root)
    print(f"Dataset: {config['project']['dataset']}")
    for item in selected:
        summary = item["summary"]
        print(f"  {summary['scene_name']} | {summary.get('method', 'RaDe-GS')} | {summary['status']} | {summary['output_path']}")
    issues = [row for row in history if row["Status"] in ("tidak_dapat_dibaca", "tidak_dapat_diverifikasi")]
    if issues:
        print(f"PERHATIAN: {len(issues)} catatan tidak dapat diverifikasi; lihat riwayat_percobaan.csv.")
    if dry_run:
        print("Dry-run: hanya membaca hasil; tidak membuat laporan atau mengubah berkas.")
        return {"selected":len(selected), "history":len(history), "issues":len(issues)}
    destination = central_directory(results, prefix="kumpulan_")
    outputs = [item["output"] for item in selected if item["retry"] is None]
    retries = [item["retry"] for item in selected if item["retry"] is not None]
    export(outputs, destination, pairs=pairs, mesh_retries=retries)
    write_table(destination / "riwayat_percobaan.csv", HISTORY_FIELDS, history)
    write_json(destination / "pengumpulan.json", {
        "output_root":str(output_root),"results_csv":str(results),
        "selection":"One result per scene and method. Prefer recorded successful runs. Choose the newest source training run; within that source choose the latest validated mesh recovery. No metric/visual quality selection.",
        "selected":[{"source":row["history"]["File sumber"],"experiment_id":row["summary"].get("experiment_id"),
                     "split":row["summary"].get("rendering", {}).get("split")} for row in selected],
        "history_count":len(history),"unverified_records":len(issues),
        "note":"No training, metric recalculation, source migration, or mesh/PLY duplication. Existing technical CSV and original summaries remain unchanged.",
    })
    details = read_json(destination / "detail_laporan.json")["experiments"]
    write_report_index(destination, details)
    update_result_tables(destination, results)
    entry = refresh_catalog(results.parent)
    print(f"Kumpulan {len(selected)} hasil: {destination}")
    print(f"Buka: {entry}")
    return {"directory":str(destination),"entry":str(entry),"selected":len(selected),"history":len(history)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=5,
                        help="Jumlah pasangan visual per objek; dipilih merata, default 5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        for path in args.config:
            collect(load_config(path), pairs=args.pairs, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Pengumpulan laporan gagal: {exc}\n")


if __name__ == "__main__":
    main()
