"""Export existing RaDe-GS results to a research table and paired PNGs. CPU only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import uuid

FIELDS = [
    "Waktu ringkasan (UTC+08)", "Folder", "Metode", "PSNR (dB)", "SSIM",
    "LPIPS", "Peak VRAM seluruh GPU (GiB)", "Training (menit)",
    "Split evaluasi", "Jumlah gambar evaluasi", "Status",
    "Jumlah foto terdaftar", "Foto training", "Foto test", "Gaussian awal",
    "Splats PLY akhir", "Iterasi PLY", "Folder output",
    "Status GT 3D", "Alasan GT dilewati", "Chamfer (satuan rekonstruksi)",
    "Precision geometri", "Recall geometri", "F1 geometri", "File GT 3D",
]
# Preserve the saved evaluation/GT table; provide a shorter main reading view.
SUMMARY_FIELDS = FIELDS[:18]
RESOURCE_FIELDS = [
    "Folder", "Metode", "Tahap", "Ketersediaan data", "Durasi (menit)",
    "RAM puncak proses (MiB)", "RAM rata-rata proses (MiB)",
    "VRAM puncak proses (MiB)", "VRAM rata-rata proses (MiB)",
    "Peak VRAM seluruh GPU saat training (GiB)", "File sampel resource",
    "Folder output",
]


def resource_rows(summary, mem):
    """Readable stage measurements; unavailable process VRAM stays unknown."""
    rows = []
    for key, label in (("training", "Training"), ("rendering", "Rendering"),
                       ("mesh_extraction", "Ekstraksi mesh")):
        stage = summary.get(key, {})
        seconds = numeric(stage.get("duration_seconds"))
        row = {
            "Folder": summary["scene_name"], "Metode": "RaDe-GS", "Tahap": label,
            "Durasi (menit)": seconds / 60 if seconds is not None else None,
            "RAM puncak proses (MiB)": numeric(stage.get("peak_ram_mb")),
            "RAM rata-rata proses (MiB)": numeric(stage.get("average_ram_mb")),
            "VRAM puncak proses (MiB)": numeric(stage.get("peak_vram_mb")),
            "VRAM rata-rata proses (MiB)": numeric(stage.get("average_vram_mb")),
            "Peak VRAM seluruh GPU saat training (GiB)":
                mem["gpu_total_peak_gib"] if key == "training" else None,
            "File sampel resource": stage.get("resource_csv"),
            "Folder output": summary.get("output_path"),
        }
        recorded = any(row[field] is not None for field in RESOURCE_FIELDS[4:10])
        row["Ketersediaan data"] = "Tercatat" if recorded else "Tidak tercatat"
        rows.append(row)
    return rows


def write_table(path, fields, rows):
    """Write an additional reading view without altering the source values."""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 6) if isinstance(value, float) else value
                             for key in fields if (value := row.get(key)) is not None})


REPORT_GUIDE = """# Petunjuk membaca laporan

Buka ringkasan_hasil.csv terlebih dahulu. Satu baris adalah satu run objek.
Kolom Folder adalah nama objek; Folder output membedakan lokasi setiap run.

| File/folder | Isi |
| --- | --- |
| ringkasan_hasil.csv | 18 kolom utama: waktu, objek, metrik, VRAM GPU, waktu training, foto, dan splats. |
| rincian_proses.csv | Waktu, RAM, dan VRAM untuk training, rendering, dan ekstraksi mesh. |
| tabel_evaluasi.csv | Tabel evaluasi lengkap, termasuk status dan hasil GT 3D. |
| detail_laporan.json | Data teknis, sumber angka, dan lokasi berkas pembanding. |
| *_daftar_foto.csv | Nama foto kamera, split, ukuran, dan pasangan gambar. |
| 01_<objek>/ dan seterusnya | Foto referensi di kiri dan hasil render di kanan. |

Waktu ringkasan adalah waktu pencatatan hasil, bukan waktu mulai training.
Training (menit) mencakup persiapan dan penyimpanan dalam proses training.
Durasi rendering terpisah dari waktu menjalankan metric.py.
RAM adalah sampel RSS proses beserta proses anak; MiB = 1024**2 byte.
VRAM proses hanya diisi jika monitor menyediakan datanya.
Peak VRAM seluruh GPU mencakup aplikasi lain; GiB = 1024**3 byte.
Angka puncak berasal dari sampel, sehingga puncak di antara sampel bisa terlewat.
Kolom kosong berarti tidak tersedia, bukan nol. Tidak tercatat bukan status gagal.
Nilai train berasal dari foto latih; nilai test berasal dari foto yang dipisahkan.
Status success menunjukkan tahapan wajib selesai, bukan jaminan kualitas visual.
Jika Excel menampilkan ######## pada waktu, lebarkan kolomnya.
CSV teknis utama tetap tersimpan sesuai results.csv pada konfigurasi.
"""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def maximum(rows, key, divisor=1):
    values = [numeric(row.get(key)) for row in rows]
    values = [v for v in values if v is not None and v >= 0]
    return max(values) / divisor if values else None


def read_csv(path):
    text = path.read_text(encoding="utf-8-sig")
    if "\x00" in text:
        raise ValueError(f"CSV contains null bytes; inspect the incomplete recording: {path}")
    import io
    rows = list(csv.DictReader(io.StringIO(text)))
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Incomplete CSV row: {path}")
    return rows


def memory_info(output):
    directory = output / "research" / "profiling"
    manifests = sorted(directory.glob("manifest_*.json"))
    if len(manifests) > 1:
        raise ValueError(f"Multiple profiler runs in {directory}; select an isolated run.")
    info = {
        "gpu_total_peak_gib": None, "pytorch_allocated_peak_gib": None,
        "pytorch_reserved_peak_gib": None, "gpu_samples": 0,
        "note": "GPU total includes other applications; a sampled maximum, not process VRAM.",
    }
    if not manifests:
        summary_path = output / "research/summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        resource = summary.get("training", {}).get("resource_csv")
        if not resource:
            info["note"] += " No training resource CSV available."
            return info
        # Also support copied Linux output folders inspected on Windows.
        filename = str(resource).replace(chr(92), "/").rsplit("/", 1)[-1]
        local = output / "research/logs" / filename
        resource_path = local if local.is_file() else Path(resource)
        if not resource_path.is_file():
            info["note"] += " Training resource CSV is missing."
            return info
        rows = [r for r in read_csv(resource_path) if r.get("device_status") == "available"]
        devices = {r.get("gpu_uuid") or r.get("gpu_index") for r in rows}
        if len(devices) == 1:
            info["gpu_total_peak_gib"] = maximum(rows, "device_vram_used_mb", 1024)
            info["gpu_samples"] = sum(numeric(r.get("device_vram_used_mb")) is not None for r in rows)
        elif len(devices) > 1:
            info["note"] += " Multiple GPUs: total peak withheld until the training GPU is identified."
        info["gpu_csv"] = str(resource_path)
        info["note"] += " External training monitor; no internal training profiler."
        return info
    manifest = read_json(manifests[0])
    run_id = manifest["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid profiler run ID")
    gpu = directory / f"gpu_{run_id}.csv"
    training = directory / f"training_{run_id}.csv"
    info["run_id"] = run_id
    if gpu.exists():
        rows = read_csv(gpu)
        rows = [r for r in rows if r.get("device_status") == "available"]
        devices = {r.get("gpu_uuid") or r.get("gpu_index") for r in rows}
        if len(devices) > 1:
            info["note"] += " Multiple GPUs: total peak withheld until the training GPU is identified."
        else:
            info["gpu_total_peak_gib"] = maximum(rows, "device_vram_used_mb", 1024)
            info["gpu_samples"] = sum(numeric(r.get("device_vram_used_mb")) is not None for r in rows)
        info["gpu_csv"] = str(gpu)
    if training.exists():
        rows = read_csv(training)
        if any(r.get("run_id") != run_id for r in rows):
            raise ValueError("Training CSV run ID does not match manifest")
        info["pytorch_allocated_peak_gib"] = maximum(rows, "cuda_peak_allocated_bytes", 1024**3)
        info["pytorch_reserved_peak_gib"] = maximum(rows, "cuda_peak_reserved_bytes", 1024**3)
        info["training_csv"] = str(training)
    return info


def comparison_images(output, summary, target, requested):
    rendering = summary.get("rendering", {})
    split = rendering.get("split")
    method = rendering.get("method")
    if split not in ("train", "test") or not re.fullmatch(r"ours_\d+", method or ""):
        return {"status": "unavailable", "reason": "No evaluated split/iteration in summary", "files": []}
    folder = output / split / method
    gt, renders = folder / "gt", folder / "renders"
    files = sorted(renders.glob("*.png"))
    expected = rendering.get("evaluation_image_count")
    if not files:
        return {"status": "missing_images", "required": [str(gt), str(renders)], "files": []}
    if expected is not None and len(files) != expected:
        raise ValueError(f"Expected {expected} renders but found {len(files)} in {renders}")
    missing = [f.name for f in files if not (gt / f.name).is_file()]
    if missing:
        raise ValueError(f"Missing GT pairs: {missing[:5]}")
    count = min(requested, len(files))
    indices = [0] if count == 1 else [round(i * (len(files)-1)/(count-1)) for i in range(count)]
    from PIL import Image, ImageDraw
    target.mkdir(parents=True, exist_ok=False)
    generated = []
    for index in indices:
        name = files[index].name
        with Image.open(gt / name) as left, Image.open(renders / name) as right:
            left, right = left.convert("RGB"), right.convert("RGB")
            if left.size != right.size:
                raise ValueError(f"Image dimensions differ: {name}")
            width, height = left.size
            canvas = Image.new("RGB", (2*width, height+48), "white")
            canvas.paste(left, (0, 48))
            canvas.paste(right, (width, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 15), f"Reference ({split}) | {name}", fill="black")
            draw.text((width+12, 15), f"RaDe-GS render | {method} | {name}", fill="black")
            destination = target / f"{Path(name).stem}_comparison.png"
            canvas.save(destination)
        generated.append(str(destination))
    return {
        "status": "complete", "split": split, "method": method,
        "selection": "Evenly spaced indices in sorted render filenames; no quality selection.",
        "note": "RGB pixels preserved without resizing or brightness adjustment. No per-pair metrics recomputed.",
        "files": generated,
    }



def ply_count(output, summary):
    """Read Gaussian vertex count from the final saved PLY header, never a mesh."""
    iteration = summary.get("training", {}).get("final_iteration")
    if not isinstance(iteration, int) or iteration < 0:
        return {"status": "unavailable", "count": None, "reason": "No final training iteration"}
    candidates = [output / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                  output / f"iteration_{iteration}" / "point_cloud.ply"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {"status": "missing", "count": None, "iteration": iteration}
    with path.open("rb") as stream:
        header = []
        for _ in range(1024):
            line = stream.readline(4096)
            if not line or len(line) >= 4096:
                raise ValueError(f"Invalid PLY header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        else:
            raise ValueError(f"PLY header exceeds limit: {path}")
        if header[0] != "ply" or header[-1] != "end_header":
            raise ValueError(f"Invalid PLY: {path}")
    properties = {line.split()[-1] for line in header if line.startswith("property ")}
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "f_dc_0"}
    if not required <= properties:
        raise ValueError(f"Expected Gaussian PLY properties: {path}")
    vertices = [line.split()[-1] for line in header if line.startswith("element vertex ")]
    if len(vertices) != 1 or not vertices[0].isdigit():
        raise ValueError(f"Invalid Gaussian vertex count: {path}")
    return {"status": "available", "count": int(vertices[0]), "iteration": iteration,
            "path": str(path), "note": "Declared header count; not a full payload integrity check."}


def photo_inventory(output, summary):
    """Use explicit render manifests; never guess original names from numeric PNGs."""
    rendering = summary.get("rendering", {})
    method = rendering.get("method")
    if not re.fullmatch(r"ours_\d+", method or ""):
        iteration = summary.get("training", {}).get("final_iteration")
        method = f"ours_{iteration}" if isinstance(iteration, int) else None
    rows, manifests = [], {}
    if method is None:
        return rows, manifests
    for split in ("train", "test"):
        folder = output / split / method
        manifest_path = folder / "render_manifest.json"
        mapping = {}
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest["split"] != split or f"ours_{manifest['iteration']}" != method:
                raise ValueError(f"Render manifest does not match folder: {manifest_path}")
            entries = manifest["images"]
            for item in entries:
                filename = item["file"]
                if not re.fullmatch(r"\d+\.png", filename) or filename in mapping:
                    raise ValueError(f"Invalid or duplicate render filename: {filename}")
                mapping[filename] = item
            manifests[split] = len(entries)
        names = sorted(set(mapping) | {p.name for p in (folder / "renders").glob("*.png")})
        for name in names:
            item = mapping.get(name, {})
            rows.append({
                "Split": split, "Iterasi": method.removeprefix("ours_"),
                "Nama foto kamera": item.get("camera_image_name", ""),
                "File render": (folder / "renders" / name).relative_to(output).as_posix(),
                "File referensi": (folder / "gt" / name).relative_to(output).as_posix(),
                "Lebar piksel": item.get("width", ""),
                "Tinggi piksel": item.get("height", ""),
                "Pasangan tersedia": (folder / "renders" / name).is_file() and (folder / "gt" / name).is_file(),
                "Sumber nama": "render_manifest.json" if item else "Tidak tersedia",
            })
    return rows, manifests


def scene_data(output, summary, mem):
    training = summary.get("training", {})
    photos, manifest_counts = photo_inventory(output, summary)
    cameras = {}
    if mem.get("run_id"):
        path = output / "research" / "profiling" / f"cameras_{mem['run_id']}.json"
        if path.is_file():
            cameras = read_json(path)
    initial = training.get("initial_point_count")
    initial_source = "summary.training.initial_point_count" if initial is not None else None
    if initial is None:
        initial = cameras.get("initial_active_gaussians")
        if initial is not None:
            initial_source = "profiler.initial_active_gaussians"
    log = output / "research/logs/training.log"
    if initial is None and log.is_file():
        from research_pipeline.training_parser import read_initial_point_count
        initial = read_initial_point_count(log)
        if initial is not None:
            initial_source = "research/logs/training.log"
    train_count, test_count = training.get("training_camera_count"), training.get("test_camera_count")
    if train_count is None:
        train_count = len(cameras["cameras"]) if "cameras" in cameras else manifest_counts.get("train")
    if test_count is None:
        test_count = manifest_counts.get("test")
    total = train_count + test_count if train_count is not None and test_count is not None else training.get("camera_count")
    splats = ply_count(output, summary)
    return {
        "Jumlah foto terdaftar": total, "Foto training": train_count, "Foto test": test_count,
        "Gaussian awal": initial, "Splats PLY akhir": splats["count"],
        "Iterasi PLY": splats.get("iteration"), "Folder output": summary.get("output_path", str(output)),
    }, {"photos": photos, "ply": splats,
        "initial_gaussian_count_source": initial_source,
        "note": "Counts refer to registered camera images, not every raw file or original video frame. "
                "Camera names are recorded by the renderer and may omit the original extension."}


PHOTO_FIELDS = ["Split", "Iterasi", "Nama foto kamera", "File render", "File referensi",
                "Lebar piksel", "Tinggi piksel", "Pasangan tersedia", "Sumber nama"]


def update_report_table(path, rows, fields=FIELDS):
    """Atomic table for sequential pipeline runs, keyed by output folder."""
    from research_pipeline.utils import atomic_write
    previous = read_csv(path) if path.exists() else []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if next(csv.reader(stream), []) != fields:
                raise ValueError(f"Report CSV schema mismatch: {path}")
    indexed = {row["Folder output"]: row for row in previous}
    for row in rows:
        indexed[row["Folder output"]] = row
    def write(stream):
        stream.write("\ufeff")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(indexed.values())
    atomic_write(path, write)


class ReportGenerationError(RuntimeError):
    """Keep report locations available when a later export step fails."""
    def __init__(self, error, directory, table, summary_table):
        super().__init__(str(error))
        self.reporting = {"directory": str(directory), "table": str(table),
                          "summary_table": str(summary_table)}


def sync_report_status(result):
    """Correct any already-written report rows after a reporting failure."""
    from research_pipeline.utils import write_json
    report = result.get("reporting", {})
    directory = Path(report["directory"]) if report.get("directory") else None
    tables = []
    if directory is not None:
        tables.extend(((directory / "tabel_evaluasi.csv", FIELDS),
                       (directory / "ringkasan_hasil.csv", SUMMARY_FIELDS)))
    if report.get("table"):
        tables.append((Path(report["table"]), FIELDS))
    if report.get("summary_table"):
        tables.append((Path(report["summary_table"]), SUMMARY_FIELDS))
    for table, fields in tables:
        if table.is_file():
            changed = []
            for row in read_csv(table):
                if row.get("Folder output") == result["output_path"]:
                    row["Status"] = result["status"]
                    changed.append(row)
            if changed:
                update_report_table(table, changed, fields=fields)
    if directory is not None and (directory / "detail_laporan.json").is_file():
        path = directory / "detail_laporan.json"
        details = read_json(path)
        for item in details["experiments"]:
            if item["table_row"].get("Folder output") == result["output_path"]:
                item["table_row"]["Status"] = result["status"]
                item["report_failure"] = report.get("error")
        write_json(path, details)


def update_result_tables(directory, master_csv):
    """Refresh readable result tables; keep original measurements unchanged."""
    table = master_csv.with_name(master_csv.stem + "_laporan.csv")
    summary_table = master_csv.with_name(master_csv.stem + "_ringkasan.csv")
    update_report_table(table, read_csv(directory / "tabel_evaluasi.csv"))
    # Include earlier report rows when creating or refreshing the short table.
    rows = [{key: row.get(key) for key in SUMMARY_FIELDS} for row in read_csv(table)]
    update_report_table(summary_table, rows, fields=SUMMARY_FIELDS)
    print(f"Tabel evaluasi gabungan: {table}")
    print(f"Ringkasan seluruh run: {summary_table}")


def automatic_report(output, master_csv):
    destination = output / "research" / ("laporan_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    table = master_csv.with_name(master_csv.stem + "_laporan.csv")
    summary_table = master_csv.with_name(master_csv.stem + "_ringkasan.csv")
    try:
        export([output], destination)
        update_result_tables(destination, master_csv)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        raise ReportGenerationError(exc, destination, table, summary_table) from exc
    return {"status": "complete", "directory": str(destination), "table": str(table),
            "summary_table": str(summary_table),
            "resource_table": str(destination / "rincian_proses.csv")}


def export(outputs, destination, pairs=5):
    if pairs < 1:
        raise ValueError("--pairs must be at least 1")
    prepared = []
    for output in outputs:
        output = output.resolve()
        summary = read_json(output / "research" / "summary.json")
        stamp = datetime.fromisoformat(summary["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Summary timestamp has no timezone")
        rendering = summary.get("rendering", {})
        mem = memory_info(output)
        seconds = numeric(summary.get("training", {}).get("duration_seconds"))
        row = dict(zip(FIELDS, [
            stamp.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
            summary["scene_name"], "RaDe-GS", numeric(rendering.get("psnr")),
            numeric(rendering.get("ssim")), numeric(rendering.get("lpips")),
            mem["gpu_total_peak_gib"], seconds / 60 if seconds is not None else None,
            rendering.get("split"), rendering.get("evaluation_image_count"), summary.get("status"),
        ]))
        counts, data = scene_data(output, summary, mem)
        row.update(counts)
        geometry = summary.get("geometry", {})
        row.update({
            "Status GT 3D": geometry.get("status"),
            "Alasan GT dilewati": geometry.get("skip_reason"),
            "Chamfer (satuan rekonstruksi)": numeric(geometry.get("chamfer_distance")),
            "Precision geometri": numeric(geometry.get("precision")),
            "Recall geometri": numeric(geometry.get("recall")),
            "F1 geometri": numeric(geometry.get("f1_score")),
            "File GT 3D": geometry.get("ground_truth_path"),
        })
        prepared.append((output, summary, mem, row, data))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    details = []
    for index, (output, summary, mem, row, data) in enumerate(prepared, 1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", summary["scene_name"])
        review = comparison_images(output, summary, destination / f"{index:02d}_{safe_name}", pairs)
        photo_file = destination / f"{index:02d}_{safe_name}_daftar_foto.csv"
        with photo_file.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PHOTO_FIELDS)
            writer.writeheader()
            writer.writerows(data.pop("photos"))
        details.append({
            "source_summary": str(output / "research/summary.json"),
            "original_output_path": summary.get("output_path"),
            "experiment_id": summary.get("experiment_id"),
            "table_row": row, "memory": mem, "comparison": review,
            "geometry": summary.get("geometry", {}),
            "data": data, "photo_inventory": str(photo_file),
        })
        print(f"{summary['scene_name']}: comparison images = {review['status']}")
    write_table(destination / "tabel_evaluasi.csv", FIELDS,
                [row for _, _, _, row, _ in prepared])
    write_table(destination / "ringkasan_hasil.csv", SUMMARY_FIELDS,
                [row for _, _, _, row, _ in prepared])
    write_table(destination / "rincian_proses.csv", RESOURCE_FIELDS,
                [row for _, summary, mem, _, _ in prepared
                 for row in resource_rows(summary, mem)])
    (destination / "PETUNJUK_LAPORAN.md").write_text(REPORT_GUIDE, encoding="utf-8")
    with (destination / "detail_laporan.json").open("x", encoding="utf-8") as stream:
        json.dump({"notes": [
            "Timestamp is summary recording time, not training start.",
            "Training duration includes preparation and saving within the training subprocess.",
            "Train-split metrics are not held-out test performance.",
            "Missing values stay blank; GPU total is not substituted with allocator/process memory.",
            "One row per supplied output; this export does not modify the pipeline master CSV.",
        ], "experiments": details}, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Report: {destination}")
    print(f"Buka ringkasan utama: {destination / 'ringkasan_hasil.csv'}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", nargs="+", type=Path, required=True,
                        help="Output folders containing research/summary.json")
    parser.add_argument("--destination", type=Path,
                        help="New report directory; must not exist")
    parser.add_argument("--results-csv", type=Path,
                        help="Base results.csv path: refresh its _laporan.csv and _ringkasan.csv "
                             "reading views; the technical CSV and source summary are unchanged")
    parser.add_argument("--pairs", type=int, default=5, help="Comparison pairs per scene (default: 5)")
    args = parser.parse_args()
    destination = args.destination or (args.outputs[0] / "research" /
        ("laporan_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
    try:
        export(args.outputs, destination, args.pairs)
        if args.results_csv is not None:
            update_result_tables(destination.resolve(), args.results_csv)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Report failed: {exc}\n")


if __name__ == "__main__":
    main()
