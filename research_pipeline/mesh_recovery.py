"""Read validated mesh retries and combine their measurements with frozen source runs."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path

from research_pipeline.utils import read_json

RECOVERED_STATUS = "success_mesh_recovered"
RECOVERY_FIELDS = [
    "Folder", "Status", "Status run asal", "Tahap gagal asal", "Folder training asal",
    "Folder output", "File mesh akhir", "Sumber result.json", "Waktu selesai retry (UTC+08)",
    "Script mesh", "Resolusi mesh", "Jumlah cluster", "Depth ratio",
    "Ukuran voxel", "Batas depth", "Block resolution", "Durasi mesh (menit)",
    "RAM puncak mesh (MiB)", "RAM rata-rata mesh (MiB)",
    "VRAM puncak proses mesh (MiB)", "VRAM rata-rata proses mesh (MiB)",
    "Peak VRAM seluruh GPU saat mesh (GiB)", "Vertex mesh", "Segitiga mesh",
    "Ukuran mesh (MiB)", "Blok voxel aktif", "Buffer struktur mesh (GiB)",
]


def positive_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing or invalid {label}")
    return value


def mesh_header(path):
    """Check the recorded mesh counts without loading a multi-million-vertex payload."""
    header = []
    with path.open("rb") as stream:
        for _ in range(1024):
            line = stream.readline(4096)
            if not line or len(line) >= 4096:
                raise ValueError(f"Incomplete PLY header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        else:
            raise ValueError(f"PLY header too long: {path}")
    if header[0] != "ply":
        raise ValueError(f"Invalid PLY: {path}")
    counts = {}
    for name in ("vertex", "face"):
        matches = [line.split()[-1] for line in header if line.startswith(f"element {name} ")]
        if len(matches) != 1 or not matches[0].isdigit() or int(matches[0]) <= 0:
            raise ValueError(f"Invalid mesh {name} count: {path}")
        counts[name] = int(matches[0])
    return counts


def prepare_mesh_recovery(path):
    """Return the original asset folder plus a new, in-memory recovered record."""
    path = Path(path).resolve()
    result_path = path / "result.json" if path.is_dir() else path
    attempt = result_path.parent
    raw = result_path.read_bytes()
    retry = read_json(result_path)
    if (retry.get("kind") != "mesh_retry" or retry.get("status") != "success"
            or type(retry.get("exit_code")) is not int or retry["exit_code"] != 0
            or retry.get("training_reused") is not True):
        raise ValueError(f"Only completed successful mesh retries can be imported: {result_path}")
    if Path(retry["attempt_path"]).resolve() != attempt:
        raise ValueError("Retry path differs from the recorded attempt_path; use the server's original folder")

    model = Path(retry["model_path"]).resolve()
    source_path = model / "research/summary.json"
    if Path(retry.get("source_summary_path", "")).resolve() != source_path:
        raise ValueError("Retry source_summary_path does not match its model folder")
    source = read_json(source_path)
    frozen = retry.get("source_summary_snapshot")
    if not isinstance(frozen, dict) or not frozen.get("experiment_id"):
        raise ValueError("Retry has no identified frozen source summary; cannot infer its training data")
    for key in ("experiment_id", "scene_name", "output_path", "training", "rendering"):
        if source.get(key) != frozen.get(key):
            raise ValueError(f"Source summary changed since retry ({key}); refusing mixed-run measurements")
    if Path(frozen["output_path"]).resolve() != model:
        raise ValueError("Frozen source output_path does not match the trained model")
    if frozen.get("status") not in ("success", "failed"):
        raise ValueError("Source pipeline run has no final status")
    if frozen.get("status") == "failed" and frozen.get("failed_stage") not in (
            "mesh_extraction", "mesh_extraction_oom", "mesh_resource_check", "mesh_validation"):
        raise ValueError("Mesh retry cannot repair a source failure outside the mesh stage")

    settings = retry["settings"]
    iteration = settings.get("iteration")
    if type(iteration) is not int or iteration != frozen.get("training", {}).get("final_iteration"):
        raise ValueError("Retry iteration does not match the original completed training iteration")
    if not (model / f"point_cloud/iteration_{iteration}/point_cloud.ply").is_file():
        raise ValueError("Original trained Gaussian PLY is missing")
    rendering = frozen.get("rendering", {})
    if rendering.get("split") not in ("train", "test"):
        raise ValueError("Source summary has no explicit evaluated image split")
    positive_number(rendering.get("evaluation_image_count"), "source evaluation image count")
    for key in ("psnr", "ssim", "lpips"):
        value = rendering.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Source {key} is unavailable; no metrics will be invented")
    stamp = datetime.fromisoformat(retry["finished_utc"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Retry completion timestamp has no timezone")

    mesh = retry.get("mesh_metrics", {})
    final_mesh = attempt / "recon_post.ply"
    if Path(mesh.get("mesh_path", "")).resolve() != final_mesh:
        raise ValueError("Recorded mesh_path does not point to this retry's recon_post.ply")
    counts = mesh_header(final_mesh)
    for key, element in (("vertex_count", "vertex"), ("face_count", "face")):
        if type(mesh.get(key)) is not int or mesh[key] != counts[element]:
            raise ValueError(f"Saved mesh header disagrees with recorded {key}")
    size_mb = positive_number(mesh.get("file_size_mb"), "mesh file size")
    if round(size_mb * 1024**2) != final_mesh.stat().st_size:
        raise ValueError("Saved mesh size differs from the validated retry recording")
    duration = positive_number(retry.get("duration_seconds"), "retry duration")
    process = retry.get("process_metrics") or {}
    if not isinstance(process, dict):
        raise ValueError("Invalid retry process metrics")

    provenance = {
        "kind": "mesh_recovery", "source_status": frozen.get("status"),
        "source_failed_stage": frozen.get("failed_stage"),
        "source_error_message": frozen.get("error_message"),
        "source_summary": str(source_path), "source_experiment_id": frozen["experiment_id"],
        "model_path": str(model), "retry_result": str(result_path),
        "retry_result_sha256": hashlib.sha256(raw).hexdigest(),
        "mesh_path": str(final_mesh), "finished_utc": retry["finished_utc"],
        "settings": copy.deepcopy(settings),
        "native_runtime": copy.deepcopy(retry.get("native_runtime", {})),
        "script_sha256": retry.get("script_sha256"),
        "native_runtime_sha256": retry.get("native_runtime_sha256"),
        "note": "Training/render measurements and images are reused from the frozen source. "
                "Only mesh measurements come from the retry. GT for the new mesh is not evaluated. "
                "Mesh validation checks header counts and saved file size, not a full payload hash.",
    }
    merged = copy.deepcopy(frozen)
    merged.update(
        kind="mesh_recovery", status=RECOVERED_STATUS, failed_stage=None, error_message=None,
        output_path=str(attempt), timestamp=retry["finished_utc"], recovery=provenance,
        experiment_id=hashlib.sha256(
            (frozen["experiment_id"] + "\nmesh_retry\n" + str(attempt)).encode()).hexdigest(),
    )
    merged["mesh_extraction"] = {
        **copy.deepcopy(settings), **copy.deepcopy(process), "duration_seconds": duration,
        **{key: mesh[key] for key in ("vertex_count", "face_count", "file_size_mb", "mesh_path")},
    }
    # GT values for a different mesh must never be inherited as new measurements.
    geometry = copy.deepcopy(frozen.get("geometry", {}))
    if geometry.get("status") != "disabled":
        geometry.update(status="not_run", skip_reason="mesh_retry_not_evaluated")
    for key in ("chamfer_distance", "precision", "recall", "f1_score", "duration_seconds"):
        geometry[key] = None
    merged["geometry"] = geometry
    merged["commands"] = {**merged.get("commands", {}), "mesh_extraction": retry.get("command", [])}
    merged["warnings"] = [*merged.get("warnings", []), *retry.get("warnings", []),
                          "Recovered mesh report; original failed run and technical CSV are unchanged."]
    merged.pop("reporting", None)
    return model, merged


def recovery_row(summary):
    """Readable mesh-specific measurements with explicit provenance and units."""
    recovery = summary["recovery"]
    mesh = summary["mesh_extraction"]
    runtime = recovery["native_runtime"]
    peak = mesh.get("device_global", {}).get("device_vram_used_mb", {}).get("peak")
    stamp = datetime.fromisoformat(recovery["finished_utc"].replace("Z", "+00:00"))
    return dict(zip(RECOVERY_FIELDS, [
        summary["scene_name"], summary["status"], recovery["source_status"],
        recovery["source_failed_stage"], recovery["model_path"], summary["output_path"],
        recovery["mesh_path"], recovery["retry_result"],
        stamp.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
        mesh.get("script"), mesh.get("resolution"), mesh.get("num_cluster"),
        mesh.get("depth_ratio"), mesh.get("voxel_size"), mesh.get("max_depth"),
        mesh.get("block_resolution"), mesh["duration_seconds"] / 60,
        mesh.get("peak_ram_mb"), mesh.get("average_ram_mb"), mesh.get("peak_vram_mb"),
        mesh.get("average_vram_mb"), peak / 1024 if peak is not None else None,
        mesh["vertex_count"], mesh["face_count"], mesh["file_size_mb"],
        runtime.get("active_voxel_blocks"),
        runtime["mesh_structure_bytes"] / 1024**3 if runtime.get("mesh_structure_bytes") is not None else None,
    ]))
