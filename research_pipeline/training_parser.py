"""Tolerant parsing of raw tqdm and RaDe-GS training logs."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COLUMNS = ["scene_name", "object_name", "iteration", "loss", "loss_normal", "loss_ncc",
           "l1", "psnr", "it_per_sec", "test_l1", "test_psnr"]
META = {
    "camera_count": r"Reading camera\s+\d+/(\d+)",
    "training_camera_count": r"Loading Training Cameras:\s*(\d+)",
    "test_camera_count": r"Loading Test Cameras:\s*(\d+)",
    "camera_extent": r"cameras extent:\s*(" + NUMBER + ")",
    "initial_point_count": r"Number of points at initiali[sz]ation:\s*(\d+)",
}

def parse_training(log: Path, csv_path: Path, scene_name: str) -> dict:
    metadata = dict.fromkeys(META)
    rows: dict[int, dict] = {}
    iteration = None
    evaluation = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
            for key, pattern in META.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match[1]) if key == "camera_extent" else int(match[1])
                    metadata[key] = value
            event = re.search(r"\[ITER\s+(\d+)\]", line)
            progress = re.search(r"(\d+)\s*/\s*\d+", line) if "Reading camera" not in line else None
            if event:
                iteration = int(event[1])
                match = re.search(r"Evaluating\s+(train|test)\s*:", line)
                evaluation = match[1] if match else None
            elif progress and ("it/s" in line or "s/it" in line or "Loss" in line or "Training progress" in line or re.fullmatch(r"\s*\d+/\d+\s*", line)):
                iteration = int(progress[1])
                evaluation = None
            if iteration is None:
                continue
            values = {}
            for key, label in (("loss", "Loss"), ("loss_normal", "loss_normal"), ("loss_ncc", "loss_ncc")):
                match = re.search(r"\b" + label + r"\s*=\s*(" + NUMBER + ")", line)
                if match:
                    values[key] = float(match[1])
            if evaluation:
                for key, label in (("l1", "L1"), ("psnr", "PSNR")):
                    match = re.search(r"\b" + label + r"\s*[=:]?\s*(" + NUMBER + ")", line)
                    if match:
                        values[("test_" if evaluation == "test" else "") + key] = float(match[1])
            speed = re.search("(" + NUMBER + r")\s*it/s", line)
            inverse = re.search("(" + NUMBER + r")\s*s/it", line)
            if speed:
                values["it_per_sec"] = float(speed[1])
            elif inverse and float(inverse[1]) > 0:
                values["it_per_sec"] = 1 / float(inverse[1])
            row = rows.setdefault(iteration, {**dict.fromkeys(COLUMNS), "scene_name": scene_name,
                                              "object_name": scene_name, "iteration": iteration})
            row.update({k: v for k, v in values.items() if math.isfinite(v)})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows[i] for i in sorted(rows))
    final = rows[max(rows)] if rows else {}
    metadata["final_iteration"] = max(rows) if rows else None
    for key in ("loss", "loss_normal", "loss_ncc", "l1", "psnr", "it_per_sec"):
        metadata["final_" + key] = final.get(key)
    return metadata
