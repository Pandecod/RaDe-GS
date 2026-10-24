"""RSS and GPU telemetry with retrying, nonfatal, incremental CSV output."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
from time import perf_counter
import psutil
from research_pipeline import gpu_monitor

COLUMNS = [
    "utc", "elapsed_seconds", "root_pid", "ram_mb", "process_vram_mb",
    *gpu_monitor.DEVICE_KEYS, "process_vram_status", "device_status", "sample_cpu_ms",
]


class ProcessMonitor:
    def __init__(self, csv_path: Path | None = None, *, sample_gpu: bool = True) -> None:
        self.ram: list[float] = []
        self.vram: list[float] = []
        self.warnings: list[str] = []
        self.sample_gpu = sample_gpu
        self.csv_path = csv_path
        self.stream = None
        self.writer = None
        self.started = perf_counter()
        self.device_samples = {key: [] for key in gpu_monitor.DEVICE_KEYS[2:]}
        self.csv_failed = False

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _write(self, row):
        if self.csv_path is None or self.csv_failed:
            return
        try:
            if self.stream is None:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                self.stream = self.csv_path.open("x", newline="", encoding="utf-8")
                self.writer = csv.DictWriter(self.stream, fieldnames=COLUMNS)
                self.writer.writeheader()
            self.writer.writerow(row)
            self.stream.flush()
        except OSError as exc:
            self.csv_failed = True
            self.warn(f"Resource CSV unavailable: {exc}")
            self.close()

    def sample(self, pid: int) -> None:
        start = perf_counter()
        row = dict.fromkeys(COLUMNS)
        row.update(utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=start - self.started, root_pid=pid)
        pids = {pid}
        try:
            root = psutil.Process(pid)
            rss = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                    pids.add(process.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            row["ram_mb"] = rss / 1024**2
            self.ram.append(row["ram_mb"])
        except psutil.NoSuchProcess:
            return
        except (psutil.Error, OSError) as exc:
            self.warn(f"RAM monitoring unavailable: {exc}")
        devices = []
        if self.sample_gpu:
            try:
                value = gpu_monitor.process_vram(pids)
                if value is not None and math.isfinite(value) and value >= 0:
                    self.vram.append(value)
                    row["process_vram_mb"] = value
                    row["process_vram_status"] = "available"
                else:
                    row["process_vram_status"] = "unavailable"
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                row["process_vram_status"] = "query_error"
                self.warn(f"Process VRAM query failed; later samples will retry: {type(exc).__name__}")
            # Device sampling is independent of per-process telemetry.
            try:
                devices = gpu_monitor.device_telemetry()
                row["device_status"] = "available" if devices else "unavailable"
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                row["device_status"] = "query_error"
                self.warn(f"Device GPU query failed; later samples will retry: {type(exc).__name__}")
        else:
            row["process_vram_status"] = row["device_status"] = "handled_by_training_profiler"
        row["sample_cpu_ms"] = (perf_counter() - start) * 1000
        for device in devices:
            for key in self.device_samples:
                value = device.get(key)
                if value is not None:
                    self.device_samples[key].append(value)
            self._write({**row, **device})
        if not devices:
            self._write(row)

    def metrics(self) -> dict:
        if not self.vram and self.sample_gpu:
            self.warn("No process-attributed VRAM samples were reported")
        result = {}
        for name, samples in (("ram", self.ram), ("vram", self.vram)):
            result[f"peak_{name}_mb"] = max(samples) if samples else None
            result[f"average_{name}_mb"] = sum(samples) / len(samples) if samples else None
        # Device values are global and aggregate all reported devices, not process
        # utilization. Detailed CSV retains each device UUID for comparison.
        result["device_global"] = {
            key: {"peak": max(values) if values else None,
                  "average": sum(values) / len(values) if values else None}
            for key, values in self.device_samples.items()
        }
        if self.csv_path is not None:
            result["resource_csv"] = str(self.csv_path)
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            self.writer = None
