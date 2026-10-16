"""Best effort RSS of the process tree and process-attributed VRAM samples."""
from __future__ import annotations
import math
import subprocess
import psutil
from research_pipeline import gpu_monitor

class ProcessMonitor:
    def __init__(self) -> None:
        self.ram: list[float] = []
        self.vram: list[float] = []
        self.warnings: list[str] = []
        self.gpu_failed = False

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def sample(self, pid: int) -> None:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
            rss = 0
            pids = set()
            for process in processes:
                try:
                    rss += process.memory_info().rss
                    pids.add(process.pid)
                except psutil.NoSuchProcess:
                    pass
            self.ram.append(rss / 1024**2)
        except psutil.NoSuchProcess:
            return
        except (psutil.Error, OSError) as exc:
            self.warn(f"RAM monitoring unavailable: {exc}")
            pids = {pid}
        if not self.gpu_failed:
            try:
                value = gpu_monitor.process_vram(pids)
                # CUDA may not have created a context during the first samples.
                # Keep trying instead of permanently disabling GPU telemetry.
                if value is None:
                    return
                if not math.isfinite(value):
                    raise ValueError("Non-finite process VRAM value")
                self.vram.append(value)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                self.gpu_failed = True
                self.vram.clear()
                self.warn(f"VRAM monitoring unavailable: {exc}")

    def metrics(self) -> dict:
        if not self.vram and not self.gpu_failed:
            self.warn("No process-attributed VRAM samples were reported")
        result = {}
        for name, samples in (("ram", self.ram), ("vram", self.vram)):
            result[f"peak_{name}_mb"] = max(samples) if samples else None
            result[f"average_{name}_mb"] = sum(samples) / len(samples) if samples else None
        return result
