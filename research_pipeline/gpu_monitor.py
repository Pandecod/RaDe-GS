"""NVIDIA telemetry via bounded nvidia-smi calls; no torch imports."""
import csv
import io
import os
import shutil
import subprocess

def query(arguments: list[str]) -> list[list[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None and os.path.isfile("/usr/lib/wsl/lib/nvidia-smi"):
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable")
    result = subprocess.run([executable, *arguments, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5, check=True)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

def process_vram(pids: set[int]) -> float | None:
    rows = query(["--query-compute-apps=pid,used_gpu_memory"])
    values = []
    for row in rows:
        if len(row) != 2:
            raise ValueError("Unsupported GPU process telemetry")
        if int(row[0]) in pids:
            values.append(float(row[1]))
    # WSL may omit process telemetry entirely. Absence is not measured zero.
    return sum(values) if values else None

def free_vram() -> float:
    rows = query(["--query-gpu=index,uuid,memory.free"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        identifiers = [s.strip() for s in visible.split(",") if s.strip()]
        if not identifiers or "-1" in identifiers:
            raise RuntimeError("No CUDA devices are visible")
        if all(s.startswith("GPU-") for s in identifiers):
            rows = [r for r in rows if any(r[1].startswith(s) for s in identifiers)]
        elif any(not s.isdigit() for s in identifiers):
            raise RuntimeError("Cannot safely resolve CUDA_VISIBLE_DEVICES (including MIG)")
        # Numeric CUDA ordinals may differ from nvidia-smi indices. Conservatively
        # require every reported device to pass rather than select a wrong GPU.
    if not rows:
        raise RuntimeError("No matching GPU memory telemetry")
    if any(len(row) < 3 for row in rows):
        raise ValueError("Unsupported free GPU memory telemetry")
    return min(float(row[2]) for row in rows)
