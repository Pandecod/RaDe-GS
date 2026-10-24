"""Best-effort NVIDIA telemetry; unknown values stay unknown, never zero."""
import csv
import io
import math
import os
import shutil
import subprocess

DEVICE_FIELDS = (
    "index", "uuid", "utilization.gpu", "memory.used", "memory.total",
    "clocks.sm", "temperature.gpu", "power.draw",
)
DEVICE_KEYS = (
    "gpu_index", "gpu_uuid", "gpu_utilization_percent", "device_vram_used_mb",
    "device_vram_total_mb", "sm_clock_mhz", "temperature_c", "power_w",
)
# Preserve the supported query after probing unsupported optional fields once.
_device_fields = DEVICE_FIELDS


def query(arguments: list[str]) -> list[list[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None and os.path.isfile("/usr/lib/wsl/lib/nvidia-smi"):
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable")
    result = subprocess.run([executable, *arguments, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, check=True)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


def number(value):
    """WSL/drivers can report [N/A], N/A, [Not Supported], nan, or blanks."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def process_vram(pids: set[int]) -> float | None:
    rows = query(["--query-compute-apps=pid,used_gpu_memory"])
    values = []
    for row in rows:
        if len(row) != 2:
            continue
        try:
            pid = int(row[0])
        except ValueError:
            continue
        if pid in pids:
            value = number(row[1])
            # A partial process total would underreport memory.
            if value is None:
                return None
            values.append(value)
    return sum(values) if values else None


def device_telemetry() -> list[dict]:
    """Device-global metrics, explicitly identified by index AND UUID.

    Numeric CUDA_VISIBLE_DEVICES ordinals are not assumed to be nvidia-smi IDs.
    Report all devices; the manifest records the training CUDA ordinal.
    """
    global _device_fields
    fields = _device_fields
    try:
        rows = query(["--query-gpu=" + ",".join(fields)])
    except subprocess.CalledProcessError:
        if fields != DEVICE_FIELDS:
            raise
        # Probe optional fields once. A missing clock field must not discard
        # temperature/power measurements that the driver does support.
        supported = list(DEVICE_FIELDS[:5])
        for optional in DEVICE_FIELDS[5:]:
            try:
                query(["--query-gpu=" + ",".join([*DEVICE_FIELDS[:5], optional])])
            except subprocess.CalledProcessError:
                continue
            supported.append(optional)
        fields = tuple(supported)
        rows = query(["--query-gpu=" + ",".join(fields)])
        _device_fields = fields
    result = []
    for row in rows:
        if len(row) != len(fields):
            continue
        item = dict.fromkeys(DEVICE_KEYS)
        for field, value in zip(fields, row):
            key = DEVICE_KEYS[DEVICE_FIELDS.index(field)]
            item[key] = value if key in ("gpu_index", "gpu_uuid") else number(value)
        result.append(item)
    return result


def free_vram() -> float:
    rows = query(["--query-gpu=index,uuid,memory.free"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        identifiers = [s.strip() for s in visible.split(",") if s.strip()]
        if not identifiers or "-1" in identifiers:
            raise RuntimeError("No CUDA devices are visible")
        if all(s.startswith("GPU-") for s in identifiers):
            rows = [r for r in rows if len(r) > 1 and
                    any(r[1].startswith(s) for s in identifiers)]
        elif any(not s.isdigit() for s in identifiers):
            raise RuntimeError("Cannot safely resolve CUDA_VISIBLE_DEVICES (including MIG)")
    if not rows or any(len(row) < 3 for row in rows):
        raise RuntimeError("No matching GPU memory telemetry")
    values = [number(row[2]) for row in rows]
    if any(value is None for value in values):
        # Resource guard must not treat missing telemetry as available memory.
        raise RuntimeError("Free GPU memory telemetry is unavailable")
    return min(values)
