"""Conservative RAM and VRAM gate before mesh extraction."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Callable

import psutil

from research_pipeline.gpu_monitor import free_vram


def available_resources() -> tuple[float, float]:
    """Return available RAM in GiB and minimum visible free VRAM in MiB."""
    ram_gb = psutil.virtual_memory().available / 1024**3
    vram_mb = free_vram()
    return ram_gb, vram_mb


def wait_until_ready(
    config: dict,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    probe: Callable[[], tuple[float, float]] = available_resources,
) -> tuple[bool, str | None]:
    guard = config["resource_guard"]
    if not guard["enabled"]:
        return True, None

    logging.info("[RESOURCE] Preparing mesh extraction")
    cooldown = float(guard["cooldown_before_mesh_seconds"])
    if cooldown:
        sleep(cooldown)
    started = clock()

    while True:
        try:
            ram_gb, vram_mb = probe()
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            return False, f"Unable to verify free RAM/VRAM safely: {exc}"

        logging.info("[RESOURCE] Free RAM  : %.2f GB", ram_gb)
        logging.info("[RESOURCE] Free VRAM : %.0f MB", vram_mb)
        if (ram_gb >= guard["minimum_free_ram_gb"]
                and vram_mb >= guard["minimum_free_vram_mb"]):
            return True, None

        if not guard["wait_if_memory_low"]:
            return False, "Resources remain below the configured mesh thresholds"
        elapsed = clock() - started
        if elapsed >= guard["max_wait_seconds"]:
            return False, "Resources remained below threshold until max_wait_seconds"
        delay = min(float(guard["check_interval_seconds"]),
                    float(guard["max_wait_seconds"]) - elapsed)
        logging.info("[RESOURCE] Resource below threshold. Waiting %.1f seconds...", delay)
        sleep(delay)

