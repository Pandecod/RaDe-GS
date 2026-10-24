"""Opt-in training diagnostics. Importing this module never imports torch."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
import uuid
import warnings

PHASES = ("forward", "loss", "backward", "optimizer", "densification", "filter3d")
COLUMNS = [
    "run_id", "utc", "iteration", "sample_kind", "status", "camera", "width", "height",
    "sh_degree", "regularization", "gaussians_rendered_input", "gaussians_active",
    "densification_ran", "gaussians_before_densify", "gaussians_after_densify",
    "clone_created", "split_net_added", "split_parents_replaced", "split_children_created",
    "opacity_pruned", "count_balance_ok", "filter3d_ran", "opacity_reset",
    "evaluation", "save", "checkpoint", "iteration_gpu_elapsed_ms",
    "iteration_cpu_submit_ms", "raster_stats_gpu_ms", "profiler_cpu_enqueue_ms",
    "cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes", "raster_context_status", "num_rendered",
    "geometry_buffer_bytes", "binning_buffer_bytes", "image_buffer_bytes",
    "visible_gaussians", "radius_mean_visible_px", "radius_max_px",
    "tile_grid_count", "tile_instances_per_grid_tile", "instances_per_visible_gaussian",
] + [f"{name}_{kind}_ms" for name in PHASES for kind in ("gpu", "cpu_submit")]


def raster_context_metadata(image):
    """Read existing autograd metadata BEFORE backward; retain no graph/tensors."""
    try:
        ctx = image.grad_fn
        count = ctx.num_rendered
        saved = ctx.saved_tensors
        # Verified _RasterizeGaussians.save_for_backward layout in this repository.
        if len(saved) != 13 or not isinstance(count, int):
            return {"raster_context_status": "unsupported_layout"}
        buffers = saved[9:12]
        return {
            "raster_context_status": "available",
            "num_rendered": count,
            **{f"{name}_buffer_bytes": int(t.numel() * t.element_size())
               for name, t in zip(("geometry", "binning", "image"), buffers)},
        }
    except (AttributeError, RuntimeError, TypeError):
        return {"raster_context_status": "unavailable"}


def densification_counts(before, after, result, split_factor=2):
    """The existing function returns NET split growth, not generated children."""
    clone, split_net, pruned = (int(value) for value in result)
    parents = split_net // (split_factor - 1) if split_factor > 1 else None
    exact = parents is not None and parents * (split_factor - 1) == split_net
    return {
        "densification_ran": 1,
        "gaussians_before_densify": before, "gaussians_after_densify": after,
        "clone_created": clone, "split_net_added": split_net,
        "split_parents_replaced": parents if exact else None,
        "split_children_created": parents * split_factor if exact else None,
        "opacity_pruned": pruned,
        "count_balance_ok": int(after == before + clone + split_net - pruned),
    }


def profile_phase(profiler, name):
    return profiler.phase(name) if profiler is not None else nullcontext()


class TrainingProfiler:
    """Sample CUDA events without synchronizing the training loop.

    CPU wall columns measure submission (including existing host waits), not GPU
    completion. GPU elapsed includes host launch gaps on the recorded stream.
    Statistics run AFTER the iteration end event. Only close() may synchronize.
    """
    def __init__(self, output, options, metadata, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.options = options
        self.interval = options.profile_interval
        if self.interval < 1 or options.profile_from < 0 or options.profile_until < 0:
            raise ValueError("Profiling interval must be positive; bounds must be nonnegative")
        if options.profile_until and options.profile_until < options.profile_from:
            raise ValueError("profile_until must be >= profile_from")
        if not math.isfinite(options.profile_gpu_interval) or options.profile_gpu_interval < 0:
            raise ValueError("profile_gpu_interval must be finite and nonnegative")
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        self.directory = Path(output) / "research" / "profiling"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"training_{self.run_id}.csv"
        self.stream = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=COLUMNS)
        self.writer.writeheader()
        self.stream.flush()
        self.pending = deque()
        self.current = None
        self.enabled = True
        self.closed = False
        self.rows_written = 0
        self.messages = []
        self.stop = threading.Event()
        self.thread = None
        self.monitor = None
        self.split_factor = 2
        self.tile_shape = self._tile_shape()
        try:
            cuda = self.torch.cuda
            device = cuda.current_device()
            metadata = {
                **metadata, "run_id": self.run_id, "schema_version": 1,
                "pid": os.getpid(), "python_version": sys.version,
                "pytorch_version": str(self.torch.__version__),
                "cuda_version": self.torch.version.cuda,
                "cuda_device_ordinal": device, "gpu_name": cuda.get_device_name(device),
                "environment": {k: os.environ.get(k) for k in
                                ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
                                 "PYTORCH_CUDA_ALLOC_CONF")},
                "tile_shape": self.tile_shape, "git": self._git_metadata(),
                "source_sha256": self._source_hashes(),
                "timing": {
                    "cpu": "perf_counter submission wall time, not synchronized completion",
                    "gpu": "CUDA event elapsed on the training stream, including launch gaps",
                    "memory_peak": "process allocator high-water mark; never reset by profiler",
                    "raster_stats": "outside iteration end event; copied asynchronously to pinned CPU",
                    "shutdown": "normal completion/KeyboardInterrupt drains events; SIGKILL cannot flush pending rows",
                },
            }
            self._json(f"manifest_{self.run_id}.json", metadata)
            if options.profile_gpu_interval:
                try:
                    from research_pipeline.process_monitor import ProcessMonitor
                except ImportError as exc:
                    self.warn(f"External telemetry unavailable; allocator profiling remains active: {exc}")
                else:
                    self.monitor = ProcessMonitor(
                        self.directory / f"gpu_{self.run_id}.csv")
                    self.thread = threading.Thread(target=self._sample_gpu, daemon=True,
                                                   name="radegs-profile-monitor")
                    self.thread.start()
        except BaseException:
            self.stream.close()
            raise
        print(f"[PROFILE] CSV: {self.path}", flush=True)

    def _json(self, name, data):
        with (self.directory / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, default=str)

    @staticmethod
    def _git_metadata():
        root = Path(__file__).resolve().parents[1]
        result = {}
        for name, args in (
            ("commit", ["rev-parse", "HEAD"]),
            ("tracked_diff", ["diff", "--no-ext-diff", "HEAD", "--", "train.py",
                              "utils", "arguments", "gaussian_renderer", "scene",
                              "research_pipeline", "configs"]),
        ):
            try:
                run = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
                                     capture_output=True, text=True, timeout=3)
                result[name] = run.stdout.strip() if run.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                result[name] = None
        return result

    @staticmethod
    def _source_hashes():
        root = Path(__file__).resolve().parents[1]
        paths = ('train.py', 'utils/training_profiler.py',
                 'research_pipeline/gpu_monitor.py', 'research_pipeline/process_monitor.py',
                 'research_pipeline/runner.py', 'utils/camera_utils.py',
                 'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                 'arguments/__init__.py',
                 'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py')
        result = {}
        for name in paths:
            try:
                result[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            except OSError:
                result[name] = None
        return result

    @staticmethod
    def _tile_shape():
        import re
        path = Path(__file__).resolve().parents[1] / "submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h"
        try:
            source = path.read_text(encoding="utf-8")
            return tuple(int(re.search(r"(?:#define\s+" + key + r"\s+|constexpr\s+int\s+" + key + r"\s*=\s*)(\d+)", source)[1])
                         for key in ("BLOCK_X", "BLOCK_Y"))
        except (OSError, TypeError, ValueError):
            return None

    def _sample_gpu(self):
        while not self.stop.is_set():
            self.monitor.sample(os.getpid())
            self.stop.wait(self.options.profile_gpu_interval)

    def warn(self, message):
        if message not in self.messages:
            self.messages.append(message)
            warnings.warn("[PROFILE] " + message, RuntimeWarning, stacklevel=2)

    def describe_scene(self, cameras, gaussians):
        self.split_factor = inspect.signature(gaussians.densify_and_split).parameters["N"].default
        self._json(f"cameras_{self.run_id}.json", {
            "initial_active_gaussians": int(gaussians.get_xyz.shape[0]),
            "split_factor": self.split_factor,
            "cameras": [{"name": c.image_name, "width": c.image_width,
                         "height": c.image_height, "fx": c.Fx, "fy": c.Fy,
                         "cx": c.Cx, "cy": c.Cy, "device": str(c.original_image.device),
                         "dtype": str(c.original_image.dtype)} for c in cameras],
        })

    def begin_iteration(self, iteration, gaussians):
        self.drain()
        self.current = None
        control = (self.options.profile_control_samples and iteration > 1
                   and (iteration - 1) % self.interval == 0)
        selected = iteration % self.interval == 0 or control
        if (not self.enabled or not selected or iteration < self.options.profile_from
                or (self.options.profile_until and iteration > self.options.profile_until)):
            return
        if len(self.pending) >= 8:
            self.warn("Pending event queue full; skipping samples rather than stalling training")
            return
        row = dict.fromkeys(COLUMNS)
        row.update(run_id=self.run_id, utc=datetime.now(timezone.utc).isoformat(),
                   iteration=iteration, sample_kind="control" if control else "scheduled",
                   gaussians_rendered_input=int(gaussians.get_xyz.shape[0]),
                   densification_ran=0, filter3d_ran=0, opacity_reset=0)
        for phase in PHASES:
            row[f"{phase}_gpu_ms"] = 0.0
            row[f"{phase}_cpu_submit_ms"] = 0.0
        start = self.torch.cuda.Event(enable_timing=True)
        wall = perf_counter()
        start.record()
        self.current = {"row": row, "start": start, "wall": wall, "phases": {},
                        "radii": None}

    @contextmanager
    def phase(self, name):
        current = self.current
        if current is None:
            yield
            return
        if name not in PHASES:
            raise ValueError(f"Unknown profiling phase {name}")
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        wall = perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            current["phases"][name] = (start, end)
            current["row"][f"{name}_cpu_submit_ms"] = (perf_counter() - wall) * 1000
            if name == "filter3d":
                current["row"]["filter3d_ran"] = 1

    def capture_render(self, package, camera, sh_degree, regularization):
        if self.current is None:
            return
        row = self.current["row"]
        row.update(camera=camera.image_name, width=camera.image_width,
                   height=camera.image_height, sh_degree=sh_degree,
                   regularization=int(regularization))
        if self.options.profile_no_raster:
            row["raster_context_status"] = "disabled"
            return
        row.update(raster_context_metadata(package["render"]))
        self.current["radii"] = package["radii"].detach()
        if self.tile_shape:
            tx, ty = self.tile_shape
            row["tile_grid_count"] = ((camera.image_width + tx - 1) // tx
                                      * ((camera.image_height + ty - 1) // ty))

    def densification(self, before, after, result):
        if self.current is not None:
            self.current["row"].update(
                densification_counts(before, after, result, self.split_factor))

    def finish_iteration(self, gaussians, *, opacity_reset, evaluation, save, checkpoint):
        current = self.current
        if current is None:
            return
        overhead_start = perf_counter()
        row = current["row"]
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        current["end"] = end
        row.update(iteration_cpu_submit_ms=(perf_counter() - current["wall"]) * 1000,
                   gaussians_active=int(gaussians.get_xyz.shape[0]),
                   opacity_reset=int(opacity_reset), evaluation=int(evaluation),
                   save=int(save), checkpoint=int(checkpoint),
                   cuda_allocated_bytes=self.torch.cuda.memory_allocated(),
                   cuda_reserved_bytes=self.torch.cuda.memory_reserved(),
                   cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        radii = current.pop("radii")
        current["radius_cpu"] = None
        if radii is not None:
            stats_start = self.torch.cuda.Event(enable_timing=True)
            stats_end = self.torch.cuda.Event(enable_timing=True)
            stats_start.record()
            with self.torch.no_grad():
                if radii.numel():
                    values = self.torch.stack([
                        self.torch.count_nonzero(radii).to(self.torch.float64),
                        radii.sum(dtype=self.torch.float64),
                        radii.amax().to(self.torch.float64)])
                else:
                    values = self.torch.zeros(3, dtype=self.torch.float64, device=radii.device)
                host = self.torch.empty(3, dtype=self.torch.float64, device="cpu",
                                        pin_memory=True)
                host.copy_(values, non_blocking=True)
            stats_end.record()
            current["radius_cpu"] = host
            current["phases"]["raster_stats"] = (stats_start, stats_end)
        done = self.torch.cuda.Event()
        done.record()
        current["done"] = done
        row["profiler_cpu_enqueue_ms"] = (perf_counter() - overhead_start) * 1000
        self.pending.append(current)
        self.current = None
        self.drain()

    def drain(self, *, final=False):
        if final and self.pending:
            # One wait on shutdown only; phase times never wait independently.
            self.pending[-1]["done"].synchronize()
        while self.pending and self.pending[0]["done"].query():
            sample = self.pending.popleft()
            row = sample["row"]
            row["iteration_gpu_elapsed_ms"] = sample["start"].elapsed_time(sample["end"])
            for phase, (start, end) in sample["phases"].items():
                key = "raster_stats_gpu_ms" if phase == "raster_stats" else f"{phase}_gpu_ms"
                row[key] = start.elapsed_time(end)
            if sample["radius_cpu"] is not None:
                visible, radius_sum, radius_max = sample["radius_cpu"].tolist()
                row.update(visible_gaussians=int(visible), radius_max_px=radius_max,
                           radius_mean_visible_px=radius_sum / visible if visible else 0.0)
                if row["num_rendered"] is not None and visible:
                    row["instances_per_visible_gaussian"] = row["num_rendered"] / visible
            if row["num_rendered"] is not None and row["tile_grid_count"]:
                row["tile_instances_per_grid_tile"] = row["num_rendered"] / row["tile_grid_count"]
            row["status"] = "complete"
            self.writer.writerow(row)
            self.stream.flush()
            self.rows_written += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.thread is not None:
            # NVIDIA queries are bounded; never close its CSV while it is writing.
            self.thread.join()
        if self.monitor is not None:
            self.monitor.close()
        if self.current is not None:
            self.current["row"]["status"] = "interrupted"
            self.writer.writerow(self.current["row"])
            self.current = None
        try:
            self.drain(final=True)
        except Exception as exc:
            self.warn(f"Could not drain final CUDA samples: {exc}")
            # Preserve CPU fields; never read an unfinished pinned-memory copy.
            for sample in self.pending:
                sample["row"]["status"] = "incomplete_cuda"
                self.writer.writerow(sample["row"])
            self.pending.clear()
        finally:
            self.stream.flush()
            self.stream.close()
            self._json(f"status_{self.run_id}.json", {
                "complete_rows": self.rows_written, "warnings": self.messages,
                "monitor_warnings": self.monitor.warnings if self.monitor else [],
            })

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as close_error:
            if exc_type is None:
                raise
            self.warn(f"Profiler shutdown error while handling {exc_type.__name__}: {close_error}")
        return False
