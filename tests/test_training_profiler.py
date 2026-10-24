"""CPU-only diagnostics tests: never import torch or launch model training."""
import ast
from contextlib import nullcontext
import copy
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

from research_pipeline import gpu_monitor
from research_pipeline.process_monitor import ProcessMonitor
from utils.training_profiler import (
    TrainingProfiler, densification_counts, raster_context_metadata,
)


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def options(**changes):
    return SimpleNamespace(**{
        "profile_interval": 100, "profile_from": 1, "profile_until": 0,
        "profile_control_samples": False, "profile_no_raster": True,
        "profile_gpu_interval": 0, **changes,
    })


class FakeEvent:
    events = []
    ready = True
    synchronize_count = 0
    record_count = 0

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.stamp = None
        self.__class__.events.append(self)

    def record(self):
        self.__class__.record_count += 1
        self.stamp = self.__class__.record_count

    def query(self):
        return self.__class__.ready

    def synchronize(self):
        self.__class__.synchronize_count += 1
        self.__class__.ready = True

    def elapsed_time(self, other):
        if not self.enable_timing or not other.enable_timing:
            raise AssertionError("Timing disabled")
        return float(other.stamp - self.stamp)


class FakeCuda:
    Event = FakeEvent
    @staticmethod
    def current_device(): return 0
    @staticmethod
    def get_device_name(device): return "CPU test stub, no GPU"
    @staticmethod
    def memory_allocated(): return 100
    @staticmethod
    def memory_reserved(): return 200
    @staticmethod
    def max_memory_allocated(): return 300
    @staticmethod
    def max_memory_reserved(): return 400


FAKE_TORCH = SimpleNamespace(
    cuda=FakeCuda, __version__="stub", version=SimpleNamespace(cuda=None))
GAUSSIANS = SimpleNamespace(get_xyz=SimpleNamespace(shape=(10, 3)))


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        FakeEvent.events = []
        FakeEvent.ready = True
        FakeEvent.synchronize_count = 0
        FakeEvent.record_count = 0

    def profiler(self, directory, **settings):
        with patch.object(TrainingProfiler, "_git_metadata", return_value={}):
            return TrainingProfiler(directory, options(**settings), {},
                                    torch_module=FAKE_TORCH)

    def finish(self, profiler):
        profiler.finish_iteration(GAUSSIANS, opacity_reset=False,
                                  evaluation=False, save=False, checkpoint=False)

    def test_only_selected_iterations_create_events_and_bounds_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.profiler(tmp, profile_from=100, profile_until=201,
                               profile_control_samples=True) as profiler:
                for iteration in (1, 99, 100, 101, 102, 200, 201, 202):
                    profiler.begin_iteration(iteration, GAUSSIANS)
                    with profiler.phase("forward"):
                        pass
                    self.finish(profiler)
                with profiler.path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(r["iteration"]) for r in rows], [100, 101, 200, 201])
                self.assertEqual(rows[1]["sample_kind"], "control")
                self.assertEqual(FakeEvent.synchronize_count, 0)

    def test_event_read_is_deferred_and_shutdown_waits_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self.profiler(tmp)
            FakeEvent.ready = False
            for iteration in (100, 200):
                profiler.begin_iteration(iteration, GAUSSIANS)
                with profiler.phase("forward"): pass
                with profiler.phase("loss"): pass
                self.finish(profiler)
            self.assertEqual(len(profiler.pending), 2)
            self.assertEqual(profiler.rows_written, 0)
            self.assertEqual(FakeEvent.synchronize_count, 0)
            profiler.close()
            self.assertEqual(FakeEvent.synchronize_count, 1)
            rows = read_rows(profiler.path)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["status"] == "complete" for r in rows))
            self.assertEqual(rows[0]["cuda_peak_allocated_bytes"], "300")
            self.assertGreater(float(rows[0]["forward_gpu_ms"]), 0)

    def test_incremental_flush_unique_files_and_interrupt_preserve_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.profiler(tmp)
            first.begin_iteration(100, GAUSSIANS)
            self.finish(first)
            # Read while writer remains open: this row must already be durable
            # to an ordinary process stop (not a system power failure).
            rows = read_rows(first.path)
            self.assertEqual(len(rows), 1)
            first.begin_iteration(200, GAUSSIANS)
            first.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
            rows = read_rows(first.path)
            self.assertEqual([r["status"] for r in rows], ["complete", "interrupted"])
            with self.profiler(tmp) as second:
                self.assertNotEqual(first.path, second.path)
            self.assertEqual(len(read_rows(first.path)), 2)

    def test_queue_is_bounded_without_forcing_gpu_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self.profiler(tmp)
            FakeEvent.ready = False
            for iteration in range(100, 901, 100):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    profiler.begin_iteration(iteration, GAUSSIANS)
                self.finish(profiler)
            self.assertEqual(len(profiler.pending), 8)
            self.assertIsNone(profiler.current)
            self.assertEqual(FakeEvent.synchronize_count, 0)
            profiler.close()

    def test_failed_cuda_shutdown_never_reads_unfinished_host_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self.profiler(tmp)
            FakeEvent.ready = False
            profiler.begin_iteration(100, GAUSSIANS)
            self.finish(profiler)
            with patch.object(FakeEvent, "synchronize", side_effect=RuntimeError("CUDA failure")), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                profiler.close()
            rows = read_rows(profiler.path)
            self.assertEqual(rows[0]["status"], "incomplete_cuda")
            self.assertEqual(rows[0]["iteration_gpu_elapsed_ms"], "")

    def test_split_counts_are_children_not_net_growth(self):
        result = densification_counts(100, 115, (10, 8, 3))
        self.assertEqual(result["split_parents_replaced"], 8)
        self.assertEqual(result["split_children_created"], 16)
        self.assertEqual(result["count_balance_ok"], 1)
        self.assertEqual(densification_counts(100, 114, (10, 8, 3))["count_balance_ok"], 0)

    def test_existing_raster_context_metadata_and_missing_context(self):
        buffers = [SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)] * 3
        saved = [None] * 9 + buffers + [None]
        image = SimpleNamespace(grad_fn=SimpleNamespace(num_rendered=321, saved_tensors=saved))
        metadata = raster_context_metadata(image)
        self.assertEqual(metadata["num_rendered"], 321)
        self.assertEqual(metadata["binning_buffer_bytes"], 40)
        self.assertEqual(raster_context_metadata(SimpleNamespace(grad_fn=None))["raster_context_status"],
                         "unavailable")
        self.assertEqual(TrainingProfiler._tile_shape(), (16, 16))

    def test_invalid_profile_settings_fail_before_creating_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for settings in ({"profile_interval": 0}, {"profile_from": -1},
                             {"profile_until": 1, "profile_from": 2},
                             {"profile_gpu_interval": float("nan")}):
                with self.assertRaises(ValueError):
                    self.profiler(tmp, **settings)
            self.assertEqual(list(Path(tmp).iterdir()), [])


    def test_radius_copy_is_async_and_outside_iteration_timing(self):
        class Scalar:
            def __init__(self, value): self.value = value
            def to(self, dtype): return self
        class Radii:
            device = 'stub-cuda'
            def numel(self): return 3
            def sum(self, dtype): return Scalar(12)
            def amax(self): return Scalar(8)
            def detach(self): return self
        class Host:
            def copy_(self, values, non_blocking):
                self.values = [value.value for value in values]
                self.non_blocking = non_blocking
            def tolist(self):
                if not FakeEvent.ready:
                    raise AssertionError('Reading unfinished async copy')
                return self.values
        host = Host()
        def allocate(size, dtype, device, pin_memory):
            self.assertEqual((size, device, pin_memory), (3, 'cpu', True))
            return host
        torch_stub = SimpleNamespace(
            cuda=FakeCuda, __version__='stub', version=SimpleNamespace(cuda=None),
            no_grad=nullcontext, float64='float64',
            count_nonzero=lambda radii: Scalar(2), stack=lambda values: values,
            empty=allocate)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(TrainingProfiler, '_git_metadata', return_value={}):
                profiler = TrainingProfiler(tmp, options(profile_no_raster=False), {},
                                            torch_module=torch_stub)
            FakeEvent.ready = False
            profiler.begin_iteration(100, GAUSSIANS)
            camera = SimpleNamespace(image_name='test', image_width=32, image_height=32)
            profiler.capture_render(
                {'render': SimpleNamespace(grad_fn=None), 'radii': Radii()}, camera, 2, False)
            self.finish(profiler)
            sample = profiler.pending[0]
            self.assertLess(sample['end'].stamp, sample['phases']['raster_stats'][0].stamp)
            self.assertTrue(host.non_blocking)
            self.assertEqual(profiler.rows_written, 0)
            FakeEvent.ready = True
            profiler.drain()
            row = read_rows(profiler.path)[0]
            self.assertEqual(row['visible_gaussians'], '2')
            self.assertEqual(float(row['radius_mean_visible_px']), 6)
            self.assertEqual(float(row['radius_max_px']), 8)
            profiler.close()


class MonitoringTests(unittest.TestCase):
    def test_unknown_values_are_not_zero_and_partial_process_total_is_unknown(self):
        for value in ("[N/A]", "N/A", "[Not Supported]", "", "nan", "inf", "-1", None):
            self.assertIsNone(gpu_monitor.number(value))
        self.assertEqual(gpu_monitor.number("0"), 0)
        with patch.object(gpu_monitor, "query", return_value=[
                ["12", "100"], ["12", "[N/A]"], ["13", "999"]]):
            self.assertIsNone(gpu_monitor.process_vram({12}))

    def test_na_does_not_stop_device_telemetry_or_later_process_samples(self):
        device = {"gpu_index": "0", "gpu_uuid": "GPU-test",
                  "gpu_utilization_percent": 90, "device_vram_used_mb": 4000,
                  "device_vram_total_mb": 24000, "sm_clock_mhz": 2000,
                  "temperature_c": 60, "power_w": 200}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitor.csv"
            monitor = ProcessMonitor(path)
            with patch.object(gpu_monitor, "process_vram", side_effect=[
                    100.0, None, RuntimeError("temporary"), 120.0]), \
                    patch.object(gpu_monitor, "device_telemetry", return_value=[device]):
                for _ in range(4):
                    monitor.sample(os.getpid())
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["process_vram_status"] for r in rows],
                             ["available", "unavailable", "query_error", "available"])
            self.assertTrue(all(r["gpu_utilization_percent"] == "90" for r in rows))
            self.assertEqual(monitor.metrics()["average_vram_mb"], 110.0)
            self.assertEqual(monitor.metrics()["peak_vram_mb"], 120.0)
            monitor.close()

    def test_device_fields_are_independently_optional(self):
        rows = [["0", "GPU-test", "[N/A]", "4000", "24000", "2100", "65", "[N/A]"]]
        with patch.object(gpu_monitor, "_device_fields", gpu_monitor.DEVICE_FIELDS), \
                patch.object(gpu_monitor, "query", return_value=rows):
            result = gpu_monitor.device_telemetry()[0]
        self.assertIsNone(result["gpu_utilization_percent"])
        self.assertIsNone(result["power_w"])
        self.assertEqual(result["sm_clock_mhz"], 2100)
        self.assertEqual(result["temperature_c"], 65)

    def test_unsupported_clock_keeps_supported_temperature_and_caches_query(self):
        calls = []
        def query(args):
            fields = args[0].split("=", 1)[1].split(",")
            calls.append(fields)
            if "clocks.sm" in fields:
                raise subprocess.CalledProcessError(1, "nvidia-smi")
            data = dict(zip(gpu_monitor.DEVICE_FIELDS,
                            ["0", "GPU-test", "50", "100", "200", "999", "65", "100"]))
            return [[data[key] for key in fields]]
        with patch.object(gpu_monitor, "_device_fields", gpu_monitor.DEVICE_FIELDS), \
                patch.object(gpu_monitor, "query", side_effect=query):
            first = gpu_monitor.device_telemetry()[0]
            previous = len(calls)
            second = gpu_monitor.device_telemetry()[0]
        self.assertIsNone(first["sm_clock_mhz"])
        self.assertEqual(first["temperature_c"], 65)
        self.assertEqual(second["power_w"], 100)
        self.assertEqual(len(calls), previous + 1)

    def test_resource_guard_stays_conservative_when_free_memory_is_unknown(self):
        with patch.object(gpu_monitor, "query", return_value=[["0", "GPU-test", "[N/A]"]]), \
                patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                gpu_monitor.free_vram()

    def test_existing_resource_csv_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitor.csv"
            path.write_text("old experiment")
            monitor = ProcessMonitor(path, sample_gpu=False)
            monitor.sample(os.getpid())
            self.assertEqual(path.read_text(), "old experiment")
            self.assertTrue(monitor.csv_failed)
            monitor.close()



class IntegrationTests(unittest.TestCase):
    def test_cli_accepts_profiling_without_changing_model_or_optimization_defaults(self):
        from research_pipeline.config import repository_parser
        parser = repository_parser('training')
        baseline = parser.parse_args([])
        profiled = parser.parse_args([
            '--profile_training', '--profile_from', '1000', '--profile_until', '5000',
            '--profile_control_samples'])
        self.assertFalse(baseline.profile_training)
        self.assertTrue(profiled.profile_training)
        for name in ('iterations', 'resolution', 'densify_from_iter',
                     'densify_until_iter', 'densification_interval',
                     'densify_grad_threshold', 'opacity_reset_interval', 'percent_dense'):
            self.assertEqual(getattr(baseline, name), getattr(profiled, name))

    def test_runner_avoids_duplicate_gpu_queries_and_persists_ram_csv(self):
        import sys
        from research_pipeline.runner import run_process
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.log'
            with patch.object(gpu_monitor, 'process_vram', side_effect=AssertionError('duplicate query')), \
                    patch.object(gpu_monitor, 'device_telemetry', side_effect=AssertionError('duplicate query')):
                result = run_process([
                    sys.executable, '-c', 'import time; time.sleep(0.08)',
                    '--profile_training'], path, interval=0.01, tee=False)
            self.assertEqual(result.returncode, 0)
            rows = read_rows(Path(result.metrics['resource_csv']))
            self.assertTrue(rows)
            self.assertTrue(all(r['device_status'] == 'handled_by_training_profiler' for r in rows))

class AlgorithmPreservationTests(unittest.TestCase):
    def test_training_core_matches_baseline_fingerprint(self):
        # Fingerprint of the original training function before instrumentation.
        # Independent of later commits, git availability, or backup directory.
        root = Path(__file__).resolve().parents[1]
        current = ast.parse((root / 'train.py').read_text(encoding='utf-8'))
        instrumented = next(n for n in current.body
                            if isinstance(n, ast.FunctionDef) and n.name == '_training')
        original_header = ast.parse(
            'def training(dataset, opt, pipe, testing_iterations, saving_iterations, '
            'checkpoint_iterations, checkpoint, debug_from): pass').body[0]

        class Strip(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == "profiler is not None":
                    return None
                return self.generic_visit(node)
            def visit_With(self, node):
                if (len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call)
                        and isinstance(node.items[0].context_expr.func, ast.Name)
                        and node.items[0].context_expr.func.id == "profile_phase"):
                    return [self.visit(n) for n in node.body]
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name == "densify_before":
                        return None
                    if name == "densify_result":
                        return ast.Expr(value=node.value)
                return self.generic_visit(node)

        normalized = Strip().visit(copy.deepcopy(instrumented))
        normalized.name = "training"
        normalized.args = copy.deepcopy(original_header.args)
        # Output setup moved once into the public wrapper for guaranteed cleanup.
        normalized.body.insert(1, ast.parse(
            'tb_writer = prepare_output_and_logger(dataset)').body[0])
        digest = hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, 'd95fedb8c72ca6c30a1f03e2af42c7810f1842a64ea4f618204c108cff585d61')
        # No accidental diagnostic stop or optimization parameter assignment.
        self.assertFalse(any(isinstance(n, ast.Break) for n in ast.walk(normalized)
                             if getattr(n, "lineno", 0) > 200))


if __name__ == "__main__":
    unittest.main()
