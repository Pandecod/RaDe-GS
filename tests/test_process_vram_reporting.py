"""Process-only VRAM aggregation and compatible report upgrades."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from research_pipeline import gpu_monitor
from research_pipeline.process_monitor import ProcessMonitor
from research_pipeline.reporting import (
    FIELDS, SUMMARY_FIELDS, PROCESS_VRAM_FIELDS, export, memory_info, read_csv,
    read_json, read_report_table, sampled_process_memory, update_result_tables,
    write_table,
)
from research_pipeline.results import csv_row
from research_pipeline.utils import write_json
from test_research_reporting import fixture


class ProcessVRAMTests(unittest.TestCase):
    def test_monitor_filters_other_apps_and_keeps_children_with_unreadable_rss(self):
        root, child = Mock(pid=101), Mock(pid=102)
        root.children.return_value = [child]
        root.memory_info.return_value = SimpleNamespace(rss=1024**2)
        child.memory_info.side_effect = psutil.AccessDenied(102)
        device = {"gpu_uuid": "GPU-one", "device_vram_used_mb": 8192.0}
        monitor = ProcessMonitor()
        with patch("research_pipeline.process_monitor.psutil.Process", return_value=root), \
             patch.object(gpu_monitor, "query", side_effect=[
                 [["101", "100"], ["102", "50"], ["999", "8000"]],
                 [["101", "50"], ["102", "0"], ["999", "16000"]]]), \
             patch.object(gpu_monitor, "device_telemetry", return_value=[device]):
            monitor.sample(101)
            monitor.sample(101)
        measured = monitor.metrics()
        self.assertEqual(measured["peak_vram_mb"], 150.0)
        self.assertEqual(measured["average_vram_mb"], 100.0)
        self.assertEqual(measured["vram_sample_count"], 2)
        self.assertEqual(measured["device_global"]["device_vram_used_mb"]["peak"], 8192.0)
        technical = csv_row({"training": measured})
        self.assertEqual(technical["training_peak_vram_mb"], 150.0)
        self.assertEqual(technical["training_average_vram_mb"], 100.0)

    def test_deduplicates_gpu_rows_and_ignores_na_errors_and_conflicting_copies(self):
        rows = [
            {"utc": "a", "process_vram_mb": "100"},
            {"utc": "a", "process_vram_mb": "100"},
            {"utc": "b", "process_vram_mb": "200"},
            {"utc": "c", "process_vram_mb": "0"},
            {"utc": "d", "process_vram_mb": "N/A"},
            {"utc": "e", "process_vram_mb": "9999", "process_vram_status": "query_error"},
            {"utc": "f", "process_vram_mb": "-1"},
            {"utc": "g", "process_vram_mb": "nan"},
        ]
        measured = sampled_process_memory(rows)
        self.assertEqual(measured["peak_vram_mb"], 200.0)
        self.assertEqual(measured["average_vram_mb"], 100.0)
        self.assertEqual(measured["process_vram_samples"], 3)
        conflicting = [{"utc": "a", "process_vram_mb": v} for v in ("100", "200", "100")]
        self.assertIsNone(sampled_process_memory(conflicting)["peak_vram_mb"])

    def test_training_numbers_reach_csv_html_and_details_without_other_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, summary = fixture(Path(tmp), count=1)
            summary["training"].update(peak_vram_mb=23126.14453, average_vram_mb=10012.82641)
            summary["rendering"].update(peak_vram_mb=99999, average_vram_mb=88888)
            summary["mesh_extraction"] = {"peak_vram_mb": 77777, "average_vram_mb": 66666}
            source = output / "research/summary.json"
            write_json(source, summary)
            before = source.read_bytes()
            destination = export([output], Path(tmp) / "report", pairs=1)
            for filename in ("ringkasan_hasil.csv", "tabel_evaluasi.csv"):
                row = read_csv(destination / filename)[0]
                self.assertEqual(row["Peak VRAM seluruh GPU (GiB)"], "2.0")
                self.assertEqual(row["peak_vram_mb"], "23126.14453")
                self.assertEqual(row["average_vram_mb"], "10012.82641")
            training = read_csv(destination / "rincian_proses.csv")[0]
            self.assertEqual(training["VRAM puncak proses (MiB)"], "23126.14453")
            html = (destination / "index.html").read_text(encoding="utf-8")
            self.assertIn("<th>peak_vram_mb</th>", html)
            self.assertIn("10012.82641", html)
            measured = read_json(destination / "detail_laporan.json")["experiments"][0]["memory"]
            self.assertEqual(measured["process_vram_source"], "summary.training")
            self.assertEqual(csv_row(summary)["training_peak_vram_mb"], 23126.14453)
            self.assertEqual(source.read_bytes(), before)

    def test_external_csv_process_values_are_independent_of_device_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, summary = fixture(Path(tmp), count=1)
            for path in (output / "research/profiling").glob("*"):
                path.unlink()
            resource = output / "research/logs/training_resources_test.csv"
            resource.parent.mkdir()
            summary["training"].update(resource_csv="/server/research/logs/" + resource.name,
                                       peak_vram_mb=None, average_vram_mb=None)
            write_json(output / "research/summary.json", summary)
            resource.write_text(
                "utc,root_pid,gpu_uuid,device_status,device_vram_used_mb,process_vram_mb,process_vram_status\n"
                "a,101,GPU-one,available,8192,100,available\n"
                "a,101,GPU-two,available,12288,100,available\n"
                "b,101,,query_error,,300,available\n"
                "c,101,GPU-one,available,16384,,unavailable\n")
            measured = memory_info(output)
            self.assertIsNone(measured["gpu_total_peak_gib"])
            self.assertEqual(measured["peak_vram_mb"], 300.0)
            self.assertEqual(measured["average_vram_mb"], 200.0)
            self.assertEqual(measured["process_vram_samples"], 2)
            destination = export([output], Path(tmp) / "report", pairs=1)
            self.assertEqual(read_csv(destination / "ringkasan_hasil.csv")[0]["average_vram_mb"], "200.0")
            self.assertEqual(read_csv(destination / "rincian_proses.csv")[0]["VRAM rata-rata proses (MiB)"], "200.0")

    def test_profiler_csv_uses_process_samples_without_device_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = fixture(Path(tmp), count=1)
            gpu = output / "research/profiling/gpu_run.csv"
            gpu.write_text(
                "utc,gpu_uuid,device_status,device_vram_used_mb,process_vram_mb,process_vram_status\n"
                "a,GPU-one,available,8192,100,available\n"
                "b,GPU-one,available,10240,300,available\n"
                "c,GPU-one,available,12288,N/A,unavailable\n")
            measured = memory_info(output)
            self.assertEqual(measured["gpu_total_peak_gib"], 12.0)
            self.assertEqual(measured["peak_vram_mb"], 300.0)
            self.assertEqual(measured["average_vram_mb"], 200.0)
            self.assertEqual(measured["process_vram_samples"], 2)
            gpu.write_text(
                "device_status,gpu_uuid,device_vram_used_mb,process_vram_mb,process_vram_status\n"
                "available,GPU-one,2048,N/A,unavailable\n")
            destination = export([output], Path(tmp) / "report", pairs=1)
            row = read_csv(destination / "ringkasan_hasil.csv")[0]
            self.assertEqual(row["Peak VRAM seluruh GPU (GiB)"], "2.0")
            self.assertEqual(row["peak_vram_mb"], "")
            self.assertEqual(row["average_vram_mb"], "")
            self.assertEqual(memory_info(output)["pytorch_allocated_peak_gib"], 1.0)

    def test_legacy_tables_upgrade_without_losing_old_rows_and_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, summary = fixture(root, count=1)
            summary["training"].update(peak_vram_mb=300, average_vram_mb=200)
            write_json(output / "research/summary.json", summary)
            destination = export([output], root / "report", pairs=1)
            old = {"Folder": "lama", "Folder output": "/old/run",
                   "Peak VRAM seluruh GPU (GiB)": "7.5", "Training (menit)": "40.25"}
            tables = (("_laporan.csv", FIELDS), ("_ringkasan.csv", SUMMARY_FIELDS))
            for suffix, fields in tables:
                write_table(root / ("results" + suffix),
                            [key for key in fields if key not in PROCESS_VRAM_FIELDS], [old])
            update_result_tables(destination, root / "results.csv")
            for suffix, fields in tables:
                rows = read_csv(root / ("results" + suffix))
                self.assertEqual(list(rows[0]), fields)
                previous, current = rows
                self.assertEqual(previous["Peak VRAM seluruh GPU (GiB)"], "7.5")
                self.assertEqual(previous["Training (menit)"], "40.25")
                self.assertEqual(previous["peak_vram_mb"], "")
                self.assertEqual(current["peak_vram_mb"], "300.0")
                self.assertEqual(current["average_vram_mb"], "200.0")
            bad = root / "bad.csv"
            bad.write_text("Folder,unexpected\nold,data\n")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                read_report_table(bad, SUMMARY_FIELDS)


if __name__ == "__main__":
    unittest.main()
