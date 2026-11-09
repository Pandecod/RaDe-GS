"""Real report fixtures verify organization, provenance, selection, and portable visuals."""
import copy
from html.parser import HTMLParser
from pathlib import Path
import shutil
import tempfile
import unittest
from urllib.parse import unquote
from unittest.mock import patch

from PIL import Image
from research_pipeline.config import DEFAULTS
from research_pipeline.report_collection import collect, inspect_outputs
from research_pipeline.report_index import write_report_index
from research_pipeline.reporting import automatic_report, read_csv, read_json
from research_pipeline.utils import write_json
from test_research_reporting import fixture
from test_mesh_recovery_reporting import source_run, mesh_retry


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value and not value.startswith("#"):
                self.paths.append(value)


def config(root, dataset="togog_bali"):
    data = copy.deepcopy(DEFAULTS)
    data["project"]["dataset"] = dataset
    data["output"]["root"] = str(root / "output" / dataset)
    data["results"]["csv"] = str(root / "research_results/hasil_radegs_v2" / f"experiment_results_{dataset}.csv")
    return data


class ReportCollectionTests(unittest.TestCase):
    def test_automatic_package_is_centralized_and_copyable_without_original_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, _ = fixture(root / "output")
            original = {p:p.read_bytes() for p in output.rglob("*") if p.is_file()}
            results = root / "research_results/hasil_radegs_v2/experiment_results_togog_bali.csv"
            report = automatic_report(output, results)
            folder = Path(report["directory"])
            self.assertEqual(folder.parent, results.parent / "laporan/togog_bali")
            self.assertFalse(any((output / "research").glob("laporan_*")))
            self.assertTrue((folder / "index.html").is_file())
            self.assertTrue((results.parent / "MULAI_DI_SINI.html").is_file())
            self.assertEqual(original, {p:p.read_bytes() for p in original})
            copied = root / "salinan_laporan"
            shutil.copytree(folder,copied)
            parser = Links()
            parser.feed((copied / "index.html").read_text(encoding="utf-8"))
            self.assertGreater(len(parser.paths),5)
            for url in parser.paths:
                self.assertNotIn(":",url)
                self.assertFalse(url.startswith("/"))
                self.assertTrue((copied / unquote(url)).is_file(), url)
            with Image.open(copied / "01_patung/00006_comparison.png") as pair:
                header=pair.height-12
                self.assertEqual(pair.getpixel((0,header)),(60,40,60))
                self.assertEqual(pair.getpixel((24,header)),(60,40,60))

    def test_collects_nested_success_and_retry_without_changing_technical_csv_or_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=config(root)
            outputs=Path(cfg["output"]["root"])
            failed, summary=source_run(outputs,"detya")
            summary.update(failed_stage="training",timestamp="2026-11-04T01:00:00+00:00")
            write_json(failed / "research/summary.json",summary)
            completed,new=source_run(failed / "runs","run_baru",status="success")
            new.update(scene_name="detya",timestamp="2026-11-05T01:00:00+00:00")
            write_json(completed / "research/summary.json",new)
            tualen,old=source_run(outputs,"tualen")
            retry,_=mesh_retry(tualen,old)
            results=Path(cfg["results"]["csv"])
            results.parent.mkdir(parents=True)
            results.write_bytes(b"original technical CSV remains unchanged")
            before={p:p.read_bytes() for p in outputs.rglob("*") if p.is_file()}
            result=collect(cfg,pairs=1)
            directory=Path(result["directory"])
            rows={r["Folder"]:r for r in read_csv(directory / "ringkasan_hasil.csv")}
            self.assertEqual(set(rows),{"detya","tualen"})
            self.assertEqual(rows["detya"]["Folder output"],str(completed))
            self.assertEqual(rows["tualen"]["Folder output"],str(retry))
            self.assertEqual(rows["tualen"]["Status"],"success_mesh_recovered")
            self.assertEqual(rows["tualen"]["PSNR (dB)"],"25.123457")
            self.assertEqual(rows["tualen"]["Split evaluasi"],"test")
            history=read_csv(directory / "riwayat_percobaan.csv")
            self.assertEqual(len(history),4)
            self.assertEqual(sum(r["Terpilih"]=="True" for r in history),2)
            self.assertTrue(any(r["Folder"]=="detya" and r["Status"]=="failed" for r in history))
            self.assertEqual(results.read_bytes(),b"original technical CSV remains unchanged")
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            self.assertEqual(read_json(directory / "detail_laporan.json")["experiments"][1]["recovery"]["settings"]["block_resolution"],8)
            self.assertIn('riwayat_percobaan.csv',(directory / "index.html").read_text(encoding="utf-8"))

    def test_newest_source_success_is_selected_without_metric_quality_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs=Path(tmp) / "output"
            old,summary=source_run(outputs,"patung",status="success")
            summary["rendering"]["psnr"]=40
            summary["timestamp"]="2026-11-04T01:00:00+00:00"
            write_json(old / "research/summary.json",summary)
            newer,data=source_run(old / "runs","run2",status="success")
            data.update(scene_name="patung",timestamp="2026-11-05T01:00:00+00:00")
            data["rendering"]["psnr"]=20
            write_json(newer / "research/summary.json",data)
            newest,bad=source_run(old / "runs","run3")
            bad.update(scene_name="patung",timestamp="2026-11-06T01:00:00+00:00",failed_stage="training")
            write_json(newest / "research/summary.json",bad)
            chosen,history=inspect_outputs(outputs)
            self.assertEqual(len(chosen),1)
            self.assertEqual(chosen[0]["output"],newer)
            self.assertEqual(chosen[0]["summary"]["rendering"]["psnr"],20)
            self.assertEqual(len(history),3)

    def test_failed_and_unverified_retries_are_visible_and_never_become_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs=Path(tmp)
            model,summary=source_run(outputs,"tualen")
            retry,record=mesh_retry(model,summary)
            (retry / "recon_post.ply").unlink()
            incomplete=model / "research/mesh_retry_running"
            incomplete.mkdir()
            write_json(incomplete / "result.json",{"status":"running","attempt_path":str(incomplete)})
            chosen,history=inspect_outputs(outputs)
            self.assertEqual(chosen[0]["summary"]["status"],"failed")
            self.assertIsNone(chosen[0]["retry"])
            self.assertTrue(any(row["Status"]=="tidak_dapat_diverifikasi" for row in history))
            self.assertTrue(any(row["Status"]=="running" for row in history))

    def test_datasets_share_one_entry_but_keep_separate_packages_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for dataset in ("togog_bali","tanks_temples"):
                cfg=config(root,dataset)
                source_run(Path(cfg["output"]["root"]),"nama_sama",status="success")
                collect(cfg,pairs=1)
            results=root / "research_results/hasil_radegs_v2"
            index=(results / "MULAI_DI_SINI.html").read_text(encoding="utf-8")
            self.assertIn("togog_bali",index)
            self.assertIn("tanks_temples",index)
            parser=Links()
            parser.feed(index)
            for url in parser.paths:
                self.assertTrue((results / unquote(url)).is_file(),url)
            self.assertEqual(len(list((results / "laporan").glob("*/*/ringkasan_hasil.csv"))),2)
            self.assertEqual(len(read_csv(results / "experiment_results_tanks_temples_ringkasan.csv")),1)

    def test_dry_run_and_missing_output_create_no_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=config(root)
            with self.assertRaisesRegex(ValueError,"belum ada"):
                collect(cfg,dry_run=True)
            source_run(Path(cfg["output"]["root"]),"patung",status="success")
            paths_before=set(root.rglob("*"))
            with patch("subprocess.Popen",side_effect=AssertionError("No training")):
                result=collect(cfg,dry_run=True)
            self.assertEqual(result["selected"],1)
            self.assertEqual(set(root.rglob("*")),paths_before)
            self.assertFalse(Path(cfg["results"]["csv"]).parent.exists())

    def test_corrupt_history_is_visible_without_promoting_its_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=config(root)
            outputs=Path(cfg["output"]["root"])
            model,summary=source_run(outputs,"patung",status="success")
            broken=model / "runs/broken/research/summary.json"
            write_json(broken,{"scene_name":"patung","output_path":str(broken.parent.parent),"timestamp":None})
            result=collect(cfg,pairs=1)
            folder=Path(result["directory"])
            history=read_csv(folder / "riwayat_percobaan.csv")
            self.assertEqual(sum(row["Status"]=="tidak_dapat_dibaca" for row in history),1)
            html=(folder / "index.html").read_text(encoding="utf-8")
            self.assertIn("tidak dapat diverifikasi",html)
            self.assertIn("25.123457",html)
            self.assertNotIn("25.123456789",html)
            self.assertEqual(len(read_csv(folder / "ringkasan_hasil.csv")),1)

    def test_different_methods_for_one_scene_are_never_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs=Path(tmp)
            original,_=source_run(outputs,"patung",status="success")
            scaffold,summary=source_run(original / "runs","scaffold",status="success")
            summary.update(scene_name="patung",method="RaDe-GS + Scaffold anchor init")
            write_json(scaffold / "research/summary.json",summary)
            chosen,_=inspect_outputs(outputs)
            self.assertEqual(len(chosen),2)
            self.assertEqual({row["summary"].get("method","RaDe-GS") for row in chosen},
                             {"RaDe-GS","RaDe-GS + Scaffold anchor init"})

    def test_html_escapes_scene_names_and_never_labels_missing_images_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model,summary=fixture(root / "output",count=1)
            summary["scene_name"]='<script>alert("x")</script>'
            write_json(model / "research/summary.json",summary)
            shutil.rmtree(model / "train")
            report=automatic_report(model,root / "results.csv")
            html=(Path(report["directory"]) / "index.html").read_text(encoding="utf-8")
            self.assertNotIn('<script>',html)
            self.assertIn('&lt;script&gt;',html)
            self.assertIn('missing_images',html)
            self.assertIn('belum tersedia',html)


if __name__ == "__main__":
    unittest.main()
