"""Offline navigation for report packages; no model or metric computation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import quote
import uuid

from research_pipeline.utils import atomic_write, write_json

STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#18232e;font:16px/1.6 system-ui,sans-serif}
main{max-width:1150px;margin:auto;padding:36px 22px}h1{font-size:2rem;line-height:1.2}h2{margin-top:30px}a{color:#145da0}
nav,.links{display:flex;gap:12px;flex-wrap:wrap}.links a,nav a{padding:8px 12px;background:#fff;border:1px solid #d5dce3;border-radius:7px;text-decoration:none}
section,.card{background:#fff;border:1px solid #d5dce3;border-radius:10px;padding:22px;margin:20px 0}.muted{color:#52616e}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e8ed;vertical-align:top}.scroll{overflow-x:auto}
figure{margin:22px 0}img{width:100%;height:auto;border:1px solid #d5dce3}figcaption{font-size:.9rem;color:#52616e}
code{overflow-wrap:anywhere;font-size:.9rem}details{margin-top:16px}.status{font-weight:650}.ok{color:#146342}.bad{color:#a53125}
@media(max-width:600px){main{padding:22px 14px}section,.card{padding:16px}h1{font-size:1.6rem}}
"""


def text(value):
    if isinstance(value, float):
        value = round(value, 6)
    return escape(str(value)) if value is not None and value != "" else "Tidak tersedia"


def relative_link(path, directory):
    relative = Path(path).absolute().relative_to(Path(directory).absolute()).as_posix()
    return quote(relative, safe="/.-_")


def document(title, content):
    return ("<!doctype html><html lang=\"id\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{content}</main></body></html>")


def central_directory(master_csv, *, prefix="laporan_"):
    master_csv = Path(master_csv).resolve()
    group = master_csv.stem.removeprefix("experiment_results_")
    group = re.sub(r"[^A-Za-z0-9_.-]", "_", group) or "hasil"
    name = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    return master_csv.parent / "laporan" / group / name


def write_report_index(directory, details):
    """Link only artifacts inside this package, so copied packages stay readable."""
    directory = Path(directory).resolve()
    columns = ["Folder", "Metode", "Status", "Split evaluasi", "PSNR (dB)", "SSIM", "LPIPS", "Training (menit)", "Peak VRAM seluruh GPU (GiB)"]
    body = ["<h1>Laporan penelitian rekonstruksi 3D</h1>",
            "<p class=\"muted\">Tabel, foto perbandingan, dan sumber pengukuran dalam satu paket. Buka berkas HTML ini setelah folder disalin.</p>",
            "<nav>"]
    links = [("ringkasan_hasil.csv", "Ringkasan utama"), ("tabel_evaluasi.csv", "Evaluasi lengkap"),
             ("rincian_proses.csv", "Waktu, RAM, dan VRAM"), ("detail_laporan.json", "Detail dan sumber data"),
             ("PETUNJUK_LAPORAN.md", "Petunjuk laporan"), ("pemulihan_mesh.csv", "Pemulihan mesh"),
             ("riwayat_percobaan.csv", "Riwayat percobaan")]
    for filename, title in links:
        if (directory / filename).is_file():
            body.append(f'<a href="{quote(filename)}">{escape(title)}</a>')
    body.append('</nav>')
    collection_path = directory / "pengumpulan.json"
    if collection_path.is_file():
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        body.append('<p>Paket gabungan memilih hasil sukses dari run sumber terbaru per objek/metode, lalu retry mesh sukses terbaru untuk sumber itu. '
                    'Jika belum ada hasil sukses, kegagalan terakhir tetap ditampilkan. Pemilihan tidak berdasarkan besarnya metrik atau tampilan foto.</p>')
        if collection.get("unverified_records"):
            body.append(f'<p class="status bad">Ada {text(collection["unverified_records"])} catatan yang tidak dapat diverifikasi. Periksa riwayat_percobaan.csv sebelum memakai paket ini.</p>')
    body.append('<p>Peak VRAM adalah puncak sampel seluruh GPU saat training, termasuk aplikasi lain. Nilai kosong berarti tidak tersedia. '
                'Foto dipilih merata berdasarkan urutan nama, tanpa memilih hasil yang tampak paling bagus. '
                'Pasangan menampilkan foto referensi di kiri dan render model di kanan.</p>')
    body.append('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(c)}</th>' for c in columns) + '</tr></thead><tbody>')
    for index, item in enumerate(details, 1):
        row = item["table_row"]
        cells = []
        for key in columns:
            value = text(row.get(key))
            if key == "Folder":
                value = f'<a href="#objek-{index}">{value}</a>'
            cells.append('<td>' + value + '</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    body.append('</tbody></table></div>')
    for index, item in enumerate(details, 1):
        row, review = item["table_row"], item["comparison"]
        body.append(f'<section id="objek-{index}"><h2>{text(row.get("Folder"))} · {text(row.get("Metode"))}</h2>')
        status = row.get("Status")
        css = "ok" if status in ("success", "success_mesh_recovered") else "bad"
        body.append(f'<p class="status {css}">{text(status)}</p>')
        if item.get("recovery"):
            settings = item["recovery"].get("settings", {})
            body.append(f'<p>Mesh dari retry sukses; block resolution {text(settings.get("block_resolution"))}, voxel {text(settings.get("voxel_size"))}, '
                        f'depth {text(settings.get("max_depth"))}. Training dan evaluasi memakai run asal. Evaluasi GT mesh baru tidak dihitung oleh ekspor ini.</p>')
        files = review.get("files", [])
        body.append(f'<p>Foto perbandingan: {text(review.get("status"))}; split {text(review.get("split"))}; {len(files)} pasangan.</p>')
        for file in files:
            link = relative_link(file, directory)
            name = Path(file).name
            body.append(f'<figure><a href="{link}"><img loading="lazy" src="{link}" alt="{escape(str(row.get("Folder", "")), quote=True)}: foto referensi dan hasil render"></a>'
                        f'<figcaption>{escape(name)} · <a href="{link}" download>Unduh gambar</a></figcaption></figure>')
        if not files:
            body.append('<p>Pasangan foto belum tersedia untuk hasil ini. Periksa status rendering dan sumber data.</p>')
        inventory = item.get("photo_inventory")
        if inventory:
            body.append(f'<p><a href="{relative_link(inventory, directory)}">Daftar nama foto, split, dan pasangan render</a></p>')
        data = item.get("data", {})
        body.append('<details><summary>Lokasi sumber hasil</summary>'
                    f'<p>Training/render: <code>{text(data.get("asset_root"))}</code></p>'
                    f'<p>Hasil terpilih/mesh: <code>{text(item.get("original_output_path"))}</code></p>'
                    f'<p>Ringkasan sumber: <code>{text(item.get("source_summary"))}</code></p></details></section>')
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objects": [{"scene": item["table_row"].get("Folder"), "method": item["table_row"].get("Metode"),
                     "status": item["table_row"].get("Status")} for item in details],
    }
    atomic_write(directory / "index.html", lambda stream: stream.write(document("Laporan penelitian", ''.join(body))))
    write_json(directory / "laporan.json", payload)


def refresh_catalog(results_root):
    """One entry page for all centralized report packages in a results folder."""
    root = Path(results_root).resolve()
    items = []
    for path in (root / "laporan").glob("*/*/laporan.json"):
        if not (path.parent / "index.html").is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items.append((data["created_utc"], path, data))
    body = ['<h1>Pusat laporan penelitian</h1><p>Mulai dari halaman ini untuk membuka tabel dan foto perbandingan. '
            'Setiap paket menyimpan hasil ekspornya sendiri; sumber model dan log tetap berada di folder eksperimen.</p>']
    body.append('<h2>Tabel gabungan</h2><div class="links">')
    for pattern in ("*_ringkasan.csv", "*_laporan.csv", "*_pemulihan_mesh.csv"):
        for path in sorted(root.glob(pattern)):
            body.append(f'<a href="{relative_link(path, root)}">{escape(path.name)}</a>')
    body.append('</div><h2>Paket laporan terbaru</h2>')
    for stamp, path, data in sorted(items, key=lambda value: value[0], reverse=True):
        local = datetime.fromisoformat(stamp).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        names = ', '.join(str(item.get("scene", "")) for item in data["objects"])
        counts = {}
        for item in data["objects"]:
            status = str(item.get("status"))
            counts[status] = counts.get(status, 0) + 1
        states = ', '.join(f'{name}: {count}' for name, count in sorted(counts.items()))
        body.append(f'<article class="card"><h3><a href="{relative_link(path.parent / "index.html", root)}">{escape(path.parent.parent.name)} · {escape(local)} (UTC+08)</a></h3>'
                    f'<p>{escape(names)}</p><p class="muted">{escape(states)}</p></article>')
    if not items:
        body.append('<p>Belum ada paket laporan terpadu. Jalankan collector atau ekspor dengan --results-csv.</p>')
    atomic_write(root / "MULAI_DI_SINI.html", lambda stream: stream.write(document("Pusat laporan penelitian", ''.join(body))))
    return root / "MULAI_DI_SINI.html"
