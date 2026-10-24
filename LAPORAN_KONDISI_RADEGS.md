# Laporan Kondisi RaDe-GS

Tanggal pemeriksaan terakhir: 23 Oktober 2026.

Dokumen ini berisi kondisi implementasi dan hasil pemeriksaan teknis. Petunjuk
menjalankan proyek dipisahkan ke
[Panduan Penggunaan Pipeline RaDe-GS](PANDUAN_PIPELINE_RADEGS.md).

## 1. Ringkasan kondisi

Repository berisi RaDe-GS dan Research Pipeline untuk memproses scene secara
berurutan. Satu pemanggilan dapat dimulai dari foto mentah melalui COLMAP, atau
langsung dari training untuk dataset yang sudah diproses. Setelah itu pipeline
menangani rendering, metrik citra, mesh, evaluasi geometri opsional, monitoring,
log, dan pencatatan hasil.

Pemeriksaan CPU terakhir menghasilkan:

```text
python -B -m unittest discover -s tests -v
Ran 51 tests
OK
```

Pemeriksaan sintaks seluruh modul yang diubah juga berhasil. Training,
rendering, LPIPS, dan mesh dengan GPU nyata belum dijalankan dari environment
Windows pemeriksaan karena Python aktif tidak memiliki PyTorch/CUDA. Validasi
runtime tersebut tetap perlu dilakukan di environment Linux/WSL NVIDIA.

## 2. Alur pipeline aktual

```text
Validasi input sesuai titik awal
  -> COLMAP, jika dipilih
  -> Training RaDe-GS
  -> Rendering, jika diaktifkan
  -> SSIM / PSNR / LPIPS, jika diaktifkan
  -> Pemeriksaan RAM dan VRAM
  -> Ekstraksi dan analisis mesh, jika diaktifkan
  -> Evaluasi geometri, jika diaktifkan
  -> Summary JSON dan master CSV
  -> Scene berikutnya
```

Semua subprocess berat dijalankan secara serial. Training dan mesh tidak
berjalan bersamaan dalam satu proses pipeline.

Titik awal dipilih dengan `--start-from training` atau `--start-from colmap`.
Mode training tetap menjadi default untuk menjaga perilaku perintah lama.

## 3. Kemampuan yang tersedia

| Kemampuan | Kondisi implementasi |
|---|---|
| Discovery scene | Membaca direktori anak langsung dari `dataset.root` dan mengurutkannya berdasarkan nama |
| Validasi dataset | Memeriksa `input/` untuk mode COLMAP, atau `images/`, `sparse/0/`, cameras/images, dan points3D untuk mode training |
| COLMAP opsional | Menjalankan feature extraction, matching, mapper, dan undistortion sebelum training |
| Publikasi aman COLMAP | Menyusun hasil dalam attempt sementara dan menerbitkannya setelah output lengkap |
| List dan dry-run | Memeriksa scene dan command tanpa mengimpor CUDA atau menjalankan model |
| Training | Menjalankan `train.py` untuk setiap scene |
| Progress live | Output biasa dan pembaruan `tqdm` berbasis carriage return langsung diteruskan ke terminal dan log |
| Rendering | Dapat diaktifkan atau dimatikan melalui konfigurasi |
| Metrik citra | SSIM, PSNR, dan LPIPS; dapat dimatikan terpisah dari rendering |
| Mesh | Ekstraksi TSDF dan analisis jumlah vertex, face, serta ukuran file |
| Evaluasi geometri | Chamfer, precision, recall, dan F1 terhadap GT, default nonaktif |
| Monitoring | RSS proses dan anak proses serta telemetry NVIDIA |
| Resource guard | Memeriksa RAM/VRAM sebelum mesh dan menunggu sampai batas waktu |
| Failure recovery | Kegagalan scene dicatat dan batch melanjutkan scene berikutnya |
| Hasil | Log mentah, JSON/CSV per scene, dan master CSV atomik |
| Ctrl+C | Membersihkan subprocess aktif dan keluar dengan status 130 tanpa traceback Python panjang |

## 4. Perbaikan beban dan kestabilan

### Output progress real-time

Runner sebelumnya membaca output per baris. `tqdm` memperbarui satu bar dengan
karakter `\r`, sehingga output tertahan sampai muncul newline atau proses
berhenti. Runner sekarang membaca potongan byte yang tersedia dan langsung
melakukan flush. Progress pipeline akan terlihat seperti saat menjalankan
`train.py` secara langsung.

### Memori mesh

Mesh sebelumnya menyimpan color dan depth seluruh kamera dalam dua list sebelum
integrasi. Sekarang setiap kamera langsung melewati urutan render, transfer,
integrasi TSDF, dan pelepasan buffer. Pemakaian RAM tidak lagi bertambah karena
seluruh hasil render kamera dikumpulkan sekaligus.

Voxel grid, model Gaussian, dan object kamera tetap membutuhkan RAM/VRAM. Karena
itu perubahan ini mengurangi lonjakan memori, tetapi tidak menjamin mesh bebas
OOM pada semua ukuran dataset.

### Cluster mesh

`num_cluster: 8` sebelumnya dapat menghasilkan `IndexError` jika mesh hanya
memiliki satu sampai tujuh connected components. Batas cluster sekarang
disesuaikan dengan jumlah cluster yang benar-benar ditemukan.

### Monitoring

Konfigurasi utama mengambil sampel setiap lima detik, sebelumnya setiap satu
detik. Ini mengurangi jumlah proses `nvidia-smi` dan penulisan CSV selama
training panjang.

### Kemandirian tahap

Metrik citra dijalankan sebelum mesh sehingga hasil kualitas gambar sudah
tersimpan jika mesh mengalami OOM. Jika worker metrik citra gagal, pipeline tetap
mencoba menyelesaikan mesh dan mencatat kegagalan metrik.

## 5. Identitas eksperimen dan hasil

Experiment ID dihitung dari nama scene serta konfigurasi dataset, COLMAP,
training, rendering, metrik citra, mesh, dan evaluasi geometri. Perubahan lokasi
dataset atau sakelar metrik citra menghasilkan ID berbeda.

Identitas ini belum menghitung hash seluruh isi gambar/COLMAP. Jika isi dataset
diubah tanpa mengganti path atau nama scene, gunakan `--force` atau lokasi
output/config eksperimen yang berbeda.

Master CSV menggunakan satu baris per experiment ID. Percobaan lama tidak
dihapus dari disk; attempt baru diletakkan dalam `runs/<timestamp>_<id>/` jika
folder output scene sudah berisi file.

## 6. Komponen utama repository

| Lokasi | Fungsi |
|---|---|
| `train.py` | Training Gaussian RaDe-GS |
| `render.py` | Render model dan pasangan ground truth |
| `mesh_extract.py` | Ekstraksi mesh TSDF yang dipakai pipeline |
| `arguments/` | Parameter model, optimisasi, dan pipeline |
| `scene/` | Pembacaan dataset/kamera dan representasi Gaussian |
| `gaussian_renderer/` | Rendering Gaussian dan koneksi viewer |
| `utils/` | Kamera, loss, geometri, profil training, dan utilitas lain |
| `lpipsPyTorch/` | Implementasi LPIPS |
| `research_pipeline/` | Orkestrasi, monitoring, evaluasi, dan hasil penelitian |
| `configs/` | Konfigurasi YAML pipeline |
| `tests/` | Pengujian CPU untuk pipeline dan profiler |
| `research_results/` | Lokasi master CSV relatif bawaan |
| `docs/` | Catatan metodologi tambahan |
| `dtu_eval/`, `eval_tnt/` | Toolbox evaluasi benchmark |
| `SIBR_viewers/` | Source viewer interaktif |
| `gaussian-splatting/` | Repository pendamping; bukan entry point Research Pipeline |

File pipeline utama:

| File | Tanggung jawab |
|---|---|
| `run_all.py` | CLI, pemilihan scene, list, dry-run, resume, dan force |
| `config.py` | Default, validasi YAML, serta resolusi path |
| `scene_discovery.py` | Discovery dan validasi struktur COLMAP |
| `colmap_preprocessor.py` | Persiapan scene mentah dan publikasi hasil COLMAP secara transaksional |
| `commands.py` | Penyusunan command setiap tahap |
| `runner.py` | Eksekusi serial, live output, cleanup, dan failure handling |
| `process_monitor.py` | Sampling RSS dan telemetry GPU ke CSV |
| `resource_guard.py` | Pemeriksaan memori sebelum mesh |
| `image_evaluator.py` | SSIM, PSNR, dan LPIPS |
| `mesh_analyzer.py` | Deteksi OOM, batas cluster, dan metadata mesh |
| `geometry_evaluator.py` | Evaluasi mesh terhadap ground truth |
| `results.py` | Experiment ID, summary, dan master CSV |

## 7. Batasan yang masih berlaku

- `--resume` melewati eksperimen yang sudah sukses sepenuhnya. Eksperimen gagal
  dijalankan kembali dari titik awal yang dipilih; pipeline belum melanjutkan
  otomatis dari tahap terakhir atau checkpoint terakhir. Output COLMAP yang
  sudah lengkap akan dilewati.
- Ctrl+C dapat meninggalkan output dan log parsial. Summary/master CSV untuk
  scene aktif belum tentu ditulis karena scene belum selesai.
- Resource guard mengurangi risiko OOM tetapi tidak dapat memprediksi seluruh
  kebutuhan voxel grid atau model.
- Validasi discovery memeriksa keberadaan file, bukan kebenaran isi COLMAP,
  kecocokan semua kamera, atau gambar yang rusak.
- Tidak ada lock global untuk mencegah dua terminal menulis output/master CSV
  yang sama. Jalankan satu pipeline untuk satu output pada satu waktu.
- Tanpa `--eval`, evaluasi citra biasanya memakai view training karena scene
  COLMAP tidak memiliki split test.
- Resolusi default training dibatasi ke lebar 1024, sedangkan mesh memakai
  `-r 2`; pada foto asli beresolusi tinggi tahap mesh masih dapat lebih berat.

## 8. Berkas pengujian

| Berkas | Fokus |
|---|---|
| `test_research_pipeline.py` | Config, discovery, dry-run, dan Ctrl+C |
| `test_colmap_preprocessor.py` | Transaksi output COLMAP, kegagalan, dan skip scene lengkap |
| `test_phase2.py` | Live progress, raw log, cleanup, dan failure recovery |
| `test_phase3.py` | Parsing training dan monitoring |
| `test_phase4.py` | Pemilihan render dan pasangan ground truth |
| `test_phase5.py` | Resource guard, OOM, cluster, dan metadata mesh |
| `test_phase6.py` | Chamfer dan command evaluasi geometri |
| `test_phase7.py` | Experiment ID, CSV, resume, dan force |
| `test_pipeline_flow.py` | Urutan tahap dan kemandirian metrik/mesh |
| `test_training_profiler.py` | Instrumentasi training dan telemetry |

Tes CPU membuktikan alur controller dan perhitungan unit yang diuji. Tes tersebut
tidak menggantikan validasi hasil visual dan penggunaan memori pada GPU server.
