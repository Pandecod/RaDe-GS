# Laporan Kondisi RaDe-GS dan Panduan Penggunaan

Tanggal pemeriksaan terakhir: 21 Oktober 2026 (update urutan pipeline Visual 2D Metrics).

Laporan ini berdasarkan pembacaan kode repository yang sedang dibuka dan
pengujian ulang. Pembuatan laporan tidak mengubah kode, konfigurasi, atau
algoritma RaDe-GS dan tidak menjalankan training GPU.

## 1. Kondisi saat ini

Repository memiliki RaDe-GS beserta Research Pipeline untuk menjalankan banyak
scene COLMAP satu per satu. Pipeline mengatur eksekusi, logging, monitoring,
evaluasi, dan pencatatan hasil di atas program RaDe-GS yang sudah ada.

**Kode orchestration sudah tersedia dan 25 test CPU-only lulus. Training,
rendering, metrik citra, dan mesh dengan GPU/dataset nyata masih harus divalidasi
di environment WSL/NVIDIA.** Test yang lulus tidak berarti semua operasi GPU
sudah berhasil dijalankan.

Hasil pemeriksaan environment Python Windows saat laporan dibuat:

| Komponen | Hasil pemeriksaan |
|---|---|
| Python | 3.12.0 |
| PyYAML, psutil, NumPy | Ditemukan oleh Python aktif |
| PyTorch, SciPy, trimesh | Tidak ditemukan oleh Python aktif |
| Dataset default | `/home/pande/dataset_togog_bali`; tidak tersedia melalui Python Windows ini |
| Perubahan tracked terhadap Git HEAD | Tidak ada pada saat pemeriksaan |
| Pipeline, config, test, dokumentasi baru | Ada di disk, masih untracked di Git |
| Submodule simple-knn | Memiliki file build lokal; sudah ada sebelum laporan ini |

Environment conda di server WSL dapat berbeda. Status dependency Windows di atas
bukan hasil pemeriksaan environment server.

## 2. Apa yang sudah bisa dilakukan

| Kemampuan | Perilaku implementasi sekarang |
|---|---|
| Discovery scene | Semua direktori anak langsung dari `dataset.root`, diurutkan berdasarkan nama |
| Identitas objek | Nama folder menjadi `scene_name` dan `object_name` |
| Validasi dataset | Memeriksa `images/`, `sparse/0/`, pasangan cameras/images bin atau txt, serta points3D bin atau txt |
| List scene | `--list-scenes` menampilkan scene dan memberi keterangan bila struktur tidak valid |
| Dry-run | Memvalidasi config/dataset dan mencetak command tanpa training, CUDA, atau hasil eksperimen palsu |
| Training | Memanggil `train.py -s <dataset> -m <output>`; tambahan hanya dari config |
| Rendering | Memanggil `render.py` dan menghitung jumlah citra, waktu per citra, serta FPS rata-rata proses |
| Evaluasi citra | SSIM dan PSNR dari utilitas repository, LPIPS VGG dari implementasi repository |
| Monitoring | Sampling RAM proses beserta anak proses dan VRAM proses melalui nvidia-smi |
| Parsing training | Mengambil jumlah kamera/titik, extent, iterasi, loss, L1, PSNR, dan kecepatan jika muncul di log |
| Resource guard | Cooldown dan pemeriksaan RAM/VRAM sebelum mesh; bisa menunggu hingga batas waktu |
| Ekstraksi mesh | Memanggil `mesh_extract.py` dengan parameter tetap yang diverifikasi |
| Analisis mesh | Mengambil jumlah vertex, face, dan ukuran file `recon_post.ply` |
| Geometri opsional | Membandingkan mesh dengan GT bernama scene menggunakan Chamfer, precision, recall, F1 |
| Logging | Stdout dan stderr subprocess disimpan ke file log mentah |
| Hasil | JSON per scene, training CSV, dan master CSV yang diganti secara atomik |
| Failure recovery | Kegagalan stage yang ditangani dicatat; batch lanjut ke scene berikutnya |
| Resume/force | Bisa melewati eksperimen sukses atau menjalankan ulang tanpa menghapus output sebelumnya |

Validasi dataset saat discovery memeriksa struktur/file, bukan membuktikan isi
COLMAP benar, gambar lengkap, atau semua kamera cocok. Isi rusak masih dapat
gagal ketika dibaca oleh RaDe-GS.

Alur default satu scene (urutan terbaru):

```text
Validasi dataset
  -> Training RaDe-GS
  -> Rendering
  -> Resource guard
  -> Ekstraksi dan analisis mesh
  -> SSIM / PSNR / LPIPS  (setelah mesh selesai)
  -> Evaluasi geometri jika diaktifkan
  -> Summary JSON dan master CSV
  -> Scene berikutnya
```

Satu kali menjalankan pipeline memproses seluruh scene yang ditemukan. Setelah
semua selesai, program berhenti; tidak memantau folder untuk dataset baru.

## 3. Mengatur dataset, termasuk folder Siwa

Config yang perlu diedit adalah `configs/togog_bali.yaml`.

Jika struktur Anda seperti berikut:

```text
G:/Shared drives/Dataset TogogBali/Dataset Togog Bali/Togog Bali/Dewa/
  Siwa/
    images/
    sparse/0/cameras.bin
    sparse/0/images.bin
    sparse/0/points3D.bin
  Ganesha/
    images/
    sparse/0/...
```

Maka `dataset.root` adalah folder **Dewa**, bukan Siwa. Scene yang ditemukan
adalah Siwa dan Ganesha. Discovery tidak mencari secara rekursif; folder kategori
di tingkat lain tidak otomatis dijelajahi sebagai kumpulan scene.

Contoh untuk Python Windows:

```yaml
dataset:
  root: 'G:/Shared drives/Dataset TogogBali/Dataset Togog Bali/Togog Bali/Dewa'
```

Contoh untuk Python dalam WSL, jika G: memang ter-mount di `/mnt/g`:

```yaml
dataset:
  root: '/mnt/g/Shared drives/Dataset TogogBali/Dataset Togog Bali/Togog Bali/Dewa'
output:
  root: /home/pande/radegs_results/togog_bali
results:
  csv: research_results/experiment_results.csv
```

Path WSL harus dapat diakses dari WSL. Google Drive virtual pada Windows tidak
otomatis menjamin path `/mnt/g` tersedia. Untuk server yang tidak memiliki drive
tersebut, salin dataset COLMAP ke direktori Linux yang bisa diakses dan arahkan
config ke sana. File cloud harus tersedia untuk dibaca oleh proses training.

Format COLMAP text juga diterima: pasangan `cameras.txt` dan `images.txt`, serta
`points3D.txt`. Nama folder scene digunakan persis, termasuk huruf besar/kecil.

## 4. Cara menjalankan

### A. Persiapan environment

Gunakan WSL/Linux dengan NVIDIA dan environment RaDe-GS yang sudah terpasang.
Panduan pemasangan model utama ada di `README.md`: PyTorch/torchvision, dependency
utama, rasterizer, simple-knn, warp-patch-ncc, dan fused-ssim.

Dependency tetra_triangulation/CGAL diperlukan untuk metode tetrahedra terpisah;
pipeline default menggunakan `mesh_extract.py`.

Aktifkan environment dan masuk ke folder yang memiliki `train.py`:

```bash
conda activate radegs
cd /home/pande/RaDe-GS
python -m pip install -r requirements-research.txt
```

`requirements-research.txt` hanya menambah PyYAML, psutil, dan SciPy. Itu tidak
menggantikan instalasi utama RaDe-GS; trimesh berasal dari `requirements.txt`.

### B. Periksa nama scene

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --list-scenes
```

Pastikan daftar berisi objek yang diinginkan, bukan folder kategori seperti
`Dewa` atau folder output. Jika ada keterangan INVALID, periksa struktur COLMAP.

### C. Periksa command tanpa training

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --dry-run
```

Periksa path TRAIN, RENDER, IMAGE METRICS, dan MESH. Parameter mesh harus tetap:

```bash
python mesh_extract.py -m <output> -r 2 --num_cluster 8 --depth_ratio 0.5
```

### D. Jalankan seluruh scene

Perintah berikut benar-benar memulai training dan stage lain yang diaktifkan:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml
```

Tidak perlu memanggil train/render/mesh sendiri untuk setiap objek. Setiap scene
diproses sampai selesai atau gagal sebelum pipeline pindah ke scene berikutnya.

### E. Hanya jalankan Siwa

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --scene Siwa --dry-run
python research_pipeline/run_all.py --config configs/togog_bali.yaml --scene Siwa
```

### F. Jalankan lagi setelah terputus atau ada scene baru

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --resume
```

Resume hanya melewati experiment ID yang sama dengan status success di master
CSV. Scene gagal atau belum tercatat akan dijalankan dari training lagi.
**Resume pipeline tidak otomatis melanjutkan checkpoint pada iterasi terakhir.**
Pemulihan checkpoint model adalah fitur `train.py` yang terpisah.

### G. Jalankan ulang meski sebelumnya sukses

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml --force
```

Output lama tidak dihapus. Jika folder scene sudah berisi file, attempt baru
ditempatkan di `runs/<timestamp>_<id>/`. Perintah biasa tanpa resume juga
menjalankan ulang; force menegaskan maksud tersebut. Resume dan force tidak boleh
digabung pada CLI.

## 5. Fungsi config

| Bagian | Fungsi dan default |
|---|---|
| `project` | Label proyek/dataset; tidak menentukan nama folder scene |
| `dataset.root` | Direktori induk scene COLMAP |
| `output.root` | Direktori hasil model dan research per scene |
| `results.csv` | Master tabel hasil; path relatif dihitung dari root repository |
| `training.extra_args` | Tambahan CLI training; default kosong |
| `rendering.enabled` | Tidak berpengaruh pada pipeline run (rendering selalu dijalankan); field tetap ada di config untuk kompatibilitas |
| `rendering.extra_args` | Tambahan CLI rendering |
| `mesh.enabled` | Aktif secara default |
| `mesh.resolution/num_cluster/depth_ratio` | Dikunci pada 2 / 8 / 0.5 oleh validasi |
| `mesh.extra_args` | Tambahan CLI mesh; tidak boleh mengganti parameter yang dikunci |
| `resource_monitor.interval_seconds` | Interval polling/sampling default 1 detik; query telemetry menambah waktu aktual |
| `resource_guard` | Aktif; cooldown 5 detik, RAM tersedia minimal 4 GiB, VRAM minimal 1024 MiB, cek ulang 10 detik, maksimum menunggu 300 detik |
| `geometry_evaluation` | Default tidak aktif; GT root null, 100000 titik, threshold null, seed 0 |

Tidak ada penambahan `--eval` diam-diam pada training. Dengan config default,
semua kamera COLMAP dipakai untuk training dan biasanya tidak ada kamera test.
Evaluator memilih citra test jika tersedia; jika tidak, memakai citra train.
Karena itu nilai kualitas default harus dibaca sebagai kualitas pada view train,
bukan bukti kualitas pada sudut kamera yang belum pernah dilihat.

Jika memang ingin split evaluasi, konfigurasi yang tersedia adalah
`training.extra_args: ['--eval']`. Perubahan tersebut merupakan perubahan
konfigurasi eksperimen dan menghasilkan experiment ID berbeda.

## 6. Fungsi folder repository

| Folder | Kegunaan |
|---|---|
| `arguments/` | Definisi parameter model, training, pipeline, serta pembacaan cfg_args |
| `scene/` | Pembacaan dataset/kamera, representasi Gaussian, dan appearance network |
| `gaussian_renderer/` | Rendering/integrasi Gaussian dan kode koneksi viewer |
| `utils/` | Perhitungan kamera, loss/metric, geometri, spherical harmonics, dan utilitas lainnya |
| `submodules/` | Ekstensi native/CUDA yang diperlukan model atau metode opsional |
| `submodules/diff-gaussian-rasterization/` | Rasterizer CUDA |
| `submodules/simple-knn/` | Ekstensi nearest-neighbor untuk Gaussian |
| `submodules/warp-patch-ncc/` | Ekstensi patch NCC untuk regularisasi multiview |
| `submodules/tetra_triangulation/` | Triangulasi untuk ekstraksi tetrahedra alternatif |
| `lpipsPyTorch/` | Implementasi metrik perceptual LPIPS |
| `research_pipeline/` | Otomatisasi penelitian yang ditambahkan |
| `configs/` | Konfigurasi pipeline yang dapat diubah pengguna |
| `research_results/` | Master CSV default; bukan model Gaussian |
| `tests/` | Pengujian kode CPU-only; bukan kumpulan citra uji penelitian |
| `docs/` | Catatan metodologi penelitian |
| `output/` | Output lokal yang sudah ada (`togog_bg`, `togog_bg_fix`); config pipeline saat ini menunjuk lokasi output lain |
| `scripts/` | Contoh batch shell khusus DTU/TnT, bukan runner Togog Bali |
| `dtu_eval/` | Toolbox evaluasi benchmark DTU |
| `eval_tnt/` | Toolbox evaluasi benchmark Tanks and Temples |
| `SIBR_viewers/` | Source viewer interaktif; terpisah dari pipeline batch |
| `gaussian-splatting/` | Repository pendamping yang sudah ada; bukan entry point Research Pipeline ini |
| `assets/` | Aset dokumentasi seperti gambar |
| `.git/` | Riwayat dan metadata Git |

## 7. Fungsi file utama

| File | Fungsi |
|---|---|
| `train.py` | Training Gaussian RaDe-GS; menyimpan model, konfigurasi, dan checkpoint sesuai parameter |
| `render.py` | Merender model yang sudah dilatih ke citra dan menyimpan pasangan GT |
| `mesh_extract.py` | Metode mesh yang dipakai pipeline; fusi TSDF Open3D dan post-processing |
| `mesh_extract_tetrahedra.py` | Alternatif marching tetrahedra; bukan metode mesh default pipeline; menghasilkan recon.ply |
| `mesh_extract_tnt.py` | Ekstraksi khusus workflow TnT, bukan command penelitian Togog Bali |
| `metric.py` | Evaluator citra asli; checkout ini membaca folder train dan menangkap error; pipeline menggunakan fungsi metric melalui worker sendiri |
| `geometry_metric.py` | Evaluasi depth/normal terhadap GT per kamera; berbeda dari evaluasi mesh Chamfer pipeline |
| `evaluate_dtu_mesh.py` | Evaluasi mesh untuk benchmark DTU |
| `requirements.txt` | Dependency Python utama RaDe-GS |
| `requirements-research.txt` | Dependency tambahan orchestration |
| `README.md` | Panduan asli model, instalasi, benchmark, viewer |
| `RESEARCH_PIPELINE.md` | Panduan pipeline dalam bahasa Inggris |
| `PIPELINE_SPEC.md` | Spesifikasi kebutuhan; bukan program yang dijalankan |
| `hasil_analisa.md` | Catatan analisis sebelumnya; bukan bukti hasil eksperimen baru |
| `LICENSE.md` | Ketentuan lisensi penggunaan kode |
| `.gitignore` | Pengaturan file yang tidak dimasukkan Git |
| `.gitmodules` | Daftar submodule repository |
| `LAPORAN_RADEGS_DAN_PANDUAN.md` | Laporan dan panduan Indonesia ini (diperbarui 21 Oktober 2026) |

> **File dan folder yang ditambahkan oleh Research Pipeline (tidak ada di repository asli RaDe-GS):**
>
> | File / Folder baru | Keterangan |
> |---|---|
> | `research_pipeline/` | Seluruh folder orchestration pipeline baru |
> | `research_pipeline/__init__.py` | Penanda package |
> | `research_pipeline/run_all.py` | CLI utama pipeline |
> | `research_pipeline/config.py` | Validasi YAML dan default konfigurasi |
> | `research_pipeline/scene_discovery.py` | Penemuan scene COLMAP |
> | `research_pipeline/commands.py` | Penyusunan argument subprocess |
> | `research_pipeline/runner.py` | Eksekutor stage sinkron **[diperbarui 21 Okt 2026]** |
> | `research_pipeline/process_monitor.py` | Sampling RAM/VRAM |
> | `research_pipeline/gpu_monitor.py` | Query nvidia-smi |
> | `research_pipeline/resource_guard.py` | Pemeriksaan memori sebelum mesh |
> | `research_pipeline/training_parser.py` | Parser log training ke CSV |
> | `research_pipeline/image_evaluator.py` | Worker SSIM/PSNR/LPIPS |
> | `research_pipeline/mesh_analyzer.py` | Deteksi OOM dan metadata mesh |
> | `research_pipeline/geometry_evaluator.py` | Evaluasi Chamfer/F1 mesh terhadap GT |
> | `research_pipeline/results.py` | ID eksperimen, summary, dan master CSV |
> | `research_pipeline/utils.py` | Baca/tulis JSON atomik |
> | `configs/` | Folder konfigurasi pipeline (baru) |
> | `configs/togog_bali.yaml` | Config utama proyek Togog Bali |
> | `tests/` | Folder test CPU-only (baru) |
> | `tests/test_research_pipeline.py` | Test dasar pipeline |
> | `tests/test_phase2.py` s.d. `test_phase7.py` | Test per fase pipeline |
> | `tests/test_pipeline_flow.py` | Test urutan stage penuh **[diperbarui 21 Okt 2026]** |
> | `research_results/` | Folder master CSV default (dibuat saat run pertama) |
> | `docs/` | Catatan metodologi penelitian |
> | `requirements-research.txt` | Dependency tambahan: PyYAML, psutil, SciPy |
> | `RESEARCH_PIPELINE.md` | Panduan pipeline bahasa Inggris |
> | `PIPELINE_SPEC.md` | Spesifikasi kebutuhan pipeline |
> | `hasil_analisa.md` | Catatan analisis repository |
> | `LAPORAN_RADEGS_DAN_PANDUAN.md` | Laporan dan panduan ini |

README menyebut `metrics.py` pada salah satu contoh, tetapi nama file yang benar
di checkout ini adalah `metric.py`. Untuk workflow otomatis, gunakan
`research_pipeline/run_all.py`.

## 8. Fungsi setiap file pipeline

| File dalam research_pipeline/ | Fungsi |
|---|---|
| `__init__.py` | Penanda package Python |
| `run_all.py` | CLI utama: config, scene, list, dry-run, resume, force |
| `config.py` | Default/validasi YAML, resolusi path, verifikasi extra_args terhadap CLI repository |
| `scene_discovery.py` | Menemukan scene dan memeriksa struktur COLMAP |
| `commands.py` | Menyusun argument list subprocess dan command untuk dry-run |
| `runner.py` | Menjalankan stage secara sinkron; urutan: training → rendering → mesh → image_metrics → geometry; menangani cleanup/failure dan memilih folder attempt **[diperbarui 21 Okt 2026: image_metrics dipindah ke setelah mesh]** |
| `process_monitor.py` | Mengambil sampel RAM/RSS dan VRAM serta menghitung peak/rata-rata |
| `gpu_monitor.py` | Query nvidia-smi dan dukungan path executable WSL |
| `resource_guard.py` | Memeriksa ketersediaan memori sebelum mesh |
| `training_parser.py` | Mengubah log mentah training menjadi CSV dan metrik final jika tersedia |
| `image_evaluator.py` | Worker SSIM/PSNR/LPIPS, pemilihan split dan pasangan citra |
| `mesh_analyzer.py` | Klasifikasi tanda OOM dan metadata mesh final |
| `geometry_evaluator.py` | Sampling geometri dan evaluasi mesh/point cloud terhadap GT |
| `results.py` | ID SHA-256, metadata environment, summary, dan read/update master CSV |
| `utils.py` | Baca JSON dan atomic write JSON/CSV |

## 9. Fungsi file test

| File dalam tests/ | Apa yang diuji |
|---|---|
| `test_research_pipeline.py` | Discovery, validasi config, command tetap, list/dry-run tanpa proses atau hasil |
| `test_phase2.py` | Raw log, exit code, urutan training fixture, cleanup saat interupsi |
| `test_phase3.py` | Parsing progress/evaluasi, null metric, monitoring failure dan startup CUDA tertunda |
| `test_phase4.py` | Pemilihan split, pasangan GT, jumlah citra, penanganan citra kosong |
| `test_phase5.py` | Guard memori, OOM, metadata mesh dengan pembaca simulasi |
| `test_phase6.py` | Definisi Chamfer/threshold dan command geometri |
| `test_phase7.py` | Experiment ID, summary, CSV, resume/force, failure recovery |
| `test_pipeline_flow.py` | Urutan stage lengkap (training→rendering→mesh→image_metrics), mesh OOM tanpa retry, geometri opsional **[diperbarui 21 Okt 2026]** |

Untuk menjalankan ulang test:

```bash
python -B -m unittest discover -s tests -v
```

Test tidak wajib dijalankan setiap kali menambah dataset. Mereka membantu
memeriksa perubahan kode pipeline dan tidak mengukur kualitas rekonstruksi nyata.

## 10. Hasil yang harus diperiksa setelah run

| Lokasi dalam output scene | Arti |
|---|---|
| `cfg_args` | Parameter model yang dibaca kembali oleh render/mesh |
| `input.ply` | Salinan point cloud input |
| `cameras.json`, `multi_view.json` | Informasi kamera dan relasi view |
| `point_cloud/iteration_<N>/point_cloud.ply` | Model Gaussian tersimpan, bukan mesh permukaan final |
| `chkpnt<N>.pth` | Checkpoint training jika iterasi itu dikonfigurasi untuk checkpoint |
| `train/ours_<N>/renders/` | Render kamera training |
| `train/ours_<N>/gt/` | Citra acuan kamera training |
| `test/ours_<N>/renders/`, `gt/` | Render/GT kamera evaluasi jika tersedia |
| `recon_raw.ply` | Mesh mentah metode default |
| `recon_post.ply` | Mesh final setelah post-processing; ini yang dianalisis pipeline |
| `research/logs/*.log` | Stdout/stderr training, rendering, evaluator, mesh, geometri yang sudah dijalankan |
| `research/training_metrics.csv` | Metrik yang berhasil diparse dari log training |
| `research/image_metrics.json` | SSIM, PSNR, LPIPS, split, jumlah citra evaluasi |
| `research/mesh_metrics.json` | Waktu/resource dan jumlah vertex/face/ukuran mesh sukses |
| `research/geometry_metrics.json` | Metrik terhadap GT jika stage geometri diaktifkan dan berhasil |
| `research/summary.json` | Identitas, status, failed_stage, error, metrik, warning, command, metadata environment |
| `runs/<timestamp>_<id>/` | Attempt berikutnya bila output scene sudah terisi |

Master CSV berada di lokasi `results.csv`, bukan harus di dalam output scene.
CSV dapat dibuka dengan spreadsheet. Satu experiment ID memiliki satu baris;
rerun ID yang sama memperbarui barisnya. Detail attempt sebelumnya tetap berada
di folder masing-masing. JSON/CSV stage yang belum dijalankan atau gagal tidak
harus tersedia; mulai pemeriksaan dari summary dan raw log.

Ringkasan durasi rendering mencakup proses render, loading dan penulisan citra.
FPS tersebut bukan pengukuran kecepatan viewer interaktif atau waktu kernel saja.
VRAM null berarti telemetry tidak tersedia, bukan pemakaian memori nol.

## 11. Evaluasi geometri opsional

Contoh konfigurasi (angka threshold hanya contoh, harus dipilih sesuai satuan
dan protokol penelitian):

```yaml
geometry_evaluation:
  enabled: true
  gt_root: /home/pande/geometry_gt
  sample_points: 100000
  distance_threshold: 0.01
  seed: 0
```

GT scene Siwa harus bernama `Siwa.ply`. Mesh dan GT harus sudah searah, sejajar,
dan memiliki skala/satuan yang sama. Pipeline tidak melakukan registrasi/ICP
atau penyamaan skala otomatis.

Chamfer adalah mean jarak Euclidean reconstruction-ke-GT ditambah mean
GT-ke-reconstruction, bukan squared distance. Precision/recall memakai strict
`distance < threshold`. Jika threshold null, Chamfer dihitung, precision/recall/F1
null dengan warning. Bila geometri diaktifkan tetapi GT hilang, scene gagal di
geometry_evaluation. Jika geometri tidak dibutuhkan, biarkan enabled false.

## 12. Batasan dan interpretasi yang perlu diketahui

- Resource guard mengurangi risiko memulai mesh saat memori rendah; tidak
  menjamin mesh bebas OOM setelah proses dimulai. Tidak ada perubahan parameter
  otomatis atau retry mesh dengan resolusi lain.
- Resume tidak memeriksa ulang keberadaan model lama dan tidak melakukan resume
  checkpoint otomatis. Ia membaca status success dan experiment ID dari CSV.
- Experiment ID berasal dari nama scene dan config training/render/mesh/geometri;
  tidak mencakup isi dataset atau hash kode. Mengganti isi dataset dengan nama
  scene dan config yang sama tidak menghasilkan ID baru. Gunakan force atau
  pisahkan pencatatan eksperimen ketika memang ingin mengukur data berbeda.
- Serial berlaku dalam satu proses pipeline. Belum ada pengunci antar dua
  terminal yang menjalankan batch bersamaan; jalankan satu pipeline pada satu
  waktu untuk dataset/output/master CSV yang sama.
- Saat Ctrl+C, child aktif dibersihkan, tetapi scene yang sedang berlangsung
  belum tentu memiliki baris CSV/summary baru karena penulisan dilakukan setelah
  execute_scene selesai. Log/output parsial bisa tetap tersimpan.
- Terminal tidak menampilkan seluruh progress training secara langsung karena
  stdout/stderr diarahkan ke file. Untuk melihat log pada WSL, gunakan
  `tail -f <scene_output>/research/logs/training.log` di terminal lain.
- Nilai log yang tidak ditemukan tetap null. Parsing training dilakukan setelah
  subprocess training selesai; raw log tetap sumber utama selama proses berjalan.
- Pemindahan/rename repository didukung oleh resolusi lokasi file. Pemindahan
  dataset/output lama tidak otomatis memperbarui cfg_args, summary, atau path CSV.
- Karena tidak semua error merupakan StageFailure yang ditangani, kegagalan
  tidak terduga atau proses controller yang dihentikan paksa tetap dapat
  menghentikan batch. Failure recovery yang teruji menggunakan skenario test.

## 13. Hasil pengecekan dan langkah berikutnya

Pengujian ulang pada saat laporan dibuat:

```text
python -B -m unittest discover -s tests -v
Ran 25 tests
OK
```

CLI dengan config default dijalankan lagi untuk list-scenes dan dry-run:

```text
Dataset root does not exist: C:\home\pande\dataset_togog_bali
```

Itu sesuai kondisi Python Windows dan path Linux yang belum diubah; bukan hasil
training gagal. Pada validasi sebelumnya, fixture sementara tiga scene bin/text
berhasil ditemukan secara alphabetis dan dry-run menghasilkan command tetap:

```text
Discovered 3 scenes:
1. togog_arjuna
2. togog_bima
3. togog_ganesha
...
python mesh_extract.py -m <output> -r 2 --num_cluster 8 --depth_ratio 0.5
Dry-run only: no subprocesses, CUDA, output files, or results written.
LIST_EXIT=0 DRY_EXIT=0
OUTPUT_CREATED=False
```

Sebelum penelitian nyata: sesuaikan path config, periksa environment conda WSL,
jalankan list-scenes dan dry-run di WSL, lalu lakukan validasi satu scene nyata
ketika memang siap memulai training. Periksa telemetry NVIDIA, hasil metrik
citra, mesh final, kebutuhan memori, serta GT jika dipakai. Setelah validasi itu
berhasil, gunakan batch untuk seluruh scene. Tidak ada training nyata dijalankan
untuk menyusun laporan ini.

## 14. Perubahan Pipeline (21 Oktober 2026)

Urutan stage pipeline diubah agar Visual 2D Metrics (PSNR, SSIM, LPIPS)
diproses **setelah** mesh extraction selesai, bukan sebelumnya.

**Urutan stage baru:**

```text
1. Training          -> latih model Gaussian Splatting
2. Rendering         -> render gambar dari model
3. Mesh Extraction   -> ekstrak mesh 3D (recon_post.ply)
4. Visual 2D Metrics -> hitung & simpan PSNR / SSIM / LPIPS
5. Geometry 3D       -> SKIP by default (jalankan nanti secara manual)
```

**File yang diubah:**

| File | Perubahan |
|---|---|
| `research_pipeline/runner.py` | Hapus guard `rendering.enabled`; pisah rendering dari image_metrics; pindahkan image_metrics ke setelah mesh |
| `tests/test_pipeline_flow.py` | Update 3 test: urutan stage baru, fixture OOM rendering, assertions geometry test |

**Tidak ada perubahan pada:**
- Environment conda / package Python
- `train.py`, `render.py`, `mesh_extract.py`
- `config.py`, `requirements*.txt`

**Cara menjalankan Geometry 3D nanti (manual):**

Aktifkan di config YAML saat siap:
```yaml
geometry_evaluation:
  enabled: true
  gt_root: /path/to/ground_truth_meshes
```
Atau jalankan langsung per scene:
```bash
python -m research_pipeline.geometry_evaluator \
  --reconstruction /output/scene/recon_post.ply \
  --ground-truth /gt/scene.ply \
  --sample-points 100000 --seed 0 \
  --scene-name nama_scene --output /output/scene
```
