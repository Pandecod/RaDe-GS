# Panduan Penggunaan Pipeline RaDe-GS

Panduan ini berisi langkah operasional Research Pipeline. Kondisi implementasi,
hasil tes, dan batasan teknis tersedia di
[Laporan Kondisi RaDe-GS](LAPORAN_KONDISI_RADEGS.md).

Untuk command langsung tanpa pipeline, lihat
[Perintah manual training dan ekstraksi mesh](#13-perintah-manual-training-dan-ekstraksi-mesh-togog-tualen).

## Mulai dari sini: checklist sebelum training

Tentukan dulu titik awal pipeline:

- Gunakan `--start-from training` (default) jika scene sudah mempunyai
  `images/` dan `sparse/0/` hasil COLMAP.
- Gunakan `--start-from colmap` jika scene baru mempunyai foto mentah dalam
  `input/`. Pipeline akan menyelesaikan COLMAP lalu langsung meneruskan training.

Periksa hal berikut sebelum menjalankan proses panjang:

- Environment `radegs` aktif dan perintah `nvidia-smi` dapat membaca GPU.
- Untuk mode training, setiap scene memiliki `images/` dan `sparse/0/` lengkap.
  Untuk mode COLMAP, setiap scene memiliki foto di dalam `input/`.
- `dataset.root`, `output.root`, dan `results.csv` dalam config mengarah ke lokasi yang benar.
- `--list-scenes` menampilkan scene yang ingin diproses tanpa label `INVALID`.
- `--dry-run` menampilkan command TRAIN, RENDER, IMAGE METRICS, dan MESH yang benar.
- Ruang penyimpanan, RAM, dan VRAM cukup untuk training serta mesh.

Urutan pemeriksaan cepat:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
nvidia-smi
python research_pipeline/run_all.py --config configs/togog_bali.yaml --start-from colmap --list-scenes
python research_pipeline/run_all.py --config configs/togog_bali.yaml --start-from colmap --dry-run
```

Jika dua perintah pipeline terakhir berhasil dan semua path sudah benar, jalankan
satu scene terlebih dahulu:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --start-from colmap \
  --scene togog_tualen
```

Setelah satu scene selesai dan hasilnya dapat dibuka, jalankan seluruh scene.

## 1. Memilih titik awal dan menyiapkan dataset

Setiap folder anak langsung di bawah `dataset.root` dianggap sebagai satu scene.

Untuk memulai dari foto mentah, letakkan gambar pada folder `input/`:

```text
dataset/togog_bali/
  togog_tualen/
    input/
      IMG_0001.jpg
      IMG_0002.jpg
  togog_ganesha/
    input/
      IMG_0001.jpg
      IMG_0002.jpg
```

Jalankan mode ini dengan `--start-from colmap`. Pipeline menjalankan feature
extraction, exhaustive matching, mapper, dan image undistorter. Setelah berhasil,
scene akan mempunyai struktur berikut dan langsung dipakai oleh training:

```text
dataset/togog_bali/
  togog_tualen/
    images/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
  togog_ganesha/
    images/
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt
```

Format bin dan txt diterima. Untuk struktur `Dewa/Siwa/images`, arahkan
`dataset.root` ke folder `Dewa`; nama scene yang dipilih adalah `Siwa`.

Folder kerja sementara COLMAP berada di
`<scene>/.pipeline_colmap/<timestamp_id>/`. Hasil baru diterbitkan ke `images/`
dan `sparse/0/` setelah lengkap. Jika proses gagal atau dihentikan, attempt
tersebut dipertahankan untuk pemeriksaan dan tidak dianggap dataset siap.
Menjalankan mode COLMAP pada scene yang sudah lengkap akan melewati COLMAP dan
langsung meneruskan training.

## 2. Persiapan environment

Gunakan Linux/WSL dengan NVIDIA dan environment RaDe-GS yang sudah terpasang.

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
python -m pip install -r requirements-research.txt
```

`requirements-research.txt` hanya menambah dependency orchestration. Dependency
utama, PyTorch/CUDA, dan ekstensi native tetap mengikuti `README.md`.

## 3. Mengatur konfigurasi

Konfigurasi utama berada di `configs/togog_bali.yaml`.

```yaml
# Konfigurasi penelitian utama: training 1920, 30000 iterasi, allocator GC, mesh TNT.
project:
  name: RaDe-GS
  dataset: Dataset Togog Bali
dataset:
  root: /home/pande/projects/RaDe-GS/dataset/togog_bali
output:
  root: output/togog_bali
results:
  csv: /home/pande/projects/RaDe-GS/research_results/experiment_results.csv
colmap:
  executable: colmap
  camera_model: OPENCV
  use_gpu: true
training:
  extra_args:
    - "--enable_allocator_gc"
    - "--iterations"
    - "30000"
    - "--resolution"
    - "1920"
    - "--profile_training"
    - "--profile_from"
    - "1800"
    - "--profile_until"
    - "30000"
    - "--profile_interval"
    - "100"
    - "--profile_control_samples"
    - "--profile_gpu_interval"
    - "1"
    - "--checkpoint_iterations"
    - "3000"
    - "7000"
    - "15000"
    - "25000"
rendering:
  enabled: true
  extra_args: []
image_metrics:
  enabled: true
mesh:
  enabled: true
  resolution: 2
  num_cluster: 8
  depth_ratio: 0.5
  voxel_size: 0.009
  max_depth: 8.0
  extra_args: []
```

Fungsi pengaturan penting:

| Pengaturan | Fungsi |
|---|---|
| `dataset.root` | Folder induk yang berisi semua scene |
| `output.root` | Folder hasil setiap scene |
| `results.csv` | Master hasil seluruh eksperimen |
| `colmap.executable` | Nama atau path executable COLMAP |
| `colmap.camera_model` | Model kamera untuk feature extractor, default `OPENCV` |
| `colmap.use_gpu` | Menggunakan GPU untuk ekstraksi dan pencocokan fitur |
| `training.extra_args` | Argumen tambahan untuk `train.py` |
| `rendering.enabled` | Menjalankan atau melewati rendering |
| `rendering.extra_args` | Argumen tambahan untuk `render.py` |
| `image_metrics.enabled` | Mengaktifkan SSIM, PSNR, dan LPIPS |
| `mesh.enabled` | Mengaktifkan ekstraksi mesh |
| `resource_monitor.interval_seconds` | Interval telemetry; konfigurasi utama memakai 5 detik |
| `geometry_evaluation.enabled` | Mengaktifkan evaluasi terhadap GT mesh |

Parameter mesh `resolution`, `num_cluster`, dan `depth_ratio` dikunci pada
`2`, `8`, dan `0.5` oleh validasi konfigurasi.

### 3.1 Konfigurasi penelitian 1920 dan konfigurasi percobaan lama

Mulai 30 Oktober 2026, gunakan configs/togog_bali.yaml untuk run penelitian baru.
Konfigurasi utama sekarang memakai lebar training 1920 piksel sesuai pilihan
pengguna. Pengaturan lainnya mengikuti percobaan 1600: allocator GC aktif,
30.000 iterasi, profiling, dan mesh TNT. Percobaan Punakawan 1600 telah berhasil;
run GPU penuh dengan 1920 belum diverifikasi dari komputer lokal ini.

| Penggunaan | File konfigurasi | Root output | CSV utama |
|---|---|---|---|
| Penelitian utama 1920 | configs/togog_bali.yaml | output/togog_bali | research_results/experiment_results.csv |
| Percobaan 1600 terdahulu | configs/togog_bali_1600.yaml | output/togog_bali/resolusi_1600 | research_results/experiment_results_1600.csv |
| Pembanding lama 1024 | configs/togog_bali_full_gc.yaml | output/togog_bali/full_gc | research_results/experiment_results_full_gc.csv |

Hasil penelitian disimpan sebagai <root output>/<nama scene>/, misalnya
output/togog_bali/punakawan/. CSV laporan ringkas gabungan
dibuat di research_results/experiment_results_laporan.csv.
Nama folder scene tetap menjadi nama kelas dalam laporan.

Jalankan di terminal Ubuntu-22.04-E, sebagai pengguna pande. Salin konfigurasi
dan kode pipeline terbaru ke checkout server sebelum memakai command berikut:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene punakawan \
  --start-from training \
  --dry-run
~~~

Pastikan TRAIN menampilkan --resolution 1920, --iterations 30000, dan
--enable_allocator_gc; path hasil harus menunjuk output/togog_bali/punakawan/.
Pastikan MESH memakai mesh_extract_tnt.py dan parameter yang sama seperti
konfigurasi berhasil. Dry-run tidak menjalankan CUDA atau membuat hasil.

Untuk menjalankan scene tersebut:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene punakawan \
  --start-from training
~~~

Ganti punakawan dengan nama folder dataset yang ingin dijalankan.
Untuk menjalankan semua scene secara berurutan, hilangkan --scene:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --start-from training
~~~

Command tanpa --dry-run memulai training baru, dilanjutkan rendering, evaluasi,
mesh, dan laporan. Jika folder output scene sudah berisi hasil, pipeline memakai
runs/<waktu>_<id>/ untuk run berikutnya. CSV utama memperbarui baris dengan
identitas eksperimen yang sama; CSV laporan ringkas mempertahankan folder run
yang berbeda. Jangan menjalankan beberapa pipeline yang menulis CSV yang sama.

Untuk foto berukuran 3862 x 2150 piksel, lebar training 1920 menghasilkan sekitar
1920 x 1069 piksel; foto 16:9 menghasilkan 1920 x 1080 piksel. ukuran aktual mengikuti rasio foto yang dimuat. Pengaturan
mesh -r 2 terpisah dan memakai setengah dimensi gambar input.

Nama folder hasil dan CSV utama mengikuti penamaan awal; resolusi training
sekarang 1920. Penggantian YAML hanya mengubah konfigurasi run berikutnya. Hasil lama tidak
dipindahkan, dihapus, atau diberi label ulang sebagai hasil penelitian baru.
Periksa keberhasilan serta kelengkapan laporan baru sebelum menghapus hasil lama.
Path Linux /home/pande/... mengikuti distro WSL yang sedang dibuka; pastikan
menggunakan Ubuntu-22.04-E bila penyimpanan penelitian berada pada distro itu.

## 4. Pemeriksaan sebelum menjalankan

Tampilkan scene yang ditemukan:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --list-scenes
```

Periksa seluruh command tanpa menjalankan training atau CUDA:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --dry-run
```

Perhatikan pesan `INVALID`. Pastikan path TRAIN, RENDER, IMAGE METRICS, dan MESH
menunjuk scene serta output yang benar.

Untuk foto mentah, tambahkan titik awal COLMAP pada kedua pemeriksaan:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml \
  --start-from colmap --list-scenes
python research_pipeline/run_all.py --config configs/togog_bali.yaml \
  --start-from colmap --dry-run
```

Dry-run akan menampilkan `COLMAP` sebelum `TRAIN`. Mode ini memvalidasi `input/`;
mode default memvalidasi `images/` dan `sparse/0/`.

## 5. Menjalankan pipeline

Dari dataset COLMAP yang sudah jadi (perilaku lama dan default):

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml
```

Perintah eksplisit yang setara:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml \
  --start-from training
```

Dari foto mentah, jalankan COLMAP sampai seluruh tahap selesai:

```bash
python research_pipeline/run_all.py --config configs/togog_bali.yaml \
  --start-from colmap
```

Hanya scene Tualen, dimulai dari COLMAP:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --start-from colmap \
  --scene togog_tualen
```

Nama scene harus sama persis dengan nama folder, termasuk huruf besar dan kecil.

## 6. Progress bar dan penghentian

Progress training sekarang muncul langsung saat pipeline berjalan. Output
`tqdm` diteruskan ke terminal dan sekaligus disimpan dalam
`research/logs/training.log`.

`train.py` memperbarui progress setiap 10 iterasi. Jika satu iterasi memerlukan
sekitar enam detik, angka progress dapat terlihat berubah sekitar sekali per
menit. Ini normal dan bukan tanda terminal macet.

Contoh tampilan:

```text
Training progress: 18%|...| 5360/30000 [3:19:44<40:56:50, 5.98s/it]
```

Tekan Ctrl+C satu kali untuk menghentikan pipeline. Runner membersihkan child
process dan keluar dengan status 130 tanpa traceback Python panjang. Output
parsial tidak otomatis menjadi eksperimen sukses.

## 7. Resume dan force

Lewati eksperimen dengan experiment ID sama yang sudah berstatus sukses:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --resume
```

`--resume` belum melanjutkan training dari iterasi terakhir. Scene yang gagal,
terputus, atau belum tercatat akan dimulai lagi dari titik `--start-from` yang
dipilih. Hasil COLMAP lengkap akan dideteksi dan dilewati, sehingga tidak
dihitung ulang.

Jalankan ulang walaupun sebelumnya sudah sukses:

```bash
python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --force
```

`--resume` dan `--force` tidak dapat dipakai bersamaan. Output lama tidak
dihapus; attempt baru dibuat di `runs/<timestamp>_<id>/`.

## 8. Mengurangi beban pipeline

Jika hanya membutuhkan model dan mesh, matikan rendering dan metrik citra:

```yaml
rendering:
  enabled: false
  extra_args: []
image_metrics:
  enabled: false
```

Jika tetap membutuhkan render tetapi tidak membutuhkan LPIPS/PSNR/SSIM:

```yaml
rendering:
  enabled: true
  extra_args: []
image_metrics:
  enabled: false
```

Untuk menonaktifkan mesh:

```yaml
mesh:
  enabled: false
  resolution: 2
  num_cluster: 8
  depth_ratio: 0.5
  voxel_size: 0.009
  max_depth: 8.0
  extra_args: []
```

Selalu jalankan `--dry-run` setelah mengganti sakelar tahap.

## 9. Split evaluasi citra

Tanpa `--eval`, semua kamera COLMAP biasanya dipakai untuk training dan metrik
citra memakai render view training. Untuk membuat split test:

```yaml
training:
  extra_args: ['--eval']
```

Perubahan ini menghasilkan experiment ID berbeda. Nilai metrik view training
dan view test tidak boleh dibandingkan seolah-olah berasal dari protokol sama.

## 10. Hasil yang perlu diperiksa

```text
output/togog_bali/<scene>/
  cfg_args
  point_cloud/iteration_<N>/point_cloud.ply
  train/ours_<N>/renders/
  train/ours_<N>/gt/
  recon.ply
  recon_post.ply
  research/
    logs/
      colmap.log
    training_metrics.csv
    image_metrics.json
    mesh_metrics.json
    geometry_metrics.json
    summary.json
```

| Hasil | Arti |
|---|---|
| `point_cloud/iteration_<N>/point_cloud.ply` | Model Gaussian pada iterasi tersimpan |
| `recon.ply` | Mesh mentah |
| `recon_post.ply` | Mesh final setelah post-processing |
| `research/logs/*.log` | Output mentah setiap tahap |
| `training_metrics.csv` | Metrik yang diparse dari training |
| `image_metrics.json` | SSIM, PSNR, LPIPS, split, dan jumlah citra |
| `mesh_metrics.json` | Durasi, resource, vertex, face, dan ukuran mesh |
| `summary.json` | Status lengkap satu attempt |
| `experiment_results.csv` | Rekap utama seluruh experiment ID |

Mulai pemeriksaan kegagalan dari `research/summary.json`, lalu buka log untuk
nilai `failed_stage` yang tercatat.

## 11. Evaluasi geometri opsional

Siapkan GT dengan nama yang sama seperti scene, misalnya `togog_tualen.ply`:

```yaml
geometry_evaluation:
  enabled: true
  gt_root: /home/pande/geometry_gt
  sample_points: 100000
  distance_threshold: 0.01
  seed: 0
```

Mesh rekonstruksi dan GT harus memiliki posisi, orientasi, skala, dan satuan yang
sama. Pipeline tidak menjalankan registrasi ICP atau penyamaan skala otomatis.

## 12. Pemecahan masalah

### Progress tidak bergerak

- Tunggu setidaknya 10 iterasi; pembaruan mengikuti interval `train.py`.
- Pastikan server sudah memakai versi terbaru `research_pipeline/runner.py`.
- Periksa `research/logs/training.log` untuk memastikan training masih menulis.
- Periksa proses GPU dengan `nvidia-smi`.

### Dataset tidak ditemukan

Path Linux seperti `/home/pande/...` tidak dapat dipakai oleh Python Windows.
Jalankan pipeline di environment server/WSL atau gunakan path yang dapat diakses
oleh environment aktif.

### COLMAP tidak ditemukan atau gagal

- Pastikan perintah `colmap -h` dapat dijalankan dalam environment aktif, atau
  isi `colmap.executable` dengan path executable yang benar.
- Buka `research/logs/colmap.log` untuk output proses.
- Periksa attempt di `<scene>/.pipeline_colmap/`. Attempt gagal sengaja tidak
  dipublikasikan sebagai `images/` dan `sparse/0/`.
- Jika ada `images/` atau `sparse/` parsial buatan proses lain, pindahkan folder
  tersebut terlebih dahulu. Pipeline menolaknya agar data lama dan baru tidak
  tercampur.

### Mesh kehabisan memori

Periksa `mesh_extraction_resources_*.csv` dan `mesh_extraction.log`. Implementasi
sekarang mengintegrasikan kamera satu per satu, tetapi voxel grid dan kumpulan
kamera masih menggunakan memori. Kurangi beban eksperimen hanya setelah menilai
dampaknya terhadap protokol penelitian.

### LPIPS gagal tetapi mesh dibutuhkan

Pipeline tetap mencoba mesh setelah kegagalan metrik gambar. Jika metrik memang
tidak diperlukan, set `image_metrics.enabled: false` agar worker LPIPS tidak
dijalankan.

### Ingin memeriksa kode tanpa GPU

```bash
python -B -m unittest discover -s tests -v
```

Tes ini memeriksa controller dan utilitas dengan fixture CPU. Tes tidak
menjalankan training RaDe-GS atau membuktikan kualitas hasil GPU.

## Pembaruan 27 Oktober 2026

Pada integrasi 27 Oktober, konfigurasi utama memakai allocator GC,
30.000 iterasi, resolusi 1024, dan profiling. Mulai 30 Oktober 2026,
configs/togog_bali.yaml memakai resolusi 1920 dengan root output/togog_bali
seperti dijelaskan pada bagian 3.1. Pipeline memakai
mesh_extract_tnt.py dengan Open3D 0.19.0, voxel_size 0.009, dan max_depth 8.0.
Pengaturan aktif tercantum dalam YAML Togog Bali dan contoh bagian 3.
Lihat [petunjuk penyalinan dan dry-run](docs/PIPELINE_GC_TNT.md).


## 13. Perintah manual training dan ekstraksi mesh Togog Tualen

Bagian ini memakai pengaturan yang berhasil pada server. Jalankan perintah
di terminal Ubuntu/WSL server. Perintah manual tidak membaca YAML pipeline:
flag GC, resolusi, dan parameter mesh perlu dituliskan pada command.

### A. Persiapan

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -c "import torch, open3d; print('PyTorch:', torch.__version__); print('Open3D:', open3d.__version__)"
python -m pip check
~~~

Environment percobaan berhasil memakai PyTorch 2.6.0+cu124 dan Open3D 0.19.0.
Jika versi tersebut sudah terpasang dan pemeriksaan dependensi berhasil,
tidak perlu memasang ulang paket. train.py harus memiliki --enable_allocator_gc,
dan mesh_extract_tnt.py harus tersedia di root proyek.

### B. Training baru dengan allocator GC

Gunakan langkah ini hanya untuk membuat model baru. Jika model iterasi 30000
sudah tersedia dan hanya ingin mengekstrak mesh, langsung ke bagian D.

Buat folder output unik agar hasil training sebelumnya tetap tersimpan:

~~~bash
mkdir -p /home/pande/projects/RaDe-GS/output/togog_bali
run_output=$(mktemp -d /home/pande/projects/RaDe-GS/output/togog_bali/togog_tualen_manual_XXXXXXXX)
printf 'Folder hasil: %s\n' "$run_output"
~~~

Lalu jalankan pada terminal yang sama:

~~~bash
python train.py \
  -s /home/pande/projects/RaDe-GS/dataset/togog_bali/togog_tualen \
  -m "$run_output" \
  --enable_allocator_gc \
  --iterations 30000 \
  --resolution 1920 \
  --profile_training \
  --profile_from 1800 \
  --profile_until 30000 \
  --profile_interval 100 \
  --profile_control_samples \
  --profile_gpu_interval 1 \
  --checkpoint_iterations 3000 7000 15000 25000
~~~

--enable_allocator_gc mengaktifkan allocator native dengan threshold 0.8
dan memory fraction eksplisit 1.0. Flag inilah yang harus disertakan saat
training manual; menjalankan train.py tanpa flag tersebut tidak mengaktifkannya.
Parameter dan profiling di atas mengikuti configs/togog_bali.yaml.

Tunggu sampai Training complete dan pastikan model tersedia:

~~~bash
ls -lh "$run_output/point_cloud/iteration_30000/point_cloud.ply"
~~~

### C. Ekstraksi mesh dari training baru

Setelah training berhasil, jalankan pada terminal yang sama sehingga run_output
masih menunjuk folder yang baru dibuat:

~~~bash
python mesh_extract_tnt.py \
  -m "$run_output" \
  --iteration 30000 \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5 \
  --voxel_size 0.009 \
  --max_depth 8.0
~~~

Perubahan dari command lama untuk Togog Tualen:

- Gunakan mesh_extract_tnt.py (ScalableTSDFVolume).
- Ejaan opsi yang benar adalah --depth_ratio, bukan --depth_ration.
- Pertahankan --num_cluster 8 dan --depth_ratio 0.5.
- Sertakan --voxel_size 0.009 dan --max_depth 8.0.
- Gunakan Open3D 0.19.0, versi yang telah berhasil diuji.

Parameter tersebut tidak memilih atau memotong hanya patung utama.
Jika transforms.json memiliki aabb_range, skrip dapat menghitung ulang ukuran
voxel dari AABB; periksa nilai TSDF voxel_size pada log.

### D. Ekstraksi saja dari model lama yang sudah selesai

Tidak perlu training ulang. Untuk model berhasil yang sudah ada di server,
jalankan perintah berikut. Skrip menulis recon.ply dan recon_post.ply di folder
model; menjalankan ulang akan mengganti kedua hasil mesh jika sudah ada.

~~~bash
python mesh_extract_tnt.py \
  -m /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen \
  --iteration 30000 \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5 \
  --voxel_size 0.009 \
  --max_depth 8.0
~~~

Periksa hasil model lama tersebut:

~~~bash
ls -lh \
  /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen/recon.ply \
  /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen/recon_post.ply
~~~

Untuk hasil training baru pada bagian B, periksa dengan:

~~~bash
ls -lh "$run_output/recon.ply" "$run_output/recon_post.ply"
~~~

recon.ply adalah mesh mentah; recon_post.ply adalah hasil post-processing.
Pemeriksaan keberadaan file bukan evaluasi kualitas terhadap GT.

### E. Perbedaan manual dan pipeline

Dua command manual di atas menjalankan training dan ekstraksi secara terpisah.
Rendering gambar, evaluasi citra, resource guard, dan rekap hasil pipeline
tidak otomatis dijalankan oleh kedua command tersebut.

Jika ingin urutan lengkap otomatis, periksa dry-run dahulu:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene togog_tualen \
  --start-from training \
  --dry-run
~~~

Untuk memulai run baru melalui pipeline, gunakan command yang sama tanpa
--dry-run. Ini memulai training baru; untuk model yang sudah selesai gunakan
bagian D. Panduan penyalinan kode ke server tersedia di
[Pipeline GC + TNT](docs/PIPELINE_GC_TNT.md).

## 14. Laporan otomatis: metrik, foto, splat, dan perbandingan

Setelah kode terbaru disalin ke server, pipeline membuat laporan otomatis untuk
setiap scene yang dijalankan. Perintah run tidak berubah. Contoh di Ubuntu-22.04-E:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene punakawan \
  --start-from training
```

Perintah ini memulai training baru. Gunakan --dry-run untuk memeriksa perintah
tanpa training atau membuat laporan. Untuk hasil training yang sudah ada, gunakan
ekspor manual di bawah agar tidak perlu mengulang training.

File kode yang diperlukan: research_pipeline/reporting.py,
research_pipeline/runner.py, render.py, dan scripts/export_research_report.py.
Tidak ada perubahan pada loss, densification, atau perhitungan metrik model.
render.py hanya menambahkan manifest pasangan gambar setelah rendering setiap split.

Laporan berada di <output scene>/research/laporan_<waktu>_<id>/:

| File | Isi |
| --- | --- |
| tabel_evaluasi.csv | Satu baris untuk run ini; dapat dibuka di Excel |
| detail_laporan.json | Sumber data, definisi VRAM, status gambar, lokasi PLY |
| 01_<scene>_daftar_foto.csv | Seluruh pasangan render/GT yang tersedia pada iterasi laporan, nama foto kamera, split, ukuran piksel, dan lokasi file |
| 01_<scene>/*_comparison.png | Lima contoh foto referensi di kiri dan render RaDe-GS di kanan |

Kolom tabel: waktu ringkasan (UTC+08), folder scene/kelas, metode RaDe-GS,
PSNR, SSIM, LPIPS, peak VRAM seluruh GPU (GiB), training (menit), split evaluasi,
jumlah gambar evaluasi, status run, jumlah foto terdaftar, foto training,
foto test, Gaussian awal, splats PLY akhir, iterasi PLY, dan folder output.

Pipeline juga menggabungkan baris run secara otomatis ke CSV pendamping:
nama results.csv pada konfigurasi diberi akhiran _laporan.
Untuk configs/togog_bali.yaml:

```text
research_results/experiment_results_laporan.csv
```

Tabel tersebut mempertahankan run dengan folder output berbeda, termasuk
percobaan ulang. CSV utama mengikuti results.csv pada YAML. Jangan menjalankan beberapa
pipeline bersamaan yang menulis CSV yang sama; pembaruan ini untuk pipeline sekuensial.
Run yang dilewati oleh --resume tidak dibuatkan laporan baru.

Jumlah foto terdaftar adalah gambar kamera yang dipakai dari COLMAP, bukan jumlah
semua file mentah atau frame video sebelum seleksi. Jumlah training, test, dan
evaluasi dipisahkan. Nama foto berasal dari render_manifest.json pada folder
train/ours_<iterasi>/ atau test/ours_<iterasi>/. Nama kamera dapat tidak menyertakan
ekstensi asli. Untuk run lama tanpa manifest, nama asli dikosongkan, tidak ditebak
dari nomor PNG. GT di sini adalah foto referensi, bukan ground truth geometri 3D.

Jumlah splat dibaca dari element vertex pada header
point_cloud/iteration_<iterasi akhir>/point_cloud.ply. Ini menghitung Gaussian
hasil training, bukan vertex recon.ply/recon_post.ply. Untuk bundle lokal,
iteration_<iterasi>/point_cloud.ply juga diterima. Membaca header tidak memeriksa
keutuhan seluruh isi PLY. Jika PLY tidak disertakan, jumlah splat kosong.

Peak VRAM seluruh GPU menggunakan maksimum sampel device_vram_used_mb dari run
profiler yang sama, MiB dibagi 1024 menjadi GiB. Nilai ini mencakup aplikasi lain
dan bisa melewatkan lonjakan di antara sampel. Ini berbeda dari VRAM per proses
atau peak allocator PyTorch. Peak allocated/reserved PyTorch dicatat terpisah
di detail_laporan.json. Jika tidak ada data, kolom kosong, bukan nol.
Jika beberapa GPU terdeteksi, nilai seluruh GPU dikosongkan; beberapa manifest
run dalam satu output ditolak agar eksperimen tidak tercampur.

Waktu tabel adalah waktu ringkasan, bukan waktu mulai training. Training (menit)
mencakup persiapan dan penyimpanan dalam subprocess training. Split train bukan
uji pada foto yang belum dilihat model; pertahankan label split dalam penelitian.

Lima gambar dipilih merata menurut urutan nama render, tanpa memilih yang
terlihat paling bagus. Piksel RGB tidak di-resize atau diubah kecerahannya.
Metrik tabel berasal dari evaluasi seluruh split, bukan lima contoh tersebut.
Perbandingan dengan 3DGS memerlukan render 3DGS pada kamera yang sama.

Untuk membuat laporan dari hasil lama atau bundle yang disalin, jalankan:

```bash
python -B scripts/export_research_report.py \
  --outputs output/togog_bali/punakawan \
  --pairs 5
```

Untuk menggabungkan beberapa hasil lama dalam satu tabel manual:

```bash
python -B scripts/export_research_report.py \
  --outputs output/togog_bali/punakawan \
            output/togog_bali/full_gc/togog_tualen
```

Ekspor manual memakai CPU dan tidak mengubah master CSV atau model. Setiap
pemanggilan membuat folder laporan baru. --pairs 10 membuat sepuluh contoh.

Jika memindahkan hasil ke komputer lokal untuk analisis, sertakan research/,
point_cloud/iteration_30000/point_cloud.ply, dan folder train/ours_30000/
serta test/ours_30000/ jika ada. Folder render berisi gt/, renders/, dan manifest.
Tidak perlu checkpoint untuk laporan ini. Tanpa gambar, tabel tetap dibuat
tetapi detail menandai missing_images; laporan tidak bisa membuat gambar dari
file JSON saja. Kegagalan ekspor dilaporkan sebagai report_generation, hasil
training yang sudah tersimpan tetap dipertahankan.

### Pemeriksaan sebelum menjalankan konfigurasi 1920

Jalankan pada distro Ubuntu-22.04-E di server, setelah kode terbaru disalin:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -c "import torch, open3d, PIL, yaml, psutil, scipy; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Open3D:', open3d.__version__)"
python -m pip check
df -h .
nvidia-smi

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --start-from training \
  --dry-run
~~~

Pastikan TRAIN memuat --resolution 1920 dan --enable_allocator_gc, lalu jalankan
perintah yang sama tanpa --dry-run. Pemeriksaan CPU tidak menjamin kebutuhan
VRAM/RAM atau kestabilan penyimpanan selama run GPU.

Kegagalan scene dicatat pada research_results/experiment_results.csv
(status, failed_stage, error_message, output_path) dan pada summary.json serta
logs/ di output scene/attempt. Pipeline meneruskan antrean setelah kegagalan
scene yang tertangani. Bila gagal menulis CSV utama, proses berhenti; kerusakan
penyimpanan atau WSL crash tidak dapat diatasi dengan melanjutkan antrean.
Jika pembuatan laporan gagal setelah tabel ditulis, status tabel dikoreksi
menjadi failed bila penyimpanan masih dapat ditulis. CSV utama dan summary.json
menjadi acuan pemeriksaan kegagalan.
