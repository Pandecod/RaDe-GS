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
  script: mesh_extract.py
  resolution: 2
  num_cluster: 8
  depth_ratio: 0.5
  voxel_size: 0.009
  max_depth: 7.0
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
30.000 iterasi dan profiling. Ekstraksi utama kini memakai mesh_extract.py
untuk uji Ngawasari (bagian 18). Percobaan Punakawan 1600 memakai TNT dan telah berhasil;
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
  max_depth: 7.0
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
  --max_depth 7.0
~~~

Perubahan dari command lama untuk Togog Tualen:

- Gunakan mesh_extract_tnt.py (ScalableTSDFVolume).
- Ejaan opsi yang benar adalah --depth_ratio, bukan --depth_ration.
- Pertahankan --num_cluster 8 dan --depth_ratio 0.5.
- Sertakan --voxel_size 0.009 dan --max_depth 7.0.
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

## 15. Antrean Togog Bali lalu Tanks and Temples

Gunakan RaDe-GS dan environment radegs dari checkout pande pada distro
Ubuntu-22.04-E. Dataset kedua dibaca dari dataset.root pada configs/tanks_temples.yaml;
lokasi pada konfigurasi terbaru adalah:

~~~text
/home/pande/projects/RaDe-GS/datasets/tanks_temples
~~~

Alamat Windows yang sesuai:
\\wsl.localhost\Ubuntu-22.04-E\home\pande\projects\RaDe-GS\datasets\tanks_temples.
Path dataset pada YAML menggunakan bentuk Linux. Tidak perlu menyalin dataset
ke folder pande atau menjalankan model 2DGS.

configs/tanks_temples.yaml menggunakan training 1920 piksel, 30.000 iterasi,
allocator GC, profiling, rendering, metrik, mesh TNT, dan laporan otomatis yang
sama dengan konfigurasi utama. Ini konfigurasi awal; parameter tersebut belum
diverifikasi dengan run GPU pada dataset Tanks and Temples.

Struktur yang dikonfirmasi pengguna adalah family/ dan house/ langsung di
bawah tanks_temples; masing-masing berisi images/ dan sparse/0/. Huruf besar
atau kecil nama folder mengikuti nama sebenarnya dan akan dideteksi otomatis.
Contoh isi setiap scene:

~~~text
tanks_temples/
  <nama_scene>/
    images/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
~~~

File COLMAP .txt yang lengkap juga diterima. Folder foto lain atau kelompok
bertingkat seperti training/<scene>/ tidak otomatis dibaca sebagai scene
langsung; periksa --list-scenes sebelum menjalankan. Pengguna yang menjalankan pipeline perlu izin
membaca dataset sumber; pipeline memeriksa isinya saat discovery/validasi.
Jika dataset berada pada akun pengguna lain, jangan mengubah kepemilikan
seluruh direktori home untuk mengatasi izin.

Periksa daftar folder kedua dataset:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --start-from training \
  --list-scenes
~~~

Lalu periksa seluruh perintah tanpa menjalankan model:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --start-from training \
  --dry-run
~~~

Jika tidak ada INVALID/FAILED dan semua path benar, jalankan antrean:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --start-from training
~~~

Urutan mengikuti penulisan YAML: semua scene Togog Bali diselesaikan terlebih
dahulu (training sampai laporan), kemudian semua scene Tanks and Temples.
Folder scene dalam setiap dataset diproses menurut nama.
Terminal menampilkan [CONFIG 1/2] dan [CONFIG 2/2], serta [nomor scene/total scene].
Kegagalan scene yang tertangani tidak membatalkan dataset berikutnya; proses
mengembalikan kode 1 setelah seluruh antrean selesai bila ada scene yang gagal.
Error konfigurasi, kegagalan CSV utama, dan Ctrl+C menghentikan antrean.
WSL crash juga menghentikan proses, sehingga antrean tidak berlanjut otomatis.

Tambahkan --resume pada command antrean untuk melewati scene yang sudah berhasil
dengan konfigurasi sama pada masing-masing CSV. Ini tidak melanjutkan checkpoint.
Jika Togog Bali sudah berjalan dengan command lama, command antrean baru tidak
menempel pada proses lama; jalankan setelah proses itu selesai dengan --resume.
Jangan menjalankan kedua command secara bersamaan pada GPU yang sama.
Opsi --scene, bila diberikan, berlaku pada setiap konfigurasi; untuk seluruh
antrean dua dataset, hilangkan opsi tersebut.

| Dataset | Output di checkout RaDe-GS | CSV utama |
|---|---|---|
| Togog Bali | output/togog_bali/<scene>/ | research_results/experiment_results.csv |
| Tanks and Temples | output/tanks_temples/<scene>/ | research_results/experiment_results_tanks_temples.csv |

Tabel ringkas Tanks and Temples berada di
research_results/experiment_results_tanks_temples_laporan.csv.
Gambar perbandingan, daftar foto, dan detail laporan berada di
output/tanks_temples/<scene>/research/laporan_<waktu>_<id>/, atau di dalam runs/
untuk percobaan ulang. Model, checkpoint, mesh, dan laporan RaDe-GS ditulis
ke output proyek RaDe-GS. Loader COLMAP bawaan dapat membuat
sparse/0/points3D.ply pada sumber jika file itu belum ada. Bila pengguna pande
hanya memiliki izin baca, minta pemilik dataset menyiapkan points3D.ply
terlebih dahulu; tidak perlu mengganti kepemilikan direktori sumber.

## 16. Uji ulang mesh_extract.py bawaan tanpa mengubah hasil TNT

Kedua skrip ekstraksi disediakan oleh RaDe-GS. Pemilihan untuk penelitian perlu
dicatat bersama versi kode, parameter, environment, waktu, dan hasilnya.
Tidak perlu training ulang untuk membandingkan ekstraksi dari Gaussian yang sama.

Penting: mesh_extract.py pada checkout ini sudah dimodifikasi pada percobaan
terdahulu. Antara lain, voxel_size menjadi 0.009, integrasi dilakukan per kamera,
block_count menjadi 20000, dan weight_threshold menjadi 3.0.
Jangan menyebut file tersebut versi bawaan yang belum dimodifikasi.

Uji di bawah mengambil mesh_extract.py dari commit dasar c7c56cc pada riwayat Git
checkout ini, tanpa mengubah isi skripnya. Versi tersebut memakai
VoxelBlockGrid pada CPU, voxel_size 0.002, block_count 50000, dan menyimpan
render warna/depth seluruh kamera sebelum integrasi. Ini berbeda dari
parameter/jalur TNT yang berhasil; voxel lebih kecil dan buffering seluruh kamera
dapat meningkatkan kebutuhan RAM. Belum ada jaminan run ini selesai pada server.
Perbedaan hasil tidak boleh langsung dianggap hanya akibat pilihan ekstraktor,
karena ukuran voxel serta pemrosesannya juga berbeda.

Gunakan environment radegs yang sudah memiliki Open3D 0.19.0.
Tidak perlu menyalin environment, memasang ulang paket, atau mengubah konfigurasi
pipeline utama. Jalankan uji ketika GPU dan RAM server tidak sedang dipakai
untuk training lain.

Contoh memakai model Tualen yang sebelumnya sudah selesai.
Jika model yang dibandingkan berbeda, ganti nilai model dengan folder yang
memiliki cfg_args dan point_cloud/iteration_30000/point_cloud.ply.

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -c "import torch, open3d; print('PyTorch:', torch.__version__); print('Open3D:', open3d.__version__); print('CUDA:', torch.cuda.is_available())"
free -h
df -h .
nvidia-smi
~~~

Perintah berikut membuat folder uji baru di research/. Model Gaussian hanya
diakses lewat symlink; tidak disalin dan tidak dilatih ulang. File konfigurasi
kecil dan sumber ekstraktor disimpan untuk menelusuri percobaan.
Scene akan menulis multi_view.json dan mesh ke folder uji ini.

~~~bash
(
set -eu
cd /home/pande/projects/RaDe-GS

model=/home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen
iteration=30000
revision=c7c56cc

test -f "$model/cfg_args"
test -f "$model/point_cloud/iteration_$iteration/point_cloud.ply"
git cat-file -e "$revision:mesh_extract.py"

mkdir -p "$model/research"
mesh_trial=$(mktemp -d "$model/research/mesh_vbg_original_XXXXXXXX")
git show "$revision:mesh_extract.py" > "$mesh_trial/mesh_extract.py"
git rev-parse "$revision" > "$mesh_trial/extractor_revision.txt"
git rev-parse HEAD > "$mesh_trial/checkout_revision.txt"
sha256sum "$mesh_trial/mesh_extract.py" > "$mesh_trial/extractor_sha256.txt"
cp -- "$model/cfg_args" "$mesh_trial/cfg_args"
ln -s -- "$model/point_cloud" "$mesh_trial/point_cloud"

printf 'Model: %s\nFolder uji: %s\n' "$model" "$mesh_trial"
printf 'iteration=%s\nresolution=2\nnum_cluster=8\ndepth_ratio=0.5\nvoxel_size_in_original_script=0.002\n' "$iteration" > "$mesh_trial/parameters.txt"

mesh_started=$(date +%s)
set +e
PYTHONPATH="$PWD" python -u "$mesh_trial/mesh_extract.py" \
  -m "$mesh_trial" \
  --iteration "$iteration" \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5 2>&1 | tee "$mesh_trial/mesh.log"
mesh_exit_code=${PIPESTATUS[0]}
set -e
mesh_finished=$(date +%s)

printf '%s\n' "$mesh_exit_code" > "$mesh_trial/exit_code.txt"
printf '%s\n' "$((mesh_finished - mesh_started))" > "$mesh_trial/elapsed_seconds.txt"
printf 'Kode keluar: %s\nFolder uji: %s\n' "$mesh_exit_code" "$mesh_trial"

if [ "$mesh_exit_code" -eq 0 ]; then
    ls -lh "$mesh_trial/recon.ply" "$mesh_trial/recon_post.ply"
fi
exit "$mesh_exit_code"
)
~~~

Pengaturan voxel 0.002 berasal dari skrip asli; opsi --voxel_size tidak tersedia
pada versi tersebut. Satuan koordinat tidak otomatis meter/milimeter.
Output TNT pada folder model tetap tersedia untuk dibandingkan.
Log dan hasil percobaan manual ini berada di folder uji, bukan otomatis menjadi
baris eksperimen sukses pada CSV utama. Nilai durasi merupakan keseluruhan proses
ekstraksi, termasuk pemuatan model, rendering depth, integrasi, dan post-processing.
Jika WSL berhenti atau proses terputus, exit_code.txt/elapsed_seconds.txt mungkin
belum sempat ditulis; log parsial tidak membuktikan keberhasilan.

Keberhasilan baru dinilai setelah proses selesai tanpa error, kedua mesh
tersimpan dan bisa dibuka, serta detail objek dibandingkan pada bagian/sudut
yang sama. Jangan mengubah YAML antrean utama sebelum uji ini dinilai.

## 17. Uji batas depth tetap 7.0

Opsi depth otomatis/radius telah dibatalkan. Perubahan uji sekarang hanya
menurunkan batas kedalaman dari 8.0 menjadi 7.0 dalam satuan koordinat
rekonstruksi, belum tentu meter. Nilai lebih kecil bisa mengurangi permukaan
yang jauh dari kamera, tetapi belum tentu memisahkan patung yang bersebelahan
dan bisa memotong bagian objek.

- configs/togog_bali.yaml: mesh.max_depth sekarang 7.0.
- mesh_extract.py pada checkout ini memakai batas depth tetap dari YAML/CLI;
  default-nya 7.0. Pemilihan pipeline non-TNT dijelaskan pada bagian 18.
- configs/tanks_temples.yaml serta konfigurasi 1024/1600 terdahulu tetap 8.0.
- Ekstraktor terpilih menerima --max_depth 7.0 dari YAML. Konfigurasi utama
  sekarang memilih mesh_extract.py; Tanks tetap memakai TNT.
- Training tetap 1920 piksel/30.000 iterasi, voxel tetap 0.009, dan rumus model
  tidak berubah. Nama folder hasil serta pelaporan tetap seperti sebelumnya.

Bagian 16 memakai skrip asli dari commit lama dengan depth 8.0 dan voxel 0.002;
itu percobaan berbeda. Jangan menyebut uji 7.0 pada checkout yang telah
disesuaikan sebagai salinan kode asli yang tidak dimodifikasi.

Setelah kode terbaru disalin ke server, periksa:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --start-from training \
  --dry-run
~~~

Pastikan MESH menampilkan --max_depth 7.0. Command pipeline tanpa --dry-run
memulai training baru; perubahan depth saja tidak memerlukan training ulang.

Untuk ekstraksi TNT saja dari model yang sudah selesai, gunakan command bagian
13 dengan --max_depth 7.0. Simpan hasil pembanding 8.0 terlebih dahulu karena
ekstraksi menulis ulang mesh pada folder model.
Untuk mesh_extract.py non-TNT lokal, --max_depth dan --voxel_size kini tersedia
agar pipeline dapat meneruskan angka tetap dari YAML; tidak ada mode radius otomatis.
Gunakan folder uji terpisah agar mesh TNT tetap tersimpan.

Catat max_depth 7.0 dalam laporan eksperimen. Pada pipeline, nilainya tersimpan
di research/summary.json pada mesh_extraction.max_depth dan perintah mesh.
Penurunan ini bukan pengaturan voxel atau peningkatan resolusi detail.

## 18. Uji pipeline Ngawasari dengan mesh_extract.py tanpa TNT

Konfigurasi utama configs/togog_bali.yaml sekarang memilih:

~~~yaml
mesh:
  enabled: true
  script: mesh_extract.py
  resolution: 2
  num_cluster: 8
  depth_ratio: 0.5
  voxel_size: 0.009
  max_depth: 7.0
  extra_args: []
~~~

Ini menggunakan mesh_extract.py yang ada pada checkout penelitian, dengan
VoxelBlockGrid, integrasi per kamera, dan parameter/bobot yang sudah ada.
Bukan salinan upstream murni pada bagian 16. Perubahan integrasi pipeline hanya
menambahkan pemilihan skrip dan argumen voxel/depth; tidak menambahkan radius
otomatis atau mengubah training/model. Skrip tersebut kini menerima
--voxel_size dan --max_depth agar YAML sesuai dengan ekstraksi nyata.
Tanpa opsi baru, default non-TNT tetap voxel 0.009 dan depth 7.0.

configs/tanks_temples.yaml serta konfigurasi percobaan terdahulu tetap memakai
mesh_extract_tnt.py melalui default mesh.script. Untuk kembali memakai TNT pada
konfigurasi utama, ubah script: mesh_extract_tnt.py. Identitas eksperimen
membedakan pilihan ekstraktor, sehingga --resume tidak mencampur keberhasilan
TNT dengan percobaan VoxelBlockGrid. Identitas konfigurasi TNT lama dipertahankan.

Dataset Ngawasari:
\\wsl.localhost\Ubuntu-22.04-E\home\pande\projects\RaDe-GS\dataset\togog_bali\ngawasari

Di Ubuntu-22.04-E, jalankan sebagai pande pada environment radegs.
Pastikan seluruh perubahan kode terbaru sudah di-push/pull ke server; mengganti
YAML saja tidak cukup bila runner/CLI ekstraktor masih versi lama.

Periksa input serta environment tanpa menjalankan training:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

ls -ld dataset/togog_bali/ngawasari/images
ls -lh dataset/togog_bali/ngawasari/sparse/0/
python -c "import torch, open3d; print('PyTorch:', torch.__version__); print('Open3D:', open3d.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pip check

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene ngawasari \
  --start-from training \
  --dry-run
~~~

Pastikan hanya [1/1] ngawasari yang dipilih, TRAIN memuat --resolution 1920,
dan MESH memanggil mesh_extract.py dengan --max_depth 7.0, --voxel_size 0.009,
-r 2, --num_cluster 8, --depth_ratio 0.5. Dataset harus memiliki images/ serta
sparse/0/cameras.bin, images.bin, points3D.bin (atau file teks yang lengkap).

Untuk menjalankan pipeline penuh Ngawasari:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene ngawasari \
  --start-from training
~~~

Urutan: training 30.000 iterasi, rendering, evaluasi citra, pemeriksaan resource,
ekstraksi VoxelBlockGrid, analisis mesh, kemudian laporan otomatis.
Ini memulai training baru; bukan ekstraksi ulang dari checkpoint.
Konfigurasi utama berlaku untuk semua scene Togog Bali, tetapi --scene
ngawasari membatasi command uji ini pada Ngawasari saja.

Hasil pertama berada di output/togog_bali/ngawasari/. Jika sudah berisi hasil,
run baru berada pada runs/<waktu>_<id>/. Model Gaussian berada di
point_cloud/iteration_30000/point_cloud.ply, mesh mentah non-TNT di recon_raw.ply,
dan mesh pascaproses di recon_post.ply.
Laporan berada di research/, dengan rekap utama di
research_results/experiment_results.csv dan experiment_results_laporan.csv.
Skrip yang benar-benar dipilih tercatat pada summary.json dan perintah mesh.
Kegagalan mesh tetap dicatat; kualitas dan kebutuhan memori VoxelBlockGrid pada
Ngawasari perlu dibuktikan melalui run ini. Pengujian CPU/dry-run tidak menjamin
ekstraksi GPU selesai.

Jalankan uji satu scene ini dahulu sebelum memakai konfigurasi non-TNT untuk
seluruh antrean. Jangan menghapus hasil pembanding sebelum menilai mesh baru.
