# Panduan Penggunaan Pipeline RaDe-GS

Panduan ini berisi langkah operasional Research Pipeline. Kondisi implementasi,
hasil tes, dan batasan teknis tersedia di
[Laporan Kondisi RaDe-GS](LAPORAN_KONDISI_RADEGS.md).

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
  extra_args: []

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
  recon_raw.ply
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
| `recon_raw.ply` | Mesh mentah |
| `recon_post.ply` | Mesh final setelah post-processing |
| `research/logs/*.log` | Output mentah setiap tahap |
| `training_metrics.csv` | Metrik yang diparse dari training |
| `image_metrics.json` | SSIM, PSNR, LPIPS, split, dan jumlah citra |
| `mesh_metrics.json` | Durasi, resource, vertex, face, dan ukuran mesh |
| `summary.json` | Status lengkap satu attempt |
| `experiment_results.csv` | Tabel ringkas seluruh experiment ID |

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
