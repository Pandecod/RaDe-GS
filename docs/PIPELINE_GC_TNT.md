# Pipeline GC + TNT — 27 Oktober 2026

Pipeline lokal sekarang memakai pengaturan yang berhasil pada server:
training allocator GC, kemudian rendering, metrik gambar, dan ekstraksi
mesh_extract_tnt.py dengan Open3D 0.19.0. Integrasi ini tidak menjalankan training
atau ekstraksi baru. Keberhasilan mesh nyata sebelumnya berasal dari run manual.

## File yang perlu disalin ke server

Salin dengan struktur folder yang sama di /home/pande/projects/RaDe-GS:

| File | Kegunaan |
|---|---|
| research_pipeline/config.py | Validasi CLI TNT serta nilai voxel_size dan max_depth. |
| research_pipeline/commands.py | Membentuk perintah mesh_extract_tnt.py. |
| research_pipeline/runner.py | Mencatat skrip dan parameter mesh dalam hasil run. |
| configs/togog_bali.yaml | Konfigurasi utama GC, 30.000 iterasi, resolusi 1920, profiling dan mesh TNT; hasil di output/togog_bali. |
| configs/togog_bali_full_gc.yaml | Konfigurasi pembanding 1024 dengan root output dan CSV full_gc. |
| requirements.txt | Menetapkan open3d==0.19.0 untuk instalasi berikutnya. |

train.py harus merupakan versi yang sudah memiliki --enable_allocator_gc.
mesh_extract_tnt.py harus tersedia di root proyek; gunakan skrip yang berhasil
sebelumnya. Keduanya tidak diubah dalam integrasi ini. Dependensi pipeline
tetap tercantum di requirements-research.txt.

Cadangkan file server yang akan diganti, lalu salin file di atas. Periksa
dataset.root, output.root, dan results.csv sebelum menjalankan pipeline.
Tidak perlu menyalin dataset, checkpoint, atau environment. Dokumentasi dan
tests dapat ikut disertakan ke GitHub, tetapi tidak diperlukan untuk eksekusi server.

## Pengaturan yang aktif

- Semua YAML: --enable_allocator_gc dan --iterations 30000.
- Konfigurasi utama togog_bali.yaml: --resolution 1920; full_gc tetap 1024.
- Profiling 1800–30000 setiap 100 iterasi, sampel kontrol, telemetry GPU 1 detik.
- Checkpoint pada 3000, 7000, 15000, 25000.
- Mesh: -r 2 --num_cluster 8 --depth_ratio 0.5 --voxel_size 0.009 --max_depth 8.0.
- Mesh memilih iterasi tersimpan paling akhir melalui default --iteration -1.
  Setelah training konfigurasi ini selesai, model yang dipakai adalah iterasi 30000.
  Untuk kebutuhan khusus, --iteration boleh diberikan melalui mesh.extra_args.
- Nilai voxel_size dan max_depth berada pada bagian mesh YAML, harus positif.
  Jangan menduplikasinya dalam mesh.extra_args.
- Rumus loss, optimizer, densification, pruning, dan rasterizer tidak diubah.
- Parameter mesh di atas tidak melakukan crop patung. Jika dataset memiliki
  transforms.json dengan aabb_range, skrip TNT dapat menghitung ulang voxel_size
  dari AABB. Periksa baris TSDF voxel_size pada log.
- GC tetap opt-in pada train.py manual. Tanpa --enable_allocator_gc, opsi itu
  tidak aktif. Untuk menonaktifkannya di pipeline, hapus flag tersebut dari YAML.
- Untuk menonaktifkan profiling, hapus --profile_training dari training.extra_args.

Mulai 30 Oktober 2026, konfigurasi utama menulis ke
output/togog_bali/<scene> dengan CSV
research_results/experiment_results.csv.
Alternatif full_gc menulis ke output/togog_bali/full_gc/<scene>.
Jika folder scene sudah berisi hasil, runner membuat runs/<waktu>_<id>
untuk eksekusi baru. Dry-run menampilkan path dasar; path aktual run ulang
dapat berada di subfolder runs. Model lama tidak perlu dilatih ulang.

## Pemeriksaan server tanpa menjalankan model

Jalankan di Ubuntu/WSL server:

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -c "import torch, open3d; print('PyTorch:', torch.__version__); print('Open3D:', open3d.__version__)"
python -m pip check

python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene togog_tualen \
  --start-from training \
  --dry-run
~~~

Pastikan TRAIN memuat --enable_allocator_gc, --iterations 30000, --resolution 1920.
Pastikan MESH memanggil mesh_extract_tnt.py dengan voxel_size 0.009 dan max_depth 8.0.
Dry-run tidak menjalankan subprocess training/mesh atau membuat hasil eksperimen.

requirements.txt tidak otomatis mengganti versi paket yang sudah terpasang.
Jika server sudah memakai Open3D 0.19.0 dan pip check berhasil, tidak perlu
memasang ulang. Jika belum sesuai, gunakan panduan pemulihan untuk instalasi
dependensi dan uji TSDF kecil sebelum ekstraksi.

## Perintah eksekusi untuk run baru

Jalankan hanya ketika memang ingin memulai training baru beserta pipeline lengkap:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene togog_tualen \
  --start-from training
~~~

Untuk model 30000 yang sudah berhasil, tidak perlu menjalankan perintah ini lagi.
Perintah manual ekstraksi pada panduan pemulihan tetap tersedia; pipeline
yang dimulai dari training memang melakukan training baru.

## Validasi dan pengembalian

Pengujian lokal menggunakan fixture CPU dan executor palsu: memeriksa konfigurasi,
parser argumen skrip asli, dry-run tanpa subprocess, urutan tahapan, metadata,
dan penanganan kegagalan mesh. Ini bukan pengujian GPU end-to-end pada server.

Salinan file sebelum integrasi tersedia di folder lokal yang diabaikan Git:
arsip_lokal/integrasi_pipeline_20261027T000636Z/.
Untuk membatalkan integrasi, pulihkan file dengan path relatif yang sama dari
folder itu setelah memeriksa perubahan yang dibuat sesudahnya. File baru panduan
ini dapat dihapus terpisah. Pemulihan kode tidak mengubah paket environment;
jangan mengganti Open3D server yang sudah berhasil hanya untuk membatalkan
perubahan konfigurasi.

Tidak ada commit, instalasi paket, training, atau ekstraksi otomatis dalam
pekerjaan integrasi ini.
