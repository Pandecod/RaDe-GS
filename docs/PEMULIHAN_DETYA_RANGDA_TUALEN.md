# Pemulihan detya, rangda, dan tualen

Diagnosis ini berasal dari log antrean Togog Bali tanggal 4 November 2026.
Status ketiga objek memerlukan penanganan berbeda.

| Objek | Bukti dari log | Tindakan |
| --- | --- | --- |
| detya | Training gagal pada iterasi tercatat 14800 dengan CUDA out of memory saat densify_and_split/torch.cat. | Ulangi training dengan penyimpanan foto bawaan di CPU. |
| rangda | Training dan evaluasi selesai; mesh_extraction berakhir -11. | Gunakan model 30000 yang ada untuk retry mesh dengan pencatatan crash. |
| tualen | Training dan evaluasi selesai; mesh_extraction berakhir -11. | Gunakan model 30000 yang ada untuk retry mesh dengan pencatatan crash. |

## Detya: mengurangi VRAM untuk foto dengan opsi bawaan

Saat gagal, PyTorch mencatat alokasi 20.43 GiB, cadangan belum terpakai 3.38 GiB,
dan permintaan tambahan 372 MiB yang gagal. Ini bukti kehabisan VRAM.
Besarnya cadangan belum membuktikan fragmentasi sebagai satu-satunya penyebab.
Angka miliaran GiB pada bagian non-PyTorch merupakan pembacaan yang tidak masuk
akal untuk GPU ini; jangan memakainya sebagai pemakaian memori nyata.

Pada kode kamera, RGB dan grayscale disimpan di `data_device`; default-nya CUDA.
Opsi asli `--data_device cpu` menyimpan persediaan foto tersebut di RAM.
Training tetap memakai CUDA: foto aktif dipindahkan melalui `.cuda()` dan
pencocokan antarfoto memindahkan grayscale yang diperlukan melalui `.to("cuda")`.

Preset `configs/togog_bali_detya_cpu_images.yaml` hanya menambahkan opsi bawaan
tersebut pada konfigurasi utama. Resolusi 1920, 30000 iterasi, split --eval,
allocator GC, loss, densifikasi, serta parameter mesh tetap sama.
Kode model, renderer, metric.py, dan mesh_extract.py tidak diedit untuk preset ini.
Konfigurasi utama Togog Bali dan Tanks and Temples tetap seperti sebelumnya.

Opsi ini mengurangi VRAM persediaan foto, tetapi menambah penggunaan RAM dan
transfer CPU-GPU. Catat data_device saat membandingkan waktu dan peak VRAM.
Run server pada 5 November 2026 berhasil menyelesaikan training, rendering,
metric.py pada 72 foto test, serta mesh untuk detya. Hasil run berada di
`detya/runs/20261104T175137_fb0014f833/`. Keberhasilan ini tidak menjamin
penyebab crash ekstraksi pada objek lain ikut teratasi.

Setelah kode/preset terbaru tersedia di server, periksa dahulu:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B research_pipeline/run_all.py \
  --config configs/togog_bali_detya_cpu_images.yaml \
  --scene detya \
  --start-from training \
  --dry-run
```

TRAIN harus memuat `--data_device cpu`, `--resolution 1920`,
`--enable_allocator_gc`, dan `--eval`. Daftar harus berisi satu objek: detya.
Simpan log percobaan gagal sebelumnya untuk pembanding.

Untuk menjalankan percobaan:

```bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali_detya_cpu_images.yaml \
  --scene detya \
  --start-from training \
  --resume
```

Gunakan `--scene detya` agar preset pemulihan tidak menjalankan ulang objek lain.
Training dimulai dari awal; log gagal belum menunjukkan penyimpanan checkpoint
15000. Gaussian PLY iterasi 7000 tidak sama dengan checkpoint optimizer.
Output lama tetap disimpan; percobaan baru dialokasikan pada `detya/runs/<id>/`.
Rendering, metrik foto test, mesh, GT opsional, dan laporan tetap mengikuti pipeline.

Perubahan argumen menghasilkan experiment_id berbeda. Status percobaan baseline
yang gagal tetap tercatat. `--resume` pada YAML utama masih dapat mencoba baseline
detya lagi, meskipun preset CPU-images sudah sukses. Untuk melewati percobaan
CPU-images yang sukses, gunakan preset yang sama. Jangan mengubah status baseline
secara manual agar terlihat sukses.

Jika OOM tetap terjadi, baca log percobaan baru sebelum mengubah parameter
algoritma. Preset ini tidak membatasi jumlah Gaussian atau mengurangi resolusi
secara otomatis. GPU tetap harus mampu menampung model dan tensor kerja.

## Rangda dan tualen: retry mesh tanpa training

Kode -11 menunjukkan proses Linux dihentikan oleh signal 11 (SIGSEGV).
Log lama belum memuat baris fungsi saat crash; tidak cukup untuk menyatakan
bahwa penyebabnya Open3D, rasterizer CUDA, RAM, depth, atau jumlah cluster.

Mulai dari satu objek agar hasil diagnosis mudah dibaca:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

PYTHONFAULTHANDLER=1 python -B scripts/retry_mesh.py \
  --model output/hasil_radegs_v2/togog_bali/tualen \
  --script mesh_extract.py \
  --iteration 30000 \
  --resolution 2 \
  --num-cluster 8 \
  --depth-ratio 0.5 \
  --voxel-size 0.009 \
  --max-depth 8.0
```

Faulthandler meminta Python mencetak stack saat crash. Parameter model dan mesh
tetap sama; opsi ini membantu diagnosis, bukan menjamin crash sudah diperbaiki.
Retry memakai model yang sudah selesai, mencatat waktu/resource, dan menaruh
hasil pada `<model>/research/mesh_retry_<id>/`. Wrapper menyalin konfigurasi kecil
dan menautkan point_cloud; tidak menggandakan PLY training atau menimpa hasil asal.

Jika dibutuhkan, jalankan percobaan rangda dengan perintah yang sama:

```bash
PYTHONFAULTHANDLER=1 python -B scripts/retry_mesh.py \
  --model output/hasil_radegs_v2/togog_bali/rangda \
  --script mesh_extract.py \
  --iteration 30000 \
  --resolution 2 \
  --num-cluster 8 \
  --depth-ratio 0.5 \
  --voxel-size 0.009 \
  --max-depth 8.0
```

Pada folder retry yang dicetak terminal, periksa:

- `result.json` atau `result.csv`: status, exit code, pengaturan, dan waktu.
- `mesh.log`: traceback/stack lokasi crash.
- `mesh_resources_*.csv`: RAM dan VRAM yang berhasil disampel.
- `recon_post.ply`: mesh yang lolos pemeriksaan jika retry sukses.

Retry menyimpan laporan tersendiri. Status run asal pada tabel gabungan tetap
failed sampai hasil pemulihan ditinjau dan diintegrasikan dengan sumber yang
jelas. Menjalankan exporter dari summary lama saja tidak menggabungkan retry
atau otomatis membuat status run asal menjadi success.

Data GT yang belum tersedia tetap dilewati oleh pipeline; hal itu berbeda dari
kegagalan training atau mesh pada log ini.

## Pemulihan tualen: percobaan blok voxel 8

Diagnosis server tanggal 5 November 2026 menunjukkan:

- Kedua objek crash pada `mesh_extract.py:85`, yaitu `vbg.extract_triangle_mesh()`.
- Tualen: 41.578 blok aktif, kapasitas 50.000, RAM proses 7,67 GiB,
  RAM tersedia 48,47 GiB tepat sebelum ekstraksi.
- Pada Open3D 0.19.0, buffer bantu berisi empat Int32 per voxel.
  Untuk blok 16, buffer ini mencapai 2.537720 GiB pada 41.578 blok.
  Perhitungan byte offset 32-bit merupakan dugaan kuat, belum konfirmasi
  stack C++ atau hasil uji perbaikan pustaka. Data ini belum tersedia untuk rangda.

Opsi retry `--block-resolution 8` mengubah ukuran kelompok voxel dalam memori,
dari 16 x 16 x 16 menjadi 8 x 8 x 8. Jarak antarvoxel tetap 0.009,
depth tetap 8.0, mesh tetap -r 2, dan PLY training 30000 yang sama dipakai.
Block resolution berbeda dari resolution gambar dan ukuran voxel.

Implementasinya berupa launcher `research_pipeline/native_mesh_runtime.py`
yang menjalankan `mesh_extract.py` yang ada, dengan override constructor
VoxelBlockGrid hanya untuk proses retry tersebut. Integrasi dan pembentukan
segitiga tetap dipanggil pada Open3D. Source `mesh_extract.py`, kode training,
renderer, metric.py, YAML utama, dan hasil asal tetap seperti sebelumnya.

Ini percobaan pemulihan, belum terbukti berhasil pada tualen. Mengubah ukuran blok
dapat mengubah blok yang dialokasikan saat integrasi. Jangan mengklaim mesh pasti
identik dengan block_resolution=16. Catat parameter ini dalam metodologi dan
periksa cakupan permukaan, detail, vertex/face, serta metrik geometri jika GT ada.
Untuk mempertahankan layout 16 juga, arah perbaikannya tetap pengujian pustaka
Open3D dengan pengalamatan buffer yang diperbaiki.

Jumlah blok aktif bisa bertambah saat blok diperkecil. Karena itu launcher
memeriksa `jumlah_blok * block_resolution^3 * 4 * 4` sebelum ekstraksi.
Jika buffer masih melebihi 2 GiB, percobaan dihentikan dengan pesan guard,
bukan diberi status sukses atau diubah otomatis ke metode/voxel lain.
Guard ini ditujukan pada buffer yang dicurigai; kegagalan internal lain masih mungkin.

Setelah perubahan di-push dan di-pull ke server, jalankan uji CPU:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

PYTHONPATH=tests python -B -m unittest test_native_mesh_runtime
```

Jika Open3D 0.19.0 tersedia, uji tambahan membentuk bidang depth sintetis dengan
kedua layout. Ini memeriksa API dan bidang sederhana, bukan menjamin kesetaraan
mesh nyata atau keberhasilan ekstraksi tualen.

Jalankan pemulihan tualen saja:

```bash
PYTHONFAULTHANDLER=1 python -B scripts/retry_mesh.py \
  --model output/hasil_radegs_v2/togog_bali/tualen \
  --script mesh_extract.py \
  --iteration 30000 \
  --resolution 2 \
  --num-cluster 8 \
  --depth-ratio 0.5 \
  --voxel-size 0.009 \
  --max-depth 8.0 \
  --block-resolution 8
```

Terminal harus mencetak `[NATIVE MESH RETRY]`, block_resolution 8, dan voxel_size
0.009. Sebelum ekstraksi, jumlah blok, kapasitas, dan ukuran buffer ikut dicetak.
Jika belum ada flag baru pada `--help`, perubahan ini belum terpasang di server.

Output berada di folder retry baru yang dicetak terminal:

- `result.json` dan `result.csv`: status, parameter block_resolution,
  waktu, RAM/VRAM terukur, jumlah blok dan ukuran buffer jika sudah tercapai.
- `native_mesh_runtime.json`: versi Open3D, tahap terakhir dan data alokasi.
- `mesh.log` dan `mesh_resources_*.csv`: log serta sampel resource.
- `recon.ply` dan `recon_post.ply`: hasil jika penulisan/validasi berhasil.

Buka `recon_post.ply` di folder retry baru saat status `success`. Jangan mengambil
mesh lama dari folder model asal. CSV utama di `research_results` tetap menjadi
rekaman percobaan awal; retry belum digabungkan atau mengubah statusnya.
Tanpa `--block-resolution`, wrapper memakai jalur eksekusi native awal (layout 16).
Rangda dapat diuji terpisah dengan mengganti path model setelah hasil tualen ditinjau.

Sumber pemeriksaan:
- [Kernel ekstraksi Open3D 0.19.0](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h).
- [Pengindeksan buffer Open3D 0.19.0](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/GeometryIndexer.h).
- [API VoxelBlockGrid](https://www.open3d.org/docs/0.19.0/python_api/open3d.t.geometry.VoxelBlockGrid.html).

Sumber opsi dan diagnosis:
- [Opsi data_device pada implementasi asli 3DGS](https://github.com/graphdeco-inria/gaussian-splatting#--data_device).
- [Python: exit code negatif berarti signal](https://docs.python.org/3.12/library/subprocess.html#subprocess.CompletedProcess.returncode).
- [Python faulthandler](https://docs.python.org/3.12/library/faulthandler.html).
