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

## Pemulihan tualen dan rangda: blok voxel 8

Diagnosis server tanggal 5 November 2026 menunjukkan:

- Kedua objek crash pada `mesh_extract.py:85`, yaitu `vbg.extract_triangle_mesh()`.
- Tualen: 41.578 blok aktif, kapasitas 50.000, RAM proses 7,67 GiB,
  RAM tersedia 48,47 GiB tepat sebelum ekstraksi.
- Pada Open3D 0.19.0, buffer bantu berisi empat Int32 per voxel.
  Untuk blok 16, buffer ini mencapai 2.537720 GiB pada 41.578 blok.
  Perhitungan byte offset 32-bit merupakan dugaan kuat, belum konfirmasi
  stack C++ atau hasil uji perbaikan pustaka. Jumlah blok pada layout 16
  belum diukur untuk rangda.

Opsi retry `--block-resolution 8` mengubah ukuran kelompok voxel dalam memori,
dari 16 x 16 x 16 menjadi 8 x 8 x 8. Jarak antarvoxel tetap 0.009,
depth tetap 8.0, mesh tetap -r 2, dan PLY training 30000 yang sama dipakai.
Block resolution berbeda dari resolution gambar dan ukuran voxel.

Implementasinya berupa launcher `research_pipeline/native_mesh_runtime.py`
yang menjalankan `mesh_extract.py` yang ada, dengan override constructor
VoxelBlockGrid hanya untuk proses retry tersebut. Integrasi dan pembentukan
segitiga tetap dipanggil pada Open3D. Source `mesh_extract.py`, kode training,
renderer, metric.py, YAML utama, dan hasil asal tetap seperti sebelumnya.

Run server pada 5 November 2026 berhasil mengekstrak tualen dan rangda dengan
block_resolution=8; ringkasan dan foldernya dicatat di bawah. Mengubah ukuran
blok dapat mengubah blok yang dialokasikan saat integrasi. Jangan mengklaim mesh pasti
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
Untuk rangda, gunakan perintah retry yang sama dengan path model
`output/hasil_radegs_v2/togog_bali/rangda`. Hasil percobaan yang sudah
berhasil berikut tidak perlu diulang hanya untuk memperoleh catatannya.

### Hasil server yang berhasil, 5 November 2026 (UTC+08)

Sumber angka adalah log terminal yang diberikan pengguna, bukan pengukuran ulang
oleh komputer lokal. Vertex di tabel ini merupakan vertex mesh, bukan jumlah
Gaussian pada PLY training.

| Objek | Status retry | Block resolution | Blok aktif | Kapasitas blok | Buffer bantu (GiB) | Vertex mentah | Vertex pascaproses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tualen | success | 8 | 213290 | 400000 | 1.627274 | 5704117 | 5418957 |
| rangda | success | 8 | 174622 | 200000 | 1.332260 | 4719960 | 4397482 |

Kedua percobaan memakai model 30000 yang sudah selesai, mesh_extract.py melalui
launcher pemulihan, resolusi mesh 2, cluster 8, depth ratio 0.5, voxel 0.009,
max depth 8.0, serta Open3D 0.19.0. Training tidak dijalankan ulang.

Folder hasil, relatif terhadap `/home/pande/projects/RaDe-GS/`:

```text
output/hasil_radegs_v2/togog_bali/tualen/research/mesh_retry_20261104T225758Z_743ee5a1/
output/hasil_radegs_v2/togog_bali/rangda/research/mesh_retry_20261104T230358Z_78370698/
```

Gunakan `recon_post.ply` dari folder masing-masing untuk mesh akhir,
`recon.ply` untuk mesh mentah, dan `result.json`/`result.csv` untuk detail
percobaannya. Nama folder memakai UTC, sehingga tanggal 20261104 pada nama
folder sesuai dengan log pagi 5 November 2026 pada UTC+08.

Pada server, `test_native_mesh_runtime` menjalankan 11 tes dengan hasil `OK`,
termasuk ekstraksi bidang sintetis pada layout 16 dan 8. Baris `failed` di dalam
keluaran tes berasal dari skenario kegagalan yang sengaja disimulasikan; status
retry sebenarnya dibaca pada proses setelah ringkasan tes.

Peringatan rangda `Write Ply clamped color value to valid range` muncul saat
penulisan mesh mentah dan akhir. Penulis PLY Open3D mendeteksi komponen warna
di luar rentang [0, 1], kemudian mengonversinya ke warna UInt8 dengan pembatasan
nilai. Cabang ini mengolah warna, bukan mengubah koordinat vertex atau indeks
segitiga. Kedua penulisan dilanjutkan dan wrapper mencatat `success`.
Jumlah vertex yang terdampak atau besar koreksi warna tidak tersedia dalam log;
periksa juga warna saat membuka PLY.
[Sumber penulis PLY Open3D 0.19.0](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/io/file_format/FilePLY.cpp#L608).

Dua keberhasilan ini membuktikan opsi pemulihan bekerja pada percobaan tersebut.
Ini belum membuktikan kesetaraan mesh terhadap layout 16 atau penyebab C++ yang
pasti. Untuk pelaporan penelitian, pisahkan status percobaan asal yang gagal
dengan retry sukses dan cantumkan block_resolution=8. Tabel utama
`research_results` belum digabungkan otomatis oleh wrapper retry.

Sumber pemeriksaan:
- [Kernel ekstraksi Open3D 0.19.0](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h).
- [Pengindeksan buffer Open3D 0.19.0](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/GeometryIndexer.h).
- [API VoxelBlockGrid](https://www.open3d.org/docs/0.19.0/python_api/open3d.t.geometry.VoxelBlockGrid.html).

Sumber opsi dan diagnosis:
- [Opsi data_device pada implementasi asli 3DGS](https://github.com/graphdeco-inria/gaussian-splatting#--data_device).
- [Python: exit code negatif berarti signal](https://docs.python.org/3.12/library/subprocess.html#subprocess.CompletedProcess.returncode).
- [Python faulthandler](https://docs.python.org/3.12/library/faulthandler.html).

## Memasukkan hasil retry ke laporan research_results

Hasil yang sudah selesai dapat digabungkan tanpa mengulang training, rendering,
metric.py, atau ekstraksi mesh. Gunakan exporter baru dengan `--mesh-retries`.
Semua input dibaca dari file hasil run; tidak ada nilai eksperimen yang dibuat ulang.

Setelah pembaruan kode tersedia di server, jalankan:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B scripts/export_research_report.py \
  --outputs output/hasil_radegs_v2/togog_bali/detya/runs/20261104T175137_fb0014f833 \
  --mesh-retries \
    output/hasil_radegs_v2/togog_bali/tualen/research/mesh_retry_20261104T225758Z_743ee5a1 \
    output/hasil_radegs_v2/togog_bali/rangda/research/mesh_retry_20261104T230358Z_78370698 \
  --results-csv research_results/hasil_radegs_v2/experiment_results_togog_bali.csv
```

`--outputs` mengekspor ulang laporan Detya yang sudah sukses. `--mesh-retries`
menggabungkan catatan training dan evaluasi asal Tualen/Rangda dengan data
mesh dari retry sukses yang dipilih. Opsi `--outputs` boleh dihilangkan bila
hanya ingin memasukkan kedua retry; baris Detya yang sudah ada tetap dipertahankan.

Exporter memastikan sumber training/evaluasi cocok dengan snapshot saat retry,
iterasi model sesuai, dan header vertex/face serta ukuran file mesh sesuai
catatan retry yang sudah divalidasi. Ini pemeriksaan header/ukuran file,
bukan pengujian ulang seluruh payload atau penilaian kualitas visual.
Pertahankan folder asal, `research/summary.json`, Gaussian PLY, `result.json`,
dan `recon_post.ply` retry di lokasi server yang tercatat.

Buka berkas berikut di `research_results/hasil_radegs_v2/`:

| File | Isi yang diperbarui |
| --- | --- |
| `experiment_results_togog_bali_ringkasan.csv` | Ringkasan utama 18 kolom; baris lama dipertahankan dan hasil retry ditambahkan. |
| `experiment_results_togog_bali_laporan.csv` | Tabel lengkap, termasuk status GT. |
| `experiment_results_togog_bali_pemulihan_mesh.csv` | Sumber run/retry, blok 8, voxel/depth, durasi mesh, RAM/VRAM, vertex/face, jumlah blok dan ukuran buffer. |
| `laporan_pemulihan_<waktu>_<id>/` | Laporan gabungan baru, rincian proses, detail JSON, daftar foto, dan gambar perbandingan jika render/referensi asal tersedia. |

Arti status untuk pemilihan data laporan:

- `success`: run pipeline yang berhasil seperti Detya.
- `success_mesh_recovered`: training dan evaluasi asal dipakai bersama hasil
  retry mesh yang berhasil. Ini baris pemulihan terpisah, bukan perubahan
  riwayat run asal menjadi sukses.
- `failed`: percobaan gagal sebelumnya tetap tersimpan.

Untuk memasukkan hasil berhasil ke analisis, pilih `success` dan
`success_mesh_recovered`, lalu catat konfigurasi pemulihan pada metodologi.
Metrik foto tidak dihitung ulang atau diganti split-nya; foto perbandingan
tetap berasal dari rendering model asal. Vertex mesh tidak digunakan sebagai
pengganti jumlah Gaussian pada kolom Splats PLY akhir.

CSV teknis dasar `experiment_results_togog_bali.csv`, summary asal dan result.json
retry tidak diubah oleh ekspor. Karena itu `--resume` pipeline utama tidak
otomatis melewati baseline gagal hanya karena baris pemulihannya sudah diekspor.
Menjalankan ekspor lagi memperbarui baris dengan Folder output yang sama,
tanpa duplikasi. Input retry gagal, belum selesai, berubah sumbernya, atau
kehilangan mesh akan ditolak.

GT 3D untuk mesh pemulihan tidak dihitung oleh perintah ekspor ini.
Statusnya `not_run` dengan alasan `mesh_retry_not_evaluated` (atau tetap
`disabled` jika evaluasi itu dinonaktifkan sejak awal). Nilai GT dari mesh lama
tidak disalin sebagai hasil mesh baru. Kolom data yang tidak tercatat tetap kosong.
