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
Ini percobaan pemulihan; keberhasilannya belum diuji pada GPU server.

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

Sumber opsi dan diagnosis:
- [Opsi data_device pada implementasi asli 3DGS](https://github.com/graphdeco-inria/gaussian-splatting#--data_device).
- [Python: exit code negatif berarti signal](https://docs.python.org/3.12/library/subprocess.html#subprocess.CompletedProcess.returncode).
- [Python faulthandler](https://docs.python.org/3.12/library/faulthandler.html).
