# Workflow aktif: training dan metrik RaDe-GS asli

Panduan ini menggantikan perintah profiling internal dan evaluasi train pada panduan
historis. README pembuat asli tidak diubah.

## Kode dan konfigurasi yang digunakan

- `scene/__init__.py`, `utils/camera_utils.py`, dan `metric.py`
  dikembalikan persis ke commit sebelum modifikasi penelitian:
  `d72f20792005ae1d6555a82aa2d15345f247604e`.
- Rumus training di `train.py` mengikuti baseline yang sama. Tambahan runtime
  `--enable_allocator_gc` dipertahankan sesuai hasil diagnosis sebelumnya.
- `mesh_extract.py` kembali ke implementasi awal, dengan satu pengecualian
  yang diminta pengguna: voxel `0.009` (aslinya 0.002). Depth tetap `8.0`.
  Depth/voxel adalah satuan koordinat rekonstruksi, belum tentu meter.
- Konfigurasi utama Togog Bali dan Tanks and Temples memakai 30000 iterasi,
  `--resolution 1920` dan `--eval`. Lebar 1920 mempertahankan rasio aspek foto;
  tinggi tidak selalu 1080.
- `--eval` memakai split bawaan COLMAP loader: setelah nama foto diurutkan,
  indeks kelipatan 8 menjadi test. Foto test tidak dipakai untuk optimasi.
  Gunakan split yang sama pada metode pembanding.
- Pipeline langsung menjalankan `python metric.py -m <output>` pada folder
  `test/`. Importer hanya memeriksa lalu menyalin angka ke laporan, tanpa
  menghitung ulang PSNR/SSIM/LPIPS. Tidak ada fallback ke foto training.
- CLI mesh awal tidak memiliki `--max_depth` atau `--voxel_size`. YAML mencatat
  nilai tetap 8.0 dan 0.009; konfigurasi lain ditolak agar laporan sesuai kode.
  Pilihan mesh tetap `-r 2 --num_cluster 8 --depth_ratio 0.5` sesuai percobaan
  manual. Ini pilihan eksperimen, bukan semua nilai default parser.
- Allocator GC aktif melalui `--enable_allocator_gc`; profiling internal
  tetap nonaktif. Pengaturan runtime sebelum import torch adalah
  `PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.8`,
  lalu `torch.cuda.set_per_process_memory_fraction(1.0)` sebelum training.
  Ini pengaturan cache/memori, bukan perubahan rumus loss atau optimisasi.
  Waktu dan Peak VRAM dapat dipengaruhi; catat pengaturannya dalam metodologi.
  Kecepatan serta hasil numerik identik tidak dijamin tanpa pengujian server.
  Antrean, log, waktu tahap,
  monitoring RAM/VRAM eksternal, dan laporan tetap tersedia. Peak VRAM seluruh
  GPU adalah maksimum sampel, termasuk aplikasi lain, bukan VRAM khusus proses.
- Open3D tetap dipin 0.19.0. Perubahan kode ini tidak memasang/menghapus paket.
  `render.py` tetap menulis manifest nama foto untuk laporan; rumus render
  tidak diubah. Tahap GT 3D aktif dan dilewati bila pembanding belum tersedia.

Saat training dimulai dengan flag aktif, log harus memuat:

```text
[ALLOCATOR] Explicit memory fraction=1.0; native GC threshold=0.8
```

Pada perintah manual tanpa flag, allocator mengikuti pengaturan environment yang
sudah ada. Flag aktif memakai konfigurasi native di atas. Jika model dijalankan
melalui pipeline, periksa dry-run: TRAIN harus memuat `--enable_allocator_gc`.
Pengaturan tidak menghapus model atau Gaussian. Hanya blok cache allocator yang
sudah bebas yang dapat direklamasi.

## Jalankan di Ubuntu-22.04-E

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

# Periksa perintah; tidak menjalankan model atau menulis hasil.
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene ngawasari --start-from training --dry-run

# Uji satu objek beserta laporan lengkap.
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene ngawasari --start-from training

# Semua Togog Bali, kemudian Tanks and Temples.
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --start-from training
```

`--resume` melewati eksperimen sukses dengan konfigurasi cocok. Perubahan split
atau parameter menghasilkan identitas eksperimen baru. Bila folder hasil sudah
berisi data, hasil baru masuk `runs/<timestamp>/`, bukan menimpa hasil lama.

## Perintah manual dengan output baru

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
scene=ngawasari
out="output/hasil_radegs_v2/togog_bali/$scene/manual_eval_$(date +%Y%m%d_%H%M%S)_$$"

python train.py -s "dataset/togog_bali/$scene" -m "$out" \
  --iterations 30000 --resolution 1920 --eval --enable_allocator_gc &&
python render.py -m "$out" &&
python metric.py -m "$out"
```

Pastikan `results.json` dan `per_view.json` berisi `ours_30000`. `metric.py`
asli menangkap exception sehingga exit code 0 saja belum membuktikan keberhasilan.
Pipeline memeriksa output tersebut secara otomatis.

Ekstraksi manual setelah model tersimpan:

```bash
python mesh_extract.py -m "$out" --iteration 30000 \
  -r 2 --num_cluster 8 --depth_ratio 0.5
```

Perintah manual asli tidak menghasilkan seluruh laporan pipeline otomatis.
Gunakan pipeline untuk pencatatan lengkap.

## Hasil dan batasan

Di folder output percobaan yang dicetak pipeline:

- `results.json`, `per_view.json`: hasil asli `metric.py`.
- `test/ours_30000/gt/` dan `renders/`: referensi dan render test.
- `research/summary.json` dan `research/image_metrics.json`: ringkasan dan sumber angka.
- `research/logs/`: log dan CSV penggunaan resource setiap tahap.
- `research/laporan_<timestamp>/`: tabel, daftar foto, dan gambar perbandingan.
- `point_cloud/iteration_30000/point_cloud.ply`: hasil Gaussian.
- `recon.ply`, `recon_post.ply`: mesh awal dan sesudah postprocessing.

Master CSV tetap mengikuti `results.csv` pada YAML. Tabel laporan berada sejajar
dengan nama `<nama_csv>_laporan.csv`.

Hasil lama dengan `eval=False` tetap merupakan evaluasi training. Jangan mengubah
cfg_args untuk menyebutnya hasil test. Training baru diperlukan untuk evaluasi
held-out. Angka train lama tidak langsung sebanding dengan angka test baru.

Pemulihan kode bukan bukti crash Open3D sudah teratasi. Walaupun voxel tetap
0.009, mesh awal menampung depth/warna semua kamera sebelum integrasi dan memakai
block_count 50000, sehingga kebutuhan memorinya dapat berbeda dari versi streaming.
Depth 8.0 juga tidak menjamin setiap scene tercakup; kamera pertama dewa_brahma
pernah memiliki depth minimum lebih dari 8. Uji satu objek dahulu.
Pengujian otomatis lokal tidak menjalankan CUDA/model.

Preset `togog_bali_1600.yaml` dan `togog_bali_full_gc.yaml` tetap menjadi alternatif
historis resolusi 1600/1024. Allocator GC juga aktif pada kedua preset tersebut,
sedangkan profiling internal nonaktif. Konfigurasi utama penelitian
tetap `togog_bali.yaml` dan `tanks_temples.yaml` pada resolusi 1920.

## Penyimpanan baru dan GT 3D opsional

Konfigurasi utama memakai pohon hasil baru:

```text
output/hasil_radegs_v2/
  togog_bali/<objek>/
  tanks_temples/<objek>/
research_results/hasil_radegs_v2/
  experiment_results_togog_bali.csv
  experiment_results_togog_bali_laporan.csv
  experiment_results_tanks_temples.csv
  experiment_results_tanks_temples_laporan.csv
```

Nama baru v2 memisahkan run ulang dari awal. Folder dan CSV sebelumnya di
output/hasil_radegs/ dan research_results/hasil_radegs/ tetap disimpan.
Untuk run pertama gunakan perintah di atas tanpa --resume. Gunakan --resume
ketika ingin melewati eksperimen yang sudah sukses pada antrean v2.

Di setiap folder objek, Gaussian tetap berada di
`point_cloud/iteration_30000/point_cloud.ply`, sedangkan mesh bernama
`recon.ply` dan `recon_post.ply`. Folder hasil dibuat ketika pipeline berjalan.
Hasil lama di `output/togog_bali` dan `output/tanks_temples` tidak dipindah atau
dihapus. CSV baru mencegah pencampuran evaluasi train lama dan test baru.
CSV baru memuat kolom status GT; CSV lama tidak ditulis ulang oleh perubahan ini.

Urutan: training -> render -> metrik foto test -> mesh -> evaluasi GT 3D -> laporan.

`geometry_evaluation` pada kedua konfigurasi utama sekarang:

```yaml
geometry_evaluation:
  enabled: true
  gt_root: dataset/ground_truth/togog_bali
  skip_missing_gt: true
  sample_points: 100000
  distance_threshold: null
  seed: 0
```

Untuk Tanks and Temples, gunakan `gt_root: dataset/ground_truth/tanks_temples`.
Path relatif dihitung dari root repositori. Letakkan hanya referensi 3D yang
memang dipakai untuk evaluasi, misalnya:

```text
dataset/ground_truth/togog_bali/ngawasari.ply
dataset/ground_truth/tanks_temples/family.ply
```

Susunan `<gt_root>/<objek>/<objek>.ply` juga didukung. Susunan datar memiliki
prioritas jika kedua file tersedia. Nama file harus sama dengan nama folder objek.
Lokasi dapat diganti melalui `gt_root`. Jangan memakai `input.ply`,
`points3D.ply` dari COLMAP, atau hasil rekonstruksi sendiri sebagai pengganti GT.

Jika pembanding tidak tersedia, hanya evaluasi geometri yang dilewati:
- `geometry.status: skipped` dan `skip_reason: missing_ground_truth`.
- Chamfer/precision/recall/F1 tetap null/kosong, bukan angka nol.
- Status proses utama tetap success bila tahap wajib lainnya berhasil.
- Alasan tersimpan pada `research/geometry_metrics.json`, `research/summary.json`,
  `research/logs/geometry_evaluation.log`, master CSV, dan tabel laporan.
- GT rusak, direktori bernama .ply, atau kegagalan evaluator dicatat sebagai failed.

Evaluasi foto tidak mengikuti kebijakan skip ini: `metric.py` tetap membutuhkan
foto test dan pasangannya di `gt/`. Foto GT dan referensi geometri 3D adalah dua
jenis data berbeda.

Tahap GT 3D ini memakai `research_pipeline.geometry_evaluator` yang sudah tersedia,
bukan `metric.py` atau protokol benchmark resmi Tanks and Temples. Rumusnya tetap:
rata-rata jarak Euclidean rekonstruksi-ke-GT ditambah rata-rata GT-ke-rekonstruksi.
Rekonstruksi dan GT harus sudah sejajar posisi, orientasi, dan skalanya.
Tidak ada registrasi, crop benchmark, atau konversi ke sentimeter secara otomatis.
`distance_threshold: null` menghasilkan Chamfer saja; precision/recall/F1 tetap
kosong sampai ambang evaluasi ditentukan dan didokumentasikan.

`--resume` tetap melewati eksperimen sukses dengan konfigurasi yang sama,
termasuk yang GT-nya sebelumnya skipped. Menambahkan GT tidak otomatis membuat
training lama dijalankan ulang. Status skipped bukan bukti kesesuaian bentuk 3D.

Dataset, GT, hasil PLY/mesh, dan tabel hasil baru tetap lokal; tidak disertakan
ke Git melalui folder hasil tersebut.

## Kolom kosong dan pemulihan Gaussian awal

Chamfer, precision, recall, dan F1 kosong ketika GT 3D belum tersedia.
Precision/recall/F1 juga tetap kosong jika distance_threshold belum ditentukan.
Status GT menjelaskan mengapa pengukuran dilewati; jangan mengisi nol.

Pembaca log menerima keluaran asli RaDe-GS seperti:
`Number of points at initialisation :  235017`.
Jika ringkasan lama belum memiliki Gaussian awal, exporter laporan membaca
kembali `research/logs/training.log` dan mencatat sumber angkanya di
`detail_laporan.json`. Angka PLY akhir tidak dipakai sebagai jumlah awal.
Jika log tidak berisi baris tersebut, kolom tetap kosong.

Setelah pembaruan kode sampai ke server, buat laporan baru untuk hasil yang sudah
selesai tanpa mengulang training, rendering, atau ekstraksi:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

python -B scripts/export_research_report.py \
  --outputs output/hasil_radegs_v2/togog_bali/dewi_gangga \
  --results-csv research_results/hasil_radegs_v2/experiment_results_togog_bali.csv
```

Gunakan path aktual run (termasuk runs/<timestamp> bila ada).
Perintah mencetak folder laporan baru berisi tabel_evaluasi.csv dan gambar
perbandingan. Opsi --results-csv juga memperbarui tabel gabungan _laporan.csv
dan _ringkasan.csv, termasuk Gaussian awal yang berhasil dibaca kembali dari log.
Baris run dicocokkan melalui Folder output; objek lain tetap tersimpan.
Laporan lama, summary.json, dan CSV pengukuran teknis asal tetap disimpan.

## Membaca laporan ringkas dan rincian proses

Mulai dari tabel **18 kolom** yang mengikuti urutan contoh laporan:
waktu ringkasan, Folder (nama objek), Metode, PSNR, SSIM, LPIPS, Peak VRAM
seluruh GPU, Training (menit), split evaluasi, jumlah gambar evaluasi, status,
jumlah foto terdaftar, foto training, foto test, Gaussian awal, splats PLY akhir,
iterasi PLY, dan folder output.

Untuk hasil pipeline otomatis, tabel ringkasan seluruh run berada di:

- `research_results/hasil_radegs_v2/experiment_results_togog_bali_ringkasan.csv`
- `research_results/hasil_radegs_v2/experiment_results_tanks_temples_ringkasan.csv`

Setiap folder `research/laporan_<timestamp>/` juga berisi:

| Berkas | Untuk membaca |
| --- | --- |
| `ringkasan_hasil.csv` | Tabel utama 18 kolom; buka ini terlebih dahulu. |
| `rincian_proses.csv` | Satu baris per tahap training, rendering, dan ekstraksi mesh: durasi, RAM, VRAM, serta lokasi sampel resource. |
| `tabel_evaluasi.csv` | Tabel lengkap dengan status GT 3D dan metrik geometri. |
| `detail_laporan.json` | Sumber angka dan data teknis laporan. |
| `PETUNJUK_LAPORAN.md` | Penjelasan file, satuan, dan arti kolom kosong. |
| `*_daftar_foto.csv` dan folder `01_<objek>/` | Daftar foto dan gambar perbandingan referensi dengan render. |

CSV teknis yang dikonfigurasi melalui `results.csv` dan tabel `_laporan.csv`
tetap tersedia. Ringkasan tambahan tidak mengganti kunci data teknis atau nilai
metrik. Pada run otomatis pertama setelah pembaruan, tabel `_ringkasan.csv` juga
memuat baris lama yang sudah ada di `_laporan.csv`.

RAM proses dan VRAM proses dinyatakan dalam MiB; total GPU dalam GiB.
RAM adalah RSS proses beserta proses anak. Total VRAM GPU mencakup aplikasi
lain dan tidak digunakan untuk mengisi VRAM proses yang tidak tersedia.
Durasi rendering terpisah dari eksekusi `metric.py`.
Kolom kosong berarti tidak tersedia; keterangan `Tidak tercatat` tidak
menyatakan tahap itu gagal. Gunakan kolom Status pada ringkasan untuk status run.
Jika waktu terlihat sebagai `########` di Excel, lebarkan kolomnya.

Untuk hasil yang sudah selesai, perintah `export_research_report.py` di atas
langsung membuat berkas laporan baru ini tanpa mengulang training.
Dengan `--results-csv`, tabel gabungan `_laporan.csv` dan `_ringkasan.csv` ikut
diperbarui. Path yang diberikan adalah nilai `results.csv` dari YAML;
CSV teknis dengan nama dasar tersebut tetap menjadi rekaman asal dan tidak
ditulis ulang oleh exporter. Tanpa opsi itu, ekspor hanya membuat laporan lokal
di folder baru. Lokasi laporan dan tabel gabungan dicetak di terminal.

Untuk Tanks and Temples, misalnya:

```bash
python -B scripts/export_research_report.py \
  --outputs output/hasil_radegs_v2/tanks_temples/horse \
  --results-csv research_results/hasil_radegs_v2/experiment_results_tanks_temples.csv
```

Gunakan nama dan path run yang memang sudah ada. Ekspor ulang run yang sama
memperbarui barisnya, tanpa menambah duplikat. Run berbeda dengan folder output
berbeda tetap memiliki baris tersendiri. Data yang tidak tersedia tetap kosong.
