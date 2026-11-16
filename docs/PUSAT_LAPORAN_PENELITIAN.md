# Pusat laporan penelitian RaDe-GS

Tabel, foto perbandingan, daftar foto, rincian waktu/RAM/VRAM dan asal hasil
sekarang dikumpulkan di samping results.csv. README pembuat asli tidak diubah.
Model Gaussian, mesh dan log sumber tetap di output eksperimen; tidak dipindah
atau disalin ulang oleh collector. Tidak ada training, render atau penghitungan
PSNR/SSIM/LPIPS ulang.

## Satu pintu masuk

Untuk konfigurasi utama, buka setelah laporan dibuat:

```text
research_results/hasil_radegs_v2/MULAI_DI_SINI.html
```

Berkas ini bisa dibuka langsung melalui browser, tanpa menjalankan server.
Dari sana pilih paket laporan Togog Bali atau Tanks and Temples.

```text
research_results/hasil_radegs_v2/
  MULAI_DI_SINI.html
  experiment_results_togog_bali*.csv
  experiment_results_tanks_temples*.csv
  laporan/
    togog_bali/
      kumpulan_<waktu>_<id>/
        index.html
        ringkasan_hasil.csv
        tabel_evaluasi.csv
        rincian_proses.csv
        detail_laporan.json
        riwayat_percobaan.csv
        pengumpulan.json
        pemulihan_mesh.csv          # jika ada retry sukses terpilih
        01_<objek>_daftar_foto.csv
        01_<objek>/
          00000_comparison.png
          ...
        02_<objek>/
          ...
    tanks_temples/
      kumpulan_<waktu>_<id>/
        ...
```

Paket otomatis per run baru berada pada lokasi yang sama dengan awalan
laporan_, sedangkan ekspor retry memakai pemulihan_. Setiap ekspor membuat
paket baru. Setelah pengulangan pipeline sukses, paket mandiri yang hanya berisi
percobaan gagal yang digantikannya dibersihkan bersama barisnya pada tabel aktif.
Paket sukses dan kumpulan ekspor yang mencampur beberapa hasil tetap tersimpan
sebagai snapshot. Buat kumpulan baru untuk membaca keadaan hasil terkini.

| Berkas | Gunakan untuk |
| --- | --- |
| index.html | Membaca tabel dan melihat pasangan foto di satu halaman. |
| ringkasan_hasil.csv | Tabel utama 20 kolom, termasuk VRAM seluruh GPU dan VRAM khusus proses training. |
| tabel_evaluasi.csv | Tabel lengkap, termasuk status GT. |
| rincian_proses.csv | Waktu dan RAM/VRAM per tahap; yang tidak tersedia tetap kosong. |
| 01_<objek>/, 02_<objek>/, ... | PNG perbandingan, referensi di kiri dan render di kanan. |
| *_daftar_foto.csv | Nama kamera, split, iterasi, ukuran, dan pasangan file. |
| pemulihan_mesh.csv | Parameter blok/voxel/depth, durasi dan sumber mesh retry sukses. |
| riwayat_percobaan.csv | Run awal, nested runs, retry gagal/sukses, dan penanda hasil yang terpilih. |
| detail_laporan.json | Sumber training/render, sumber metrik dan mesh; lokasi sumber tetap tercatat. |
| pengumpulan.json | Aturan pemilihan, identitas hasil terpilih dan catatan yang tidak dapat diverifikasi. |

## Kumpulkan hasil yang sudah selesai di server

Jalankan dari checkout asli pada Ubuntu-22.04-E, tempat output lengkap berada.
Perbarui kode pada server lebih dahulu. Tidak perlu mengulang training atau mesh.

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS

# Lihat hasil yang akan dikumpulkan; tidak membuat atau mengubah file.
python -B scripts/collect_research_reports.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --dry-run

# Buat paket lengkap, default lima pasangan visual per objek.
python -B scripts/collect_research_reports.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --pairs 5
```

Ganti --pairs jika memerlukan jumlah pasangan yang berbeda. Foto dipilih merata
berdasarkan urutan file, bukan kualitas hasil. Semua foto test tetap digunakan
untuk metrik yang sudah dihitung; jumlah contoh visual tidak mengubah metrik.

Untuk satu dataset saja, tuliskan satu YAML. Konfigurasi Scaffold dapat
dikumpulkan dengan YAML-nya sendiri; output dan CSV perlakuan tetap terpisah,
 mengikuti results.csv pada konfigurasi tersebut.

Collector membaca summary.json pada output/<dataset>/<objek>/ dan
output/<dataset>/<objek>/runs/<id>/, kemudian memeriksa mesh_retry_* di
research masing-masing model. Folder analisa radegs bukan sumber otomatis:
salinan lama yang tidak mempunyai metadata asli tidak dicampurkan ke hasil test.

Aturan pemilihan paket gabungan:

1. Kelompokkan berdasarkan nama objek dan metode yang tercatat.
2. Jika ada hasil sukses, pilih run sumber terbaru yang memiliki hasil sukses.
3. Untuk sumber itu, gunakan retry sukses terbaru yang lolos validasi jika ada.
4. Jika belum ada hasil sukses, tampilkan kegagalan terakhir agar objek tidak hilang.
5. Semua catatan lain tetap ada di riwayat_percobaan.csv. Catatan rusak, retry
   belum selesai, atau tidak cocok dengan sumber tidak dipromosikan menjadi sukses.

Ini pemilihan waktu/run, bukan pemilihan hasil dengan PSNR terbaik. Perbedaan
konfigurasi seperti data_device CPU pada Detya tetap bagian dari catatan sumber;
collector tidak menyatakan semua pengaturan eksperimen identik. Jika beberapa
percobaan suatu objek memang perlu dibandingkan, ekspor output tersebut secara
eksplisit melalui export_research_report.py agar seluruhnya masuk satu paket.

## Laporan otomatis untuk run berikutnya

Pipeline utama otomatis menaruh paket baru di pusat laporan berdasarkan
results.csv. Gunakan perintah pipeline yang sudah ada; tidak ada flag model baru.

Ekspor hasil tertentu juga dapat memakai pusat yang sama:

```bash
python -B scripts/export_research_report.py \
  --outputs output/hasil_radegs_v2/tanks_temples/horse \
  --results-csv research_results/hasil_radegs_v2/experiment_results_tanks_temples.csv
```

Untuk mesh retry, gunakan --mesh-retries seperti panduan pemulihan. Tanpa
--destination, ekspor yang memakai --results-csv masuk ke pusat laporan.
--destination tetap tersedia untuk memilih folder paket secara eksplisit.
Ekspor tanpa --results-csv mempertahankan penempatan manual dekat output sumber.

CSV teknis dasar tetap menyimpan catatan pipeline awal; tabel _ringkasan.csv,
_laporan.csv dan _pemulihan_mesh.csv diperbarui sebagai tampilan laporan.
Paket kumpulan berisi satu hasil terpilih per objek/metode; tabel gabungan lama
masih mempertahankan riwayat dan dapat mempunyai beberapa baris per objek.

## Menyalin laporan untuk dianalisis

Salin seluruh research_results/hasil_radegs_v2/ bila memerlukan pusat lengkap,
atau satu folder kumpulan_<waktu>_<id>/ bila hanya memerlukan satu dataset.
Buka index.html di paket setelah disalin. Tautan foto/CSV di dalam paket relatif
sehingga masih berfungsi pada komputer lain. Lokasi sumber pada JSON adalah
jejak eksperimen server, bukan klaim bahwa model/log ikut berada di paket.

Paket ini tidak menyertakan PLY, mesh besar, checkpoint, atau seluruh render.
Cadangkan output asli secara terpisah untuk reproduksibilitas dan pemeriksaan
geometri. Render/GT dan summary asli harus masih tersedia ketika collector
membuat PNG perbandingan. Data yang hilang ditandai belum tersedia.

Peak VRAM seluruh GPU (GiB) tetap maksimum sampel seluruh GPU selama training,
termasuk aplikasi lain. Dua kolom tambahan peak_vram_mb dan average_vram_mb
mengukur puncak dan rata-rata sampel VRAM PID training RaDe-GS beserta proses
anaknya, dalam MiB (1024**2 byte). Rendering, ekstraksi mesh, dan training
Scaffold-GS berada pada baris tahap terpisah di rincian_proses.csv.

Rata-rata memakai sampel valid pada interval resource_monitor.interval_seconds
(default konfigurasi utama: 5 detik). Sampel proses yang diulang untuk beberapa
GPU dihitung sekali per waktu sampel. Kedua angka merupakan statistik sampel;
lonjakan di antara sampel bisa terlewat. Rumus training tetap sama.

Data berasal dari training pada summary.json atau CSV resource training.
Ekspor ulang dapat memakai sampel proses lama yang masih tersedia, tanpa training
ulang. CSV gabungan dengan format 18 kolom otomatis mendapat dua kolom baru;
baris dan angka lama dipertahankan. Paket laporan historis tetap tersimpan.

Jika driver melaporkan VRAM per proses sebagai N/A, kedua kolom tetap kosong,
bukan nol dan bukan diganti VRAM perangkat atau allocator PyTorch. Data proses
yang dahulu tidak direkam tidak dapat dihitung dari Peak VRAM seluruh GPU.
Periksa process_vram_status pada CSV resource dan memory.process_vram_source
di detail_laporan.json untuk sumber angka.
Mesh recovery tetap berstatus success_mesh_recovered, dan GT mesh baru tidak
dihitung oleh proses pengumpulan ini. Evaluasi train tidak diberi label test.

## Memperbesar tulisan foto perbandingan

PNG baru memakai judul besar yang mengikuti lebar foto, nama objek, split,
iterasi dan nama file pada header tersendiri. Header menyesuaikan tinggi teks;
nama panjang dibungkus agar tetap berada di panel yang benar. Foto referensi
dan render dipertahankan pada resolusi dan piksel RGB aslinya.

Untuk memperbarui PNG yang sudah telanjur memakai label kecil, perbarui kode
server lalu jalankan kembali collector yang sama. Tidak perlu training,
rendering, ekstraksi mesh atau penghitungan metrik ulang.

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
python -B scripts/collect_research_reports.py \
  --config configs/togog_bali.yaml configs/tanks_temples.yaml \
  --pairs 5
```

Buka paket kumpulan_ terbaru melalui MULAI_DI_SINI.html. PNG lama tetap ada di
paket sebelumnya. Jika hanya membutuhkan satu dataset, gunakan satu YAML.
Ukuran font dan tinggi header setiap gambar tercatat dalam label_layouts pada
detail_laporan.json. Pillow minimal 10.1 diperlukan untuk font scalable bawaan;
requirements-research.txt mencatat batas versi tersebut. Font sistem dipakai
jika tersedia, dengan fallback lokal tanpa unduhan font.
