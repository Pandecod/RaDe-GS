# Panduan run penuh Togog Tualen dengan allocator GC

> Status penataan 26 Oktober 2026: dokumen ini menyimpan konfigurasi dan prosedur
> run penuh yang sudah dipakai. Model 30.000 iterasi sudah tersimpan di server,
> dan mesh manual berhasil dengan mesh_extract_tnt.py + Open3D 0.19.0. Jangan mengulang training
> untuk melanjutkan diagnosis mesh. Lihat [Peta kode aktif](PETA_KODE_AKTIF.md).
> Pembaruan 27 Oktober 2026: pipeline sudah memakai mesh_extract_tnt.py dengan
> parameter run manual yang berhasil. Lihat [integrasi dan penyalinan](PIPELINE_GC_TNT.md)
> serta [panduan pemulihan](PANDUAN_PEMULIHAN_TRAINING_DAN_MESH.md).


## Keputusan eksperimen

Lanjutkan persiapan satu run dari awal sampai 30.000 iterasi setelah dry-run server
valid dan pengguna memberikan persetujuan terpisah. Tidak perlu mengulang uji
3.000 iterasi yang sama: uji allocator sebelumnya sudah merekam pelepasan cache
dan menghilangkan perlambatan pada rentang 2.600-2.999.

Run penuh tetap merupakan validasi baru. Regularisasi mulai pada iterasi 7.000;
densification dijadwalkan selama iterasi <15.000. Keberhasilan pada 3.000 belum
menjamin kestabilan bagian tersebut atau kesetaraan kualitas rekonstruksi akhir.

Jangan menjalankan GPU, CUDA, training, benchmark, instalasi atau SSH dari laptop
untuk mengikuti pemeriksaan lokal ini. Seluruh perintah Linux di bawah adalah
petunjuk untuk pengguna pada server, bukan perintah yang dijalankan oleh Codex.

## A. File dan audit

Root lokal:
C:/Users/user/Downloads/00_Baru/Download_Baru/RaDe-GS/RaDe-GS

Root server:
/home/pande/projects/RaDe-GS

| File di root lokal | Tujuan server | Status dan alasan |
|---|---|---|
| train.py | /home/pande/projects/RaDe-GS/train.py | Mengganti versi lama; menambahkan opsi --enable_allocator_gc yang default-nya nonaktif. |
| configs/togog_bali_full_gc.yaml | /home/pande/projects/RaDe-GS/configs/togog_bali_full_gc.yaml | Konfigurasi eksperimen baru, terpisah dari configs/togog_bali.yaml. |
| docs/togog_tualen_full_gc_runbook.md | /home/pande/projects/RaDe-GS/docs/togog_tualen_full_gc_runbook.md | Panduan baru; penyalinan ke server opsional. |

train.py mengatur PYTORCH_CUDA_ALLOC_CONF sebelum mengimpor torch ketika opsi
diaktifkan. Nilainya native dengan garbage_collection_threshold:0.8.
Setelah parsing argumen dan sebelum safe_state(), Scene atau loop training,
memory fraction disetel satu kali ke 1.0. Nilai yang dilaporkan ikut disimpan
di profile_options pada manifest profiler. Tidak ada empty_cache baru.

Backup train.py sebelum perubahan ada di:
.profiling_backup/main_allocator_gc_20261025T235526Z/train.py

Fungsi training dan perhitungan model tidak diubah. commands.py meneruskan
training.extra_args ke train.py; config.py membaca deklarasi CLI secara statis,
sehingga opsi baru dikenali tanpa mengimpor torch. run_all.py memilih scene
melalui --scene. runner.py menjalankan training, rendering, evaluasi gambar,
dan mesh secara berurutan.

TrainingProfiler masih merekam Gaussian sebelum/sesudah densification, clone,
split, pruning, num_rendered, visible Gaussian, radius, ukuran buffer,
timing fase, memori allocated/reserved/peak, identitas run dan status rasterizer.
num_rendered adalah jumlah instance Gaussian-tile. Snapshot/allocator_stats
khusus launcher diagnostik tidak otomatis ditambahkan pada run utama.

GPU monitor Windows yang telah berhasil tetap memakai:
scripts/monitor_windows_gpu.ps1
File ini tidak diubah dalam integrasi run utama. Tidak perlu menyalinnya ulang
jika versi server sudah sama.

## B. Konfigurasi dan pemindahan

Konfigurasi baru menetapkan:
- Iterasi 30.000 dari awal, tanpa start_checkpoint.
- Lebar training 1024; untuk gambar Tualen yang telah diperiksa menjadi 1024x570.
  Ini setara dengan auto-resize 1024 dalam kode baseline untuk dataset tersebut.
- --enable_allocator_gc aktif di YAML ini dan configs/togog_bali.yaml.
- Profiling iterasi 1800-30000 setiap 100 iterasi dan sampel kontrol.
- Telemetry GPU setiap 1 detik.
- Checkpoint .pth pada 3000, 7000, 15000, 25000.
- Penyimpanan Gaussian/evaluasi bawaan pada 7000 dan 30000 tetap berlaku.
- Rendering, metrik gambar, dan mesh aktif; geometry evaluation nonaktif.
- Mesh memakai mesh_extract_tnt.py -r 2 --num_cluster 8 --depth_ratio 0.5
  --voxel_size 0.009 --max_depth 8.0, dengan Open3D 0.19.0.
- Output: output/togog_bali/full_gc/togog_tualen.
- CSV: research_results/experiment_results_full_gc.csv.

Folder output/togog_bali dipertahankan sesuai preferensi pengguna. Contoh path
Rade-GS_results pada rencana bukan keharusan, sehingga tidak dipakai di sini.

Konfigurasi dataset menunjuk direktori induk. Pemilihan hanya Tualen dilakukan
oleh --scene togog_tualen. Selalu gunakan command lengkap di panduan ini;
menjalankan run_all.py tanpa --scene akan memilih semua scene.

Sebelum mengganti train.py di server, cadangkan versi server dari terminal WSL:

```bash
cd /home/pande/projects/RaDe-GS
cp -n train.py "train.py.before_allocator_gc_$(date +%Y%m%d_%H%M%S)"
```

Salin train.py dan configs/togog_bali_full_gc.yaml secara manual dari lokal ke
tujuan pada tabel A. Tidak perlu menyalin seluruh repository, dataset, COLMAP,
checkpoint lama atau launcher diagnostik.

Jika output scene sudah terisi, runner membuat subfolder runs/<waktu>_<id>.
Dry-run menampilkan lokasi dasar; lokasi aktual ditampilkan saat run dimulai.
Hasil lama tidak dihapus. CSV baru terpisah dari CSV eksperimen sebelumnya.

## C. Pemeriksaan server tanpa training

Di environment radegs, jalankan:

```bash
cd /home/pande/projects/RaDe-GS
python -B -c "import ast; from pathlib import Path; ast.parse(Path('train.py').read_text(encoding='utf-8')); print('Sintaks train.py OK')"
python -B research_pipeline/run_all.py --config configs/togog_bali_full_gc.yaml --scene togog_tualen --start-from training --dry-run
df -h .
```

Dry-run tidak menjalankan subprocess GPU, membuat output, atau mengimpor torch.
Pastikan hasilnya menunjukkan:
- Found 1 datasets dan [1/1] togog_tualen.
- Start from: training; tidak ada tahap COLMAP.
- TRAIN memuat --enable_allocator_gc --iterations 30000 --resolution 1024.
- Profiling 1800-30000, interval 100, telemetry 1.
- Checkpoint 3000 7000 15000 25000; tidak ada start_checkpoint.
- RENDER, IMAGE METRICS dan MESH dengan parameter di atas.

Satu checkpoint pada dua juta Gaussian sebelumnya berukuran sekitar 1,39 GiB.
Empat checkpoint seukuran itu saja membutuhkan sekitar 5,6 GiB. Ukuran aktual
dapat bertambah mengikuti Gaussian. Sisakan ruang tambahan untuk dua PLY model,
gambar render/GT, mesh, logs dan data sementara. Ruang bebas 30 GiB atau lebih
merupakan persiapan praktis, bukan jaminan cukup untuk semua pertumbuhan model.

Checkpoint memakai gaussians.capture() beserta optimizer.state_dict() dan nomor
iterasi, jadi mendukung kelanjutan training. Kelanjutan tidak dijamin identik
bit demi bit: state RNG dan beberapa akumulator tidak seluruhnya tersimpan.
Pipeline --resume berarti melewati eksperimen yang sudah sukses, bukan otomatis
melanjutkan dari checkpoint.

Jangan menguji loading checkpoint menggunakan torch pada laptop. Untuk
checkpoint buatan sendiri yang dipercaya, masalah default weights_only PyTorch
2.6 dapat ditangani nanti dengan prosedur yang sebelumnya berhasil; run baru
ini tidak memuat checkpoint.

## D. Command setelah persetujuan run penuh

Setelah dry-run diperiksa dan pengguna menyetujui, jalankan monitoring Windows
pada terminal pertama di WSL server:

```bash
cd /home/pande/projects/RaDe-GS
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
  -NoProfile -ExecutionPolicy Bypass \
  -File "$(wslpath -w "$PWD/scripts/monitor_windows_gpu.ps1")"
```

Catat lokasi CSV Windows yang ditampilkan. Biarkan terminal pertama berjalan.
Terminal yang diam tanpa warning berulang adalah perilaku normal monitor.

Pada terminal kedua, dalam environment radegs, satu command eksperimen penuh:

```bash
python -B research_pipeline/run_all.py --config configs/togog_bali_full_gc.yaml --scene togog_tualen --start-from training
```

Jalankan dari /home/pande/projects/RaDe-GS. Command memulai training 30.000 iterasi,
kemudian rendering, metrik gambar, dan mesh. Jangan gunakan launcher
arsip_lokal/20261026_kode_percobaan/scripts/diagnose_allocator.py untuk run ini karena launcher tersebut dibatasi 3.000 dan sudah diarsipkan.

Eksperimen ini belum dijalankan oleh Codex. Menyiapkan file bukan persetujuan
untuk menjalankan training penuh.

## E. Monitoring

Gunakan lokasi output aktual yang dicetak runner/training. Contoh di bawah
berlaku bila output pertama memakai folder dasar dan bukan subfolder runs.

```bash
pgrep -af 'python.*train[.]py'
tail -n 15 output/togog_bali/full_gc/togog_tualen/research/logs/training.log
find output/togog_bali/full_gc/togog_tualen/research/profiling -maxdepth 1 -type f
find output/togog_bali/full_gc/togog_tualen -maxdepth 1 -name 'chkpnt*.pth' -ls
```

Log awal harus memuat:
[ALLOCATOR] Explicit memory fraction=1.0; native GC threshold=0.8

Manifest profiler harus mencatat environment allocator yang benar,
enable_allocator_gc=true dan allocator_memory_fraction=1.0 dalam profile_options.
CSV training/gpu harus bertambah selama proses berlangsung.
CUDA reserved tidak identik dengan dedicated memory fisik Windows.

Checkpoint akan muncul sesuai milestone. Perhatikan kecepatan/memori pada 3000,
7000, 15000 dan 25000. Kenaikan waktu setelah 7000 dapat berasal dari regularisasi
yang memang baru aktif; evaluasi timing fase sebelum menyebutnya masalah cache.

Setelah training, runner menulis log:
- research/logs/rendering.log
- research/logs/image_metrics.log
- research/logs/mesh_extraction.log

Ringkasan akhir ada di research/summary.json dan CSV eksperimen. Mesh yang
dianalisis pipeline adalah recon_post.ply pada folder output run.
Jika suatu tahap gagal, lihat failed_stage/error_message; jangan menghapus hasil.
Metrik gambar yang gagal dapat dicatat sementara mesh tetap dicoba.
Resource guard dapat menunda/menolak mesh jika memori bebas belum cukup.

Hentikan monitoring Windows dengan Ctrl+C setelah seluruh pipeline selesai dan
kembali ke prompt, termasuk rendering dan mesh. Cocokkan kolom utc antara CSV
Windows dan profiler; waktu yang ditambahkan pada tampilan terminal adalah lokal.

## F. Keberhasilan dan kualitas

Periksa secara bersama:
- Training selesai 30.000 tanpa OOM.
- Tidak kembali mengalami perlambatan ekstrem akibat memori seperti baseline.
- Allocated/reserved serta dedicated/shared GPU memory tetap terkendali.
- Rendering, metrik gambar dan mesh selesai; summary menyatakan sukses.
- Render dari kamera yang sama dan mesh dapat diperiksa untuk detail, lubang,
  noise dan permukaan.

Kecepatan dan batas memori yang wajar setelah 7000/15000 tidak harus identik
dengan rentang 2600-2999; beban algoritma berubah sesuai jadwal baseline.

Gunakan Togog Tualen dengan konfigurasi dan iterasi yang sama sebagai pembanding.
Jangan memakai Togog BG sebagai pembanding kualitas langsung.
Dataset baseline sebelumnya mempunyai 377 kamera train dan 0 test; pipeline
memakai test bila tersedia, jika tidak memakai train. PSNR/SSIM/LPIPS pada train
tidak mengukur generalisasi ke sudut baru. Geometry evaluation tetap nonaktif
tanpa geometri ground truth. Jangan menjanjikan kualitas akhir identik.

## Rollback

Opsi allocator default nonaktif. Untuk membandingkan baseline, gunakan config
terpisah tanpa --enable_allocator_gc; jangan menimpa run ini. Jika memakai shell
dengan PYTORCH_CUDA_ALLOC_CONF yang disetel manual, hapus pengaturan shell itu
juga ketika memang menginginkan allocator default.

Untuk mengembalikan kode secara penuh, salin backup train.py yang sesuai
setelah memastikan tidak ada perubahan lain yang perlu dipertahankan.
Simpan konfigurasi baru dan semua output eksperimen sebagai arsip; tidak ada
kebutuhan menghapus data.

## Validasi lokal dan batasnya

Pemeriksaan lokal mencakup sintaks Python, YAML dan parser CLI aktual, urutan
aktivasi allocator, default nonaktif, parameter optimisasi yang tidak berubah,
konstruksi empat tahap command dan dry-run dengan discovery dataset server
disimulasikan. Subprocess dicegah dan torch tidak diimpor pada pemeriksaan ini.
Pemilihan scene nyata, kapasitas disk server dan eksekusi CUDA belum diuji.

Folder output unik diverifikasi melalui logika allocate_output yang sudah ada.
Checkpoint model/optimizer diverifikasi dari kode capture/save, bukan loading
berkas .pth dengan GPU. Monitoring Windows telah diuji oleh pengguna pada run
diagnostik sebelumnya, dan skripnya dipertahankan.
