# Profiling diagnostik training RaDe-GS

Implementasi ini hanya menambahkan observasi. Loss, sampling kamera, densification,
pruning, rasterization, learning rate, resolusi default dan target iterasi tidak
diubah. Profiling default NONAKTIF. Tidak memerlukan perubahan/build ulang ekstensi
CUDA. Tidak ada training yang dijalankan saat validasi CPU.

## Mengaktifkan

Tambahkan ke command training yang sudah digunakan:

~~~bash
--profile_training --profile_interval 100 --profile_from 1000 --profile_until 5000
~~~

Pilihan tambahan:

| Argumen | Default | Arti |
|---|---|---|
| --profile_training | false | Aktifkan instrumentasi |
| --profile_interval | 100 | Sampel pada kelipatan interval |
| --profile_from | 1 | Batas awal pencatatan |
| --profile_until | 0 | Batas akhir pencatatan; 0 berarti tanpa batas |
| --profile_control_samples | false | Tambah sampel berikutnya, misalnya 2100 dan 2101 |
| --profile_no_raster | false | Lewati pembacaan context dan statistik radius |
| --profile_gpu_interval | 1.0 | Interval detik monitoring GPU; 0 menonaktifkan sampling eksternal |

--profile_until membatasi LOGGING, BUKAN menghentikan training. Tanpa perubahan
command lain, target training tetap default 30000 iterasi. Untuk eksperimen
diagnostik yang kelak disetujui, hentikan dengan Ctrl+C setelah jendela pengamatan
terpenuhi. Jangan menjalankan command eksperimen sebelum mendapat persetujuan.

Menghapus --profile_training menonaktifkan event/statistik/thread profiling.
Monitoring bawaan research_pipeline tetap berjalan ketika pipeline digunakan.

Untuk pipeline, argumen yang sama dapat diteruskan melalui training.extra_args
pada salinan konfigurasi diagnostik. Konfigurasi utama tidak perlu diedit.
Runner menghindari duplikasi query GPU ketika --profile_training aktif.
--profile_gpu_interval 0 menonaktifkan query GPU eksternal untuk training tersebut,
termasuk query GPU duplikat dari runner; RAM runner tetap dicatat.

## Lokasi hasil

Pipeline memakai output.root: output/togog_bali, relatif terhadap root repository.
Untuk dataset Tualen, hasil pertama berada pada:

~~~text
output/
  togog_bali/
    togog_tualen/
      research/
        profiling/
~~~

Percobaan ulang disimpan pada togog_tualen/runs/<run_id> oleh pipeline.
Untuk pemanggilan train.py langsung, tentukan -m output/togog_bali/<nama_run>.
Hasil lama tidak dipindahkan atau ditimpa.

Data tambahan profiler hanya CSV/JSON metrik serta metadata reproduksi.
Profiler tidak menyalin seluruh gambar, tensor Gaussian, atau buffer GPU ke file.
Waktu fase, penghitung Gaussian, memori dan ringkasan rasterizer dicatat per 100
iterasi; telemetry NVIDIA default satu detik sekali pada thread terpisah.
Manifest dan daftar kamera ditulis sekali per run.

Profiling tetap menambah overhead. Besarnya belum diukur pada GPU nyata.
CUDA events dipoll tanpa sinkronisasi paksa selama loop; statistik radius
dihitung setelah event akhir iterasi. Untuk menilai dampak terhadap durasi total,
bandingkan run dengan dan tanpa profiling pada kondisi yang sama.

## Artefak dan ketahanan saat dihentikan

Di bawah model_path/research/profiling, setiap pemanggilan mendapat UTC + UUID:

- training_<run_id>.csv: waktu, Gaussian, rasterizer, CUDA allocator.
- gpu_<run_id>.csv: RAM dan telemetry NVIDIA, jika monitoring eksternal aktif.
- manifest_<run_id>.json: command, parameter model/optimisasi/pipeline, environment,
  commit, tracked diff, hash source, dan identitas device.
- cameras_<run_id>.json: ukuran dan focal length kamera runtime, device/dtype,
  Gaussian awal dan faktor split.
- status_<run_id>.json: jumlah sampel selesai dan masalah instrumentasi.

Semua file dibuka secara eksklusif, tidak menimpa file eksperimen sebelumnya.
CSV di-flush setelah setiap baris selesai; GPU events dipoll secara nonblocking.
Baris dapat muncul satu/beberapa iterasi setelah sampelnya karena pembacaan
ditunda sampai GPU selesai. Pending queue dibatasi delapan sampel; jika penuh,
sampel berikutnya dilewati dengan warning, bukan memaksa sinkronisasi.

Ctrl+C/exception biasa menutup profiler, menunggu event terakhir sekali, lalu
menulis baris yang tersedia. Sampel terputus ditandai interrupted; jika CUDA gagal
saat shutdown, baris ditandai incomplete_cuda dan timing GPU dibiarkan kosong.
SIGKILL/kill paksa tidak dapat menjamin flush baris yang masih pending, tetapi
baris yang sudah di-flush tetap berada di CSV. Flush biasa bukan jaminan terhadap
listrik mati; tidak ada fsync pada setiap baris karena biaya I/O.

Pipeline juga membuat <stage>_resources_<uuid>.csv di research/logs. Saat
TrainingProfiler mengambil telemetry GPU, CSV runner mencatat RAM saja dengan
status handled_by_training_profiler. Telemetry GPU rinci ada pada gpu_<run_id>.csv.

## Definisi timing dan penghitung

- iteration_gpu_elapsed_ms: CUDA event dari awal iterasi setelah GUI sampai setelah
  checkpoint/optimizer/maintenance. Ini elapsed pada stream, termasuk jeda launch
  CPU, BUKAN jumlah waktu kernel aktif.
- iteration_cpu_submit_ms: wall-clock perf_counter untuk eksekusi/submission CPU,
  termasuk sinkronisasi yang SUDAH ada pada training. Ini BUKAN waktu selesai GPU
  yang disinkronkan.
- forward/loss/backward/optimizer/densification/filter3d_gpu_ms: pasangan CUDA event
  di masing-masing tahap; *_cpu_submit_ms adalah waktu host tahap tersebut.
- Tahap yang tidak dijalankan bernilai 0; flag membantu membedakan jadwal.
- Durasi total dapat melebihi jumlah enam tahap karena update statistik Gaussian,
  opacity reset, progress bar, evaluasi, penyimpanan, dan pekerjaan host lainnya.
- raster_stats_gpu_ms diukur SETELAH event akhir iterasi; biaya tersebut tetap
  memperpanjang eksperimen keseluruhan. profiler_cpu_enqueue_ms mencatat bagian
  enqueue akhir, bukan seluruh overhead profiler.
- cuda_peak_* adalah high-water mark allocator proses, bukan peak per iterasi.
  Profiler tidak mereset statistik allocator atau mengubah empty_cache.
  Peak dapat mencakup alokasi statistik profiling pada sampel sebelumnya.
- gaussians_rendered_input adalah jumlah sebelum rendering; gaussians_active adalah
  jumlah di akhir iterasi, setelah densification/pruning.
- split_net_added adalah hasil bersih split, bukan jumlah anak yang dibuat.
  Faktor N dibaca dari signature densify_and_split (saat ini N=2), sehingga
  split_parents_replaced = split_net_added/(N-1) dan
  split_children_created = split_parents_replaced*N.
- count_balance_ok memeriksa N_after = N_before + clone + split_net - opacity_pruned.
  Penghitung tidak mengubah mask atau rumus densification/pruning.

Profiling hanya mengamati iterasi yang dipilih. Ia tidak mengukur setiap iterasi
dalam interval 100. Sampel kontrol dianjurkan agar tidak hanya melihat iterasi
yang bertepatan dengan densification.

## Rasterizer tanpa mengubah ekstensi

Sebelum backward, profiler membaca num_rendered dan tiga ukuran buffer dari
context autograd yang sudah dibuat ekstensi. Ia tidak memanggil rasterizer lagi
dan tidak menyimpan graph/tensor context pada queue. Akses ini bergantung pada
layout Python wrapper versi repository ini; jika tidak cocok, kolom kosong dan
raster_context_status menunjukkan unavailable/unsupported_layout.

num_rendered adalah jumlah instansi Gaussian-tile, BUKAN jumlah Gaussian unik.
visible_gaussians dan radius mean/max dihitung dari radii yang sudah tersedia,
lalu tiga scalar disalin secara async ke pinned CPU memory. Tensor CPU hanya
dibaca setelah completion event selesai.

tile_grid_count dihitung dari ukuran kamera dan BLOCK_X/Y pada config.h.
tile_instances_per_grid_tile dan instances_per_visible_gaussian adalah indikator
kepadatan, bukan pengukuran jumlah tile yang benar-benar terisi atau distribusi
overlap per pixel. Pengukuran occupied tiles atau timing internal preprocess,
sort, dan blending memerlukan instrumentasi tambahan yang belum diterapkan.
Perubahan ekstensi CUDA tetap memerlukan persetujuan terpisah.

## Monitoring NVIDIA

Nilai [N/A], N/A, [Not Supported], nan dan kosong disimpan sebagai tidak tersedia,
bukan nol. Kegagalan satu sampel tidak menghapus sampel valid sebelumnya atau
menghentikan percobaan berikutnya. Sampling per-process VRAM dan device-global
independen, sehingga telemetry device masih dicoba jika VRAM proses tidak tersedia.

Device CSV mencatat index/UUID, utilization, VRAM used/total, SM clock, suhu, daya,
waktu query, dan status ketersediaan. Field opsional yang ditolak driver diprobe
sekali dan query yang didukung dipakai kembali. Tidak ada instalasi dependency.
Jika psutil tidak tersedia, monitoring eksternal dilewati dengan warning; event
training dan statistik CUDA allocator tetap dapat dicatat.

Angka utilization/clock/suhu/daya berlaku untuk SELURUH DEVICE, bukan hanya proses
training. Process VRAM adalah total PID training + anaknya pada device yang
dilaporkan. Pada multi-GPU, cocokkan identitas device; jangan menganggap ordinal
CUDA_VISIBLE_DEVICES sama dengan index nvidia-smi. Guard free_vram tetap gagal
secara konservatif jika data memori bebas tidak diketahui.

## Perbandingan 1024 dengan baseline 1600

Command berikut adalah contoh untuk mesin Linux training, belum dijalankan.
Gunakan output baru, dataset dan commit/biner ekstensi yang sama, parameter dan
seed yang sama, serta kondisi GPU yang sebanding.

~~~bash
RUN_ID=$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')
DATA=/home/pande/projects/RaDe-GS/dataset/togog_bali/togog_tualen
BASE=/home/pande/projects/RaDe-GS/output/togog_bali/diagnostics

# Jalankan satu per satu hanya setelah disetujui.
python train.py -s "$DATA" -m "$BASE/tualen_1600_$RUN_ID" -r 1600 \
  --profile_training --profile_from 1000 --profile_until 5000 \
  --profile_control_samples

python train.py -s "$DATA" -m "$BASE/tualen_1024_$RUN_ID" -r 1024 \
  --profile_training --profile_from 1000 --profile_until 5000 \
  --profile_control_samples
~~~

Target 30000 dan parameter optimisasi tidak ditimpa. Untuk percobaan diagnostik
yang disetujui, Ctrl+C setelah sampel 5000 selesai; jangan biarkan command
berlanjut menjadi training penuh tanpa persetujuan.

Baseline 1600 mereproduksi ukuran dua run awal, bukan mengandalkan resolution=-1
yang sekarang memakai modifikasi lokal auto-resize 1024. Menjalankan usulan
perbandingan resolusi juga menunggu persetujuan; konfigurasi utama tidak diubah.

Bandingkan nomor iterasi dan sample_kind yang sama, pisahkan densification,
opacity reset, evaluation/save, dan lihat profil waktu terhadap jumlah Gaussian,
num_rendered, radius, serta allocated/reserved. Jangan bandingkan angka EMA tqdm
dengan elapsed CUDA seolah-olah metriknya sama. Bandingkan juga run tanpa profiler
pada command yang sama untuk mengukur overhead instrumentasi.

## Validasi CPU dan batasannya

~~~bash
python -B -m unittest discover -s tests -p test_training_profiler.py -v
~~~

Tes menggunakan event/tensor stub CPU dan mock nvidia-smi. Tes memeriksa pemisahan
split net/children, polling nonblocking, flush/interrupt, file unik, antrean
terbatas, retry N/A, fallback field GPU, dan kesamaan struktur algoritma training
terhadap fingerprint baseline sebelum instrumentasi. Tidak ada model atau training yang dijalankan.

Timing GPU, biaya pinned-copy, dan kesesuaian context ekstensi terpasang belum
dapat diverifikasi secara nyata pada perangkat ini karena PyTorch/CUDA tidak
tersedia. Verifikasi runtime tersebut menunggu mesin training dan persetujuan.

## Mengembalikan kode

Sebelum penerapan, empat file asal disalin ke .profiling_backup/<id>/ dengan
struktur relatif yang sama. manifest.json menyimpan hash sebelum/sesudah dan
daftar file baru. Bandingkan perubahan berikutnya sebelum menyalin backup.

Salin kembali empat file ini dari backup ke lokasi asal:

- train.py
- research_pipeline/gpu_monitor.py
- research_pipeline/process_monitor.py
- research_pipeline/runner.py

File baru utils/training_profiler.py, tests/test_training_profiler.py, dan dokumen
ini boleh tetap disimpan; training lama tidak mengimpornya setelah train.py
dipulihkan. Jika menginginkan tree persis seperti sebelum penerapan, arsipkan
atau hapus hanya file baru yang tercantum pada manifest setelah memeriksa bahwa
tidak ada perubahan lanjutan.

Jangan melakukan reset seluruh repository. Perubahan pengguna pada
configs/togog_bali.yaml dan utils/camera_utils.py bukan bagian rollback ini.
