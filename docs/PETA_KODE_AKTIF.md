# Peta kode aktif RaDe-GS

Diperbarui 27 Oktober 2026. Panduan ini membedakan kode utama, alat pemantauan,
dan percobaan lama sebelum menyalin file ke server atau GitHub.

## File yang tetap dipakai

| File/folder | Peran dan status |
|---|---|
| train.py | Training utama; opsi --enable_allocator_gc yang sudah dipakai pada percobaan berhasil tetap tersedia. |
| gaussian_renderer/, scene/, arguments/, submodules/ | Komponen model dan ekstensi yang diperlukan program utama. |
| utils/training_profiler.py | Profiling yang diimpor training; diaktifkan melalui opsi profiling. |
| research_pipeline/ | Runner, pembentukan perintah, monitoring, dan evaluasi. |
| research_pipeline/gpu_monitor.py, research_pipeline/process_monitor.py | Monitoring yang dipakai pipeline. |
| scripts/monitor_windows_gpu.ps1 | Monitor Windows opsional yang pernah dipakai; tetap tersedia dan dirujuk runbook. |
| render.py | Rendering model yang telah dilatih. |
| mesh_extract_tnt.py | Skrip bawaan ScalableTSDFVolume; berhasil menghasilkan mesh Tualen dengan Open3D 0.19.0 pada 26 Oktober 2026. |
| mesh_extract.py | Skrip VoxelBlockGrid lama; dipertahankan untuk pembanding, tidak dipanggil pipeline. |
| configs/togog_bali_full_gc.yaml | Salinan lokal konfigurasi training 30.000 iterasi dengan allocator GC. |
| configs/togog_bali.yaml | Konfigurasi utama: GC aktif, 30.000 iterasi, resolusi 1024, profiling, dan mesh TNT. |
| tests/ | Pengujian kode yang dipertahankan; pengujian ROI ikut diarsipkan bersama modul ROI. |
| docs/training_profiling.md, docs/togog_tualen_full_gc_runbook.md | Dokumentasi profiling dan konfigurasi run penuh. |

Di server, pengguna memakai nama configs/togog_bali.yaml untuk konfigurasi run
penuh. Jangan menimpa konfigurasi server dengan file lokal bernama sama tanpa
membandingkan isinya.

Integrasi 27 Oktober 2026 mengarahkan pipeline ke mesh_extract_tnt.py dengan
voxel_size 0.009 dan max_depth 8.0; requirements.txt menetapkan Open3D 0.19.0.
Kedua YAML Togog Bali mengaktifkan GC. Validasi lokal memakai pengujian CPU dan
dry-run; tidak menjalankan training/mesh. Lihat [panduan integrasi](PIPELINE_GC_TNT.md)
untuk file yang perlu disalin ke server. Opsi yang benar adalah --depth_ratio,
bukan --depth_ration.

Training 30.000 iterasi sudah menghasilkan model di server:
 /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen/point_cloud/iteration_30000/point_cloud.ply
Pemindahan kode lokal ini tidak menjalankan model maupun mengubah hasil server.

## Percobaan yang diarsipkan

Lokasi: arsip_lokal/20261026_kode_percobaan/. Folder ini diabaikan Git.

| Berkas pada lokasi asal | Alasan diarsipkan |
|---|---|
| scripts/diagnose_allocator.py | Launcher diagnosis dibatasi 3.000 iterasi; opsi GC utama sudah tersedia di train.py. |
| mesh_extract_diagnostic.py | Hanya mengukur blok TSDF, sengaja berhenti sebelum membuat mesh. |
| mesh_extract_roi.py | Alternatif ROI tambahan yang tidak dipilih untuk uji berikutnya. |
| scripts/prepare_mesh_roi.py, scripts/mesh_roi_selector.html | Alat pembuatan dan pemilihan ROI. |
| utils/mesh_roi.py, tests/test_mesh_roi.py, docs/mesh_roi.md | Modul, pengujian, dan panduan khusus ROI. |
| mesh_roi_tools_20261026.zip | Paket lama perangkat ROI. |

Sembilan berkas dipindahkan dengan struktur relatif yang sama.
manifest.json dalam arsip merekam lokasi asal, lokasi arsip, alasan, dan SHA-256.
Kode arsip bukan entry point aktif dan jangan dijalankan langsung dari sana:
beberapa skrip bergantung pada letak relatif terhadap root proyek.

## File lokal yang tidak ikut Git

.gitignore mengecualikan arsip_lokal/, .profiling_backup/, analisa radegs/,
analisaradegs/, dan folder new/. Data analisis dan backup tetap berada di lokasi
semula; hanya kode percobaan dalam tabel di atas yang dipindahkan.
output/ sudah diabaikan sebelumnya.

Pengecualian berlaku pada Git, termasuk git add . biasa. Jangan gunakan
git add -f untuk folder tersebut. Mengunggah folder atau ZIP lewat web GitHub
secara manual tidak menggunakan aturan .gitignore.

File yang sudah tracked tidak otomatis dikeluarkan oleh .gitignore.
Penataan ini tidak menghapus file tracked maupun riwayat Git; misalnya
research_results/experiment_results.csv tetap tracked seperti sebelumnya.

Sebelum membuat commit, periksa git status --short dan git diff --stat.
Tidak ada staging, commit, atau push otomatis pada penataan ini.

## Mengembalikan kode percobaan

Baca manifest.json dalam arsip, lalu salin berkas yang diperlukan ke original_path
relatif terhadap root proyek. Periksa dahulu agar tidak menimpa perubahan baru.
Untuk jalur ROI, pulihkan seluruh kelompok modul, skrip, pengujian, dan
dokumentasinya bersama-sama. Restore launcher diagnosis hanya jika pengujian
tersebut memang akan digunakan kembali.

Salinan .gitignore dan runbook sebelum penataan disimpan di subfolder _metadata/
dalam arsip. Untuk membatalkan pengecualian Git tanpa menghapus perubahan lain,
hapus hanya blok aturan yang ditambahkan pada penataan ini.

Catatan gejala, percobaan, solusi yang berhasil, dan prosedur pengulangan ada di [Panduan pemulihan training dan mesh](PANDUAN_PEMULIHAN_TRAINING_DAN_MESH.md).
