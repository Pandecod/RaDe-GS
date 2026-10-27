# Panduan pemulihan training dan ekstraksi mesh RaDe-GS

Tanggal pencatatan: 26 Oktober 2026.
Dataset: togog_tualen, bagian dari Dataset Togog Bali.
Status: training telah menghasilkan model 30.000 iterasi; ekstraksi melalui
mesh_extract_tnt.py dengan Open3D 0.19.0 berhasil menyimpan mesh yang telah
dibuka dan dikonfirmasi pengguna.

Dokumen ini mencatat hasil percobaan pada server pengguna. Pemeriksaan lokal
oleh asisten berupa pembacaan kode, log, dan hasil; seluruh eksekusi GPU,
instalasi paket, dan pemulihan WSL dilakukan pengguna di server.

## 1. Hasil yang menjadi acuan

| Tahap | Hasil yang tercatat |
|---|---|
| Diagnosis training 3.000 iterasi sebelum pengaturan fraction eksplisit | Loop training 36 menit 34 detik. |
| Diagnosis training 3.000 iterasi dengan --enable-gc | Loop training 42 detik; keseluruhan sekitar 1 menit 40 detik. |
| Training penuh | Model iteration_30000/point_cloud.ply tersimpan, sekitar 560 MB. Durasi lengkap 30.000 iterasi belum ditetapkan dari log yang diperiksa. |
| Uji TSDF sederhana dengan Open3D 0.20.0 | 768 titik dari depth, tetapi 0 vertex dan 0 segitiga mesh. |
| Uji TSDF sederhana dengan Open3D 0.19.0 | 1.518 vertex, 2.880 segitiga, UJI MESH BERHASIL. |
| Ekstraksi dataset dengan Open3D 0.19.0 | 377 kamera selesai; mesh terbentuk dan post-processing selesai. |
| File hasil | recon.ply sekitar 660 MB; recon_post.ply sekitar 539 MB. |
| Log vertex setelah post-processing | 11.088.102 vertex. |
| Pemeriksaan visual pengguna | Mesh dapat dibuka; patung utama, alas, lantai, dan objek sekitar terlihat. |

Catatan waktu ekstraksi: log awal 19:21:28, integrasi dimulai sekitar 19:22:08,
ekstraksi mesh 19:22:34, dan log akhir post-processing 19:23:33.
Rentang log tersebut 2 menit 5 detik. Ini bukan pengukuran total proses yang
presisi karena penulisan file terakhir terjadi setelah log post-processing.

Kualitas geometri terhadap GT 3D belum dinilai. Keberhasilan membuat dan membuka
mesh tidak menjamin seluruh detail, permukaan, dan skala geometrinya akurat.

## 2. Lingkungan yang berhasil

| Komponen | Versi/kondisi yang dilaporkan |
|---|---|
| Environment | radegs, environment asli |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| Open3D pada ekstraksi berhasil | 0.19.0 |
| scikit-learn setelah dependensi dilengkapi | 1.9.1 |
| Pemeriksaan paket | python -m pip check: No broken requirements found. |
| GPU | NVIDIA GeForce RTX 4090, sekitar 24 GB VRAM |
| Linux | Ubuntu 22.04.5 LTS dalam WSL2 |
| WSL yang dilaporkan | 2.5.9.0; kernel 6.6.87.2-microsoft-standard-WSL2 |
| Folder proyek server | /home/pande/projects/RaDe-GS |
| Dataset | /home/pande/projects/RaDe-GS/dataset/togog_bali/togog_tualen |
| Model | /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen |

Versi-versi tersebut adalah catatan percobaan, bukan jaminan untuk setiap
kombinasi mesin dan paket. Pada integrasi 27 Oktober 2026, requirements.txt
sudah menetapkan open3d==0.19.0. Mengubah file requirements tidak otomatis
mengganti paket pada environment; versi server harus tetap diperiksa.

## 3. Bedakan masalahnya sebelum memilih tindakan

| Gejala | Bukti dan batas kesimpulan | Tindakan yang sesuai |
|---|---|---|
| Training sangat lambat setelah sekitar 2.500 iterasi | Cache allocator membesar; run dengan fraction eksplisit dan GC kembali cepat. | Aktifkan --enable_allocator_gc pada train.py atau konfigurasi pipeline. |
| mesh_extract.py berhenti saat kamera 91 | Terminal kembali/WSL terputus, tanpa traceback lengkap dari run itu. | Jangan menyimpulkan kamera/COLMAP rusak dari nomor kamera terakhir saja. |
| mesh_extract.py gagal setelah 377 kamera selesai | VoxelBlockGrid, pengukuran 54.132 blok TSDF, tabel bantu sekitar 3,304 GiB; analisis source dan crash sebelumnya mendukung dugaan overflow pengindeksan 32-bit. | Jalur ini belum diperbaiki atau dibuktikan pulih; gunakan jalur alternatif yang telah berhasil. |
| mesh_extract_tnt.py menghasilkan mesh 0 vertex di Open3D 0.20.0 | Reproduksi tanpa dataset juga kosong; laporan Open3D sesuai dengan kesalahan skala depth saat aktivasi blok. | Open3D 0.19.0 + uji kecil, kemudian ekstraksi dataset. |
| IndexError pada np.sort(...)[-8] | Terjadi setelah mesh kosong sehingga tidak ada cluster. | Selesaikan mesh kosong dahulu; mengganti 8 menjadi 7 tidak memperbaiki akar masalah. |
| Operation not permitted saat uninstall Open3D | Izin folder utama dan atribut pengunci tidak menjelaskan error; temp pip tercatat di /mnt/e/Arca_Project/build_tmp. | Log rinci dan temp Linux; penyebab operasi persis pada percobaan lama belum terkonfirmasi. |
| Python, tail, jurnal sistem mengalami Input/output error | C: sempat 0 GB; penyimpanan Ubuntu berada di C:. | Pulihkan ruang dan akses penyimpanan sebelum instalasi/ekstraksi berikutnya. |
| ModuleNotFoundError: sklearn setelah downgrade | Open3D dipasang dengan --no-deps sehingga dependensi belum lengkap. | Lengkapi dependensi tanpa --no-deps, verifikasi import dan pip check. |

Penyebab crash saat kamera 91 belum terbukti sama dengan kegagalan setelah
integrasi selesai. C: penuh adalah temuan penyimpanan yang penting, tetapi
tidak cukup untuk mengatribusikan semua crash terdahulu pada satu penyebab.

Keberhasilan akhir adalah kombinasi perpindahan jalur ekstraksi dari
VoxelBlockGrid ke ScalableTSDFVolume dan penggunaan Open3D 0.19.0.
Tidak ada bukti bahwa downgrade saja memperbaiki mesh_extract.py.

## 4. Pengaturan training yang berhasil tetap tersedia

train.py memiliki opsi --enable_allocator_gc. Saat aktif, kode mengatur:
- PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.8,
  sebelum import torch.
- torch.cuda.set_per_process_memory_fraction(1.0), sebelum pembuatan data/model CUDA.

Pada implementasi native PyTorch 2.6 yang diperiksa, jalur GC memerlukan
fraction disetel eksplisit. Menyetel environment threshold 0.8 saja sebelumnya
belum cukup mengatasi perlambatan.

Bukti diagnosis sebelum/sesudah:
- Sebelum fraction eksplisit: reserved akhir sekitar 26,81 GiB; num_device_free 0.
- Setelah diaktifkan: reserved puncak sekitar 19,53 GiB; num_device_free akhir 54.
- Median iterasi biasa pada rentang 2.600-2.999 turun dari sekitar 5.071 ms
  menjadi 20,7 ms.
- Pemakaian shared GPU memory Windows juga lebih rendah pada run cepat.

Ini mendukung penumpukan cache/tekanan memori sebagai penyebab utama perlambatan
training pada percobaan tersebut. Mekanisme internal perpindahan memori driver
tidak ditetapkan hanya dari CSV. Waktu training tidak dijamin sama untuk setiap run.

| Cara menjalankan | Cara mengaktifkan |
|---|---|
| Manual | Tambahkan --enable_allocator_gc pada perintah python train.py. |
| Pipeline | Tambahkan --enable_allocator_gc pada training.extra_args dalam YAML yang dipakai. |

Sejak integrasi 27 Oktober 2026, configs/togog_bali.yaml dan
configs/togog_bali_full_gc.yaml sama-sama mengaktifkan GC, 30.000 iterasi,
resolusi 1024, dan profiling. Root output dan CSV kedua konfigurasi berbeda.
Periksa path konfigurasi server saat menyalin file.

Log penanda:
~~~text
[ALLOCATOR] Explicit memory fraction=1.0; native GC threshold=0.8
~~~

Opsi training ini tidak otomatis berlaku pada subprocess rendering/mesh.
Algoritma loss, densification, pruning, dan optimizer tidak diganti untuk
menerapkan opsi allocator. Tidak perlu mengulang training untuk menguji mesh
dari model yang sudah tersimpan.

## 5. Prosedur ketika instalasi atau WSL bermasalah

Bedakan terminal:
- Windows PowerShell: prompt seperti PS C:\Users\indra>.
- Ubuntu: prompt seperti (radegs) pande@PC-AI-SERVER-1:...$.

Jika masih ada Input/output error, hentikan pemasangan. Jangan terus mengunduh
atau menghapus file tanpa mengetahui kondisi filesystem.

Pemeriksaan baca-saja di Windows PowerShell:
~~~powershell
Get-Volume | Select-Object DriveLetter, HealthStatus,
    @{Name='FreeGB';Expression={[math]::Round($_.SizeRemaining/1GB,2)}}

Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' |
    Get-ItemProperty |
    Where-Object { $_.DistributionName -eq 'Ubuntu-22.04' } |
    Format-List DistributionName, BasePath
~~~

Temuan kasus ini:
- Ubuntu berada di C:\Users\indra\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState.
- C: pernah 0 GB, lalu sekitar 4 GB; angka kapasitas kosong virtual di Ubuntu
  sebelumnya tidak menggambarkan ruang fisik C:.
- ext4.vhdx terukur 143,21 GB pada pemeriksaan tersebut.
- Salinan radegs_mesh019 terukur 9,1 GB dan telah dihapus atas permintaan pengguna.
- Folder output penelitian terukur 8 GB dan dipertahankan.
- Unduhan sementara Open3D yang diketahui berada di E: tidak ditemukan lagi
  saat diperiksa; tidak ada file yang dihapus oleh pemeriksaan itu.
- Tidak ada catatan ukuran awal C: yang dapat membuktikan seluruh pertumbuhan
  disebabkan percobaan ini.
- Pengguna kemudian melaporkan penyimpanan pulih; tindakan pemulihan akhir
  dan ruang kosong sesudahnya tidak dirinci dalam bukti yang tersedia.

Jangan menghapus ext4.vhdx: itu memuat Ubuntu, environment, dan proyek.
Penghapusan isi Ubuntu tidak selalu langsung mengecilkan VHDX di Windows.
Pemadatan atau pemindahan distro harus direncanakan bersama pengelola server,
dengan backup dan tanpa pekerjaan aktif. wsl --shutdown menghentikan seluruh WSL,
termasuk proses pengguna lain; bukan langkah otomatis untuk setiap error.

Environment salinan gagal dipakai (Exec format error) dan kemudian gagal dihapus
melalui Conda karena CorruptedEnvironmentError. Folder salinan akhirnya dihapus
secara terbatas setelah alamatnya diperiksa. Itu catatan kejadian, bukan anjuran
menghapus environment lain ketika menghadapi error.

## 6. Pemasangan Open3D yang akhirnya berhasil

Jalankan di Ubuntu server setelah penyimpanan dan akses file pulih:
~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
~~~

Percobaan yang berhasil mengganti paket awal memakai --no-deps:
~~~bash
TMPDIR=/tmp TEMP=/tmp TMP=/tmp \
python -m pip --disable-pip-version-check install \
  --only-binary=:all: --no-deps "open3d==0.19.0"
~~~

Namun, import kemudian gagal karena sklearn belum tersedia. Langkah berikut
melengkapi dependensi dan menjadi prosedur yang disarankan untuk pengulangan
kasus ini, setelah meninjau environment yang digunakan:
~~~bash
TMPDIR=/tmp TEMP=/tmp TMP=/tmp \
python -m pip --disable-pip-version-check install \
  --only-binary=:all: "open3d==0.19.0"

python -c "import torch, open3d, sklearn; print('PyTorch:', torch.__version__); print('Open3D:', open3d.__version__); print('scikit-learn:', sklearn.__version__)"
python -m pip check
~~~

TMPDIR/TEMP/TMP hanya berlaku untuk perintah tersebut. Tanpa --no-deps, pip
dapat memasang atau menyesuaikan dependensi untuk memenuhi persyaratan.
Periksa keluarannya; jangan menjanjikan bahwa semua paket selalu tidak berubah.
Pada hasil pengguna, PyTorch tetap 2.6.0+cu124, Open3D 0.19.0,
scikit-learn 1.9.1, dan pip check tidak menemukan persyaratan rusak.

Jika download terputus, tangani koneksi/unduhan. Jika download telah selesai
tetapi uninstall gagal, jangan menganggap mengunduh ulang akan memperbaiki izin.
Jika import gagal karena dependensi, perbaiki dependensi sebelum menguji TSDF.

## 7. Uji kecil sebelum ekstraksi seluruh dataset

Uji ini memakai depth buatan di CPU; tidak memuat dataset atau menjalankan training.
Salin seluruh blok di terminal Ubuntu. Baris PY penutup harus berdiri sendiri.

~~~bash
python -B - <<'PY'
import numpy as np
import open3d as o3d

print("Open3D:", o3d.__version__)
assert o3d.__version__ == "0.19.0", "Versi belum sesuai acuan percobaan."
w, h = 32, 24
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.full((h, w, 3), 128, dtype=np.uint8)),
    o3d.geometry.Image(np.full((h, w), 2.0, dtype=np.float32)),
    depth_scale=1.0,
    depth_trunc=5.0,
    convert_rgb_to_intensity=False,
)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    w, h, 28.0, 28.0, (w - 1) / 2, (h - 1) / 2
)
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.05,
    sdf_trunc=0.1,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)
volume.integrate(rgbd, intrinsic, np.eye(4))
mesh = volume.extract_triangle_mesh()
print("Vertex:", len(mesh.vertices))
print("Segitiga:", len(mesh.triangles))
assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0, "Mesh masih kosong."
print("UJI MESH BERHASIL")
PY
~~~

Hasil acuan pengguna: Vertex 1518, Segitiga 2880, UJI MESH BERHASIL.
Syarat melanjutkan: import berhasil dan mesh tidak kosong.
Jika gagal, catat error; jangan langsung menjalankan ekstraksi 377 kamera.

## 8. Perintah ekstraksi dataset yang berhasil

Ini adalah perintah historis yang benar-benar dijalankan pengguna.
Hasilnya sudah ada; menjalankan ulang pada model_path yang sama dapat menimpa
recon.ply dan recon_post.ply serta memperbarui multi_view.json.
Untuk percobaan baru, gunakan folder hasil baru seperti bagian 9.

~~~bash
python mesh_extract_tnt.py \
  -m /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen \
  --iteration 30000 \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5 \
  --voxel_size 0.009 \
  --max_depth 8.0
~~~

| Argumen | Makna dan batasnya |
|---|---|
| --iteration 30000 | Memuat Gaussian yang telah dilatih; tidak menjalankan training. |
| -r 2 | Memperkecil dimensi gambar menjadi setengah dimensi asal, bukan crop objek. |
| --num_cluster 8 | Ambang berdasarkan cluster terbesar ke-8, dengan batas minimum 50 triangle pada kode ini. Jumlah akhir bisa berbeda jika ada ukuran cluster sama atau cluster kecil. Bekerja setelah mesh terbentuk. |
| --depth_ratio 0.5 | Kode ini memilih median_depth untuk nilai >=0.5; bukan memotong 50% area. |
| --voxel_size 0.009 | Ukuran voxel dalam satuan koordinat model; jangan langsung menyebut meter/mm tanpa kalibrasi skala. |
| --max_depth 8.0 | Batas depth yang diintegrasikan dalam satuan model; bukan pemilihan patung tertentu. |

Jika dataset mempunyai transforms.json dengan aabb_range, skrip bawaan memakai
AABB dan menghitung ulang voxel_size dari sisi terpanjang / 2048. Pada run yang
berhasil, log menampilkan TSDF voxel_size 0.009. Jangan menganggap nilai CLI
selalu berlaku untuk dataset lain.

Laporan keberhasilan:
~~~text
Loading trained model at iteration 30000 [26/10 19:21:28]
TSDF voxel_size 0.009 [26/10 19:22:08]
Rendering progress: 100% ... 377/377 [00:26 ...]
extract_triangle_mesh [26/10 19:22:34]
Done clustering, #clusters=340644 [26/10 19:23:17]
num vertices raw 11088102 [26/10 19:23:33]
num vertices post 11088102 [26/10 19:23:33]
~~~

Tidak ada traceback pada percobaan tersebut, kedua PLY ditemukan di disk,
dan pengguna berhasil membukanya. Bukti ini lebih kuat daripada hanya melihat
progress 100%: progress kamera selesai belum selalu berarti mesh berhasil ditulis.

## 9. Mengulang ekstraksi tanpa menimpa hasil asli

Persiapan opsional di Ubuntu. Ini membuat folder kecil dan symlink model,
bukan menyalin ulang file Gaussian besar. Jalankan saat tidak ada proses lain
menulis model sumber.

~~~bash
(
set -e
cd /home/pande/projects/RaDe-GS
model=/home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen
test -f "$model/cfg_args"
test -f "$model/point_cloud/iteration_30000/point_cloud.ply"
hasil=$(mktemp -d /home/pande/projects/RaDe-GS/output/togog_bali/togog_tualen_mesh019_XXXXXX)
cp "$model/cfg_args" "$hasil/cfg_args"
ln -s "$model/point_cloud" "$hasil/point_cloud"
printf 'Folder hasil baru: %s\n' "$hasil"
)
~~~

Catat alamat folder yang dicetak, lalu gunakan alamat itu sebagai nilai -m
pada perintah bagian 8. Model sumber harus tetap ada karena symlink merujuk
ke sana. Seluruh mesh baru dan multi_view.json akan berada di folder baru.

## 10. Pemeriksaan hasil dan kekeliruan log vertex

Pemeriksaan file setelah run historis:
~~~bash
ls -lh \
  /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen/recon.ply \
  /home/pande/projects/RaDe-GS/output/togog_bali/full_gc/togog_tualen/recon_post.ply
~~~

Hasil pengguna:
- recon.ply: sekitar 660 MB, mesh sebelum pemilihan cluster.
- recon_post.ply: sekitar 539 MB, setelah post-processing.

Pada post_process_mesh dalam mesh_extract_tnt.py, mesh_0 = mesh merujuk objek
yang sama. Kedua log jumlah vertex dicetak sesudah objek itu dimodifikasi.
Karena itu, log "raw" dan "post" sama bukan bukti filter tidak bekerja;
angka raw asli tidak dapat diketahui dari dua baris tersebut.
Penulisan recon.ply terjadi sebelum penyaringan. Ukuran file yang berbeda
konsisten dengan adanya perubahan hasil, tetapi bukan metrik kualitas.

#clusters=340644 adalah komponen terhubung sebelum pemilihan, bukan jumlah
patung maupun jumlah cluster akhir. Jangan membaca angka itu sebagai error.

Buka recon_post.ply untuk pemeriksaan visual:
- kelengkapan patung, alas, wajah dan ornamen;
- lubang, permukaan pecah, fragmen melayang;
- bagian latar yang masih ikut;
- bagian patung yang mungkin hilang karena pemilihan cluster.

Gambar pengguna menunjukkan objek utama dan lingkungan masih ikut.
Ekstraksi belum membatasi area hanya ke patung tengah. Jika perlu memotong
objek, kerjakan pada salinan hasil dan pertahankan kedua mesh asli.

## 11. GT gambar dan GT geometri

File ls -lh hanya membuktikan keberadaan dan ukuran file, bukan evaluasi GT.

Evaluasi render yang diperiksa sebelumnya:
- 377 gambar training; 0 gambar test.
- PSNR sekitar 25,81 dB.
- SSIM sekitar 0,9033.
- LPIPS sekitar 0,0930.

Ini metrik pada view training, bukan bukti generalisasi atau akurasi mesh.
GT foto dipakai untuk evaluasi render. Evaluasi geometri memerlukan GT 3D
yang sesuai, disejajarkan, dan mempunyai skala yang konsisten.
Belum ada evaluasi GT geometri yang selesai pada catatan ini.

## 12. Status pipeline, arsip, dan pengulangan

- Jalur manual yang telah berhasil: mesh_extract_tnt.py + Open3D 0.19.0.
- Integrasi 27 Oktober 2026: research_pipeline/commands.py sekarang memanggil
  mesh_extract_tnt.py dengan voxel_size 0.009 dan max_depth 8.0.
- Integrasi diperiksa menggunakan pengujian CPU dan dry-run. Keberhasilan mesh
  nyata yang dilaporkan di atas berasal dari run manual server sebelumnya.
- Prosedur penyalinan dan pemeriksaan: [Pipeline GC + TNT](PIPELINE_GC_TNT.md).
- Skrip pengukuran TSDF, diagnosis allocator 3.000 iterasi, dan perangkat ROI
  lama berada di arsip_lokal/20261026_kode_percobaan/, diabaikan Git.
- Profiling training, monitor, train.py dan skrip utama tetap dipertahankan.
- Panduan ini berada di docs/ agar dapat disertakan ke GitHub.
- Tidak ada perubahan algoritma, instalasi, training, ekstraksi, commit,
  atau push yang dijalankan saat menulis panduan ini.

Lihat juga [Peta kode aktif](PETA_KODE_AKTIF.md),
[Profiling training](training_profiling.md), dan
[Runbook konfigurasi full GC](togog_tualen_full_gc_runbook.md).

## 13. Rujukan teknis

Rujukan berikut mendukung penjelasan teknis; hasil kasus ini tetap berasal dari
log pengguna dan kode proyek yang diperiksa.

- [Laporan mesh kosong Open3D 0.20.0](https://github.com/isl-org/Open3D/issues/7592).
- [Usulan perbaikan skala depth ScalableTSDFVolume](https://github.com/isl-org/Open3D/pull/7594).
- [Kode ScalableTSDFVolume Open3D 0.20.0](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/pipelines/integration/ScalableTSDFVolume.cpp).
- [Indexer Open3D 0.20.0](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/t/geometry/kernel/GeometryIndexer.h).
- [Implementasi VoxelBlockGrid Open3D 0.20.0](https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/t/geometry/kernel/VoxelBlockGridImpl.h).
- [Allocator native PyTorch 2.6](https://github.com/pytorch/pytorch/blob/v2.6.0/c10/cuda/CUDACachingAllocator.cpp).
- [Pengelolaan memori CUDA PyTorch 2.6](https://docs.pytorch.org/docs/2.6/notes/cuda.html#memory-management).
- [Pengelolaan ruang WSL](https://learn.microsoft.com/en-us/windows/wsl/disk-space).
- [Pemadatan disk virtual](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk).
- [Dependensi dan opsi pip install](https://pip.pypa.io/en/stable/cli/pip_install/).
- [Pemilihan direktori sementara Python](https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir).
