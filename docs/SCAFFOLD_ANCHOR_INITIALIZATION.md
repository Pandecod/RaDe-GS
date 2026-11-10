# Ablasi inisialisasi anchor Scaffold-GS pada RaDe-GS

Implementasi ini menguji perlakuan berikut pada setiap objek:

| Kondisi | Alur |
| --- | --- |
| Baseline | COLMAP -> training RaDe-GS -> render, evaluasi, mesh |
| Scaffold anchor initialization | COLMAP -> training Scaffold-GS -> ekspor anchor XYZ/RGB -> training RaDe-GS -> render, evaluasi, mesh |

Ini adalah inisialisasi dari hasil Scaffold-GS. Model, rasterizer, loss,
densifikasi, serta ekstraksi mesh RaDe-GS tetap memakai implementasi yang sama.
Ini tidak membawa jaringan prediksi atribut Scaffold ke dalam training RaDe-GS.
Implementasi saat ini hanya untuk RaDe-GS; tidak menambahkan 3DGS atau SuGaR.

## Definisi titik awal

- Ambil XYZ anchor dari PLY final resmi Scaffold-GS yang mempunyai atribut
  f_anchor_feat_* dan f_offset_*. Ini bukan pusat neural Gaussian setelah offset.
- Buang koordinat nonfinite dan duplikat persis.
- Warna awal adalah RGB titik COLMAP terdekat, menggunakan cKDTree.
  Jika points3D.ply sumber sudah tersedia, gunakan cloud yang sama dengan loader
  baseline; jika belum, baca points3D.bin atau points3D.txt.
- max_color_distance: 0.0 menonaktifkan pembatasan jarak. Nilai positif membuang
  anchor yang terlalu jauh dari titik referensi warna. Satuannya adalah
  koordinat COLMAP, belum tentu meter. Atur konsisten antarpercobaan.
- Jangan memakai opacity pada PLY anchor sebagai confidence: field ini bukan
  hasil prediksi opacity neural Gaussian. Tidak ada penyaringan opacity di sini.
- Tidak ada perubahan skala, rotasi, translasi, atau pose kamera.
  Ukuran, rotasi, opacity, dan fitur SH awal dibuat kembali oleh RaDe-GS.

Jumlah titik berubah sebagai bagian dari perlakuan. Klaim hasil berlaku untuk
perlakuan inisialisasi ini, bukan seluruh kemampuan Scaffold-GS. Anchor yang
diekspor adalah himpunan hasil training, termasuk anchor growing/pruning;
tidak mengasumsikan setiap posisi anchor bergeser selama optimasi.

## Persiapan di laptop lokal

Kode dan submodule dapat disiapkan di laptop ini sebelum environment training
tersedia. Clone repo Scaffold resmi berada di eksperimen/Scaffold-GS, dipatok pada
commit 59c833b56bbbf510f3f64d40f81721995caced66, sama dengan acuan pembanding.
Kode upstream tetap utuh; penyesuaian launcher berlangsung di memori proses.

Untuk salinan proyek lain yang belum berisi submodule:

~~~bash
git submodule update --init --recursive -- eksperimen/Scaffold-GS
git submodule status -- eksperimen/Scaffold-GS
~~~

Jangan memperbarui submodule ke versi lain di tengah eksperimen. expected_commit
pada YAML harus cocok dengan checkout; perubahan sumber upstream juga diperiksa.
Jangan menyalin hanya folder utama tanpa isi eksperimen/Scaffold-GS jika menggunakan
ZIP untuk memindahkan proyek. Submodule adalah repo terpisah yang perlu ikut diambil.

## Persiapan pada komputer training

1. Gunakan dataset COLMAP lengkap dan foto undistorsi. Model biner mendukung
   PINHOLE/SIMPLE_PINHOLE; loader teks lokal membutuhkan PINHOLE.
2. Pada Linux/WSL dengan GPU NVIDIA, siapkan environment Scaffold terpisah
   menggunakan environment.yml di repo tersebut sebagai acuan:

~~~bash
cd eksperimen/Scaffold-GS
conda env create --file environment.yml
cd ../..
~~~

Environment upstream memakai Python/PyTorch/CUDA versi lama. Kesesuaian perangkat
dan ekstensi diperiksa sebelum menjalankan eksperimen; file environment tersebut
bukan bukti bahwa semua GPU baru sudah didukung. Launcher kami kompatibel dengan
sintaks Python 3.7 yang digunakan upstream.

3. Environment RaDe-GS/runner membutuhkan requirements.txt dan
   requirements-research.txt. Ekspor CPU memakai numpy, scipy, dan plyfile.
4. repository pada YAML menggunakan path relatif eksperimen/Scaffold-GS.
   python_executable menggunakan variabel SCAFFOLD_PYTHON, sehingga tidak
   bergantung pada path komputer tertentu. Isi dengan interpreter sebenarnya:

~~~bash
export SCAFFOLD_PYTHON=/path/to/miniconda3/envs/scaffold_gs/bin/python
~~~

Variabel tersebut belum perlu diisi untuk menyiapkan kode atau menampilkan
dry-run. Variabel harus diisi sebelum preflight/training. Dataset root pada
kedua konfigurasi tetap perlu disesuaikan dengan lokasi data komputer training.

5. Jalankan pemeriksaan environment tanpa training atau pembuatan output:

~~~bash
python -B -m research_pipeline.scaffold_environment \
  --config configs/togog_bali_scaffold_init.yaml --scene ngawasari
~~~

Hasil JSON memeriksa pin repo, sumber ekstensi CUDA, interpreter, modul,
ketersediaan GPU, model kamera, dan foto train/test. Exit code 0 berarti semua
pemeriksaan lulus; exit code 1 berarti ada prasyarat yang belum tersedia.
Pemeriksaan ini belum menjalankan kernel training pada sebuah patung.

Konfigurasi:
- configs/togog_bali.yaml: baseline yang sudah ada.
- configs/togog_bali_scaffold_init.yaml: perlakuan Togog Bali.
- configs/tanks_temples_scaffold_init.yaml: perlakuan Tanks and Temples.

Konfigurasi perlakuan menyalin parameter baseline dan memakai output/CSV
terpisah. scaffold_initialization.enabled bernilai false secara default,
sehingga konfigurasi lama tetap baseline. Nilai voxel_size awal 0.001 mengikuti
default kode resmi, bukan nilai yang telah divalidasi untuk seluruh patung.
Iterasi Scaffold dan RaDe-GS merupakan dua anggaran yang terpisah.

## Jalankan antrean

Periksa satu patung terlebih dahulu; ganti ngawasari dengan nama folder objek:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali_scaffold_init.yaml \
  --scene ngawasari --start-from training --dry-run
~~~

Dry-run hanya menampilkan rencana; tidak memeriksa instalasi CUDA atau
menjalankan model. Saat eksekusi, path repository/interpreter dan kamera
diperiksa sebelum Scaffold dijalankan.

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali_scaffold_init.yaml \
  --scene ngawasari --start-from training
~~~

Untuk baseline seluruh objek, lalu perlakuan seluruh objek:

~~~bash
python -B research_pipeline/run_all.py \
  --config configs/togog_bali.yaml configs/togog_bali_scaffold_init.yaml \
  --start-from training --resume
~~~

Untuk melanjutkan eksperimen Scaffold seluruh Togog Bali dengan dewi_gangga
lalu detya sebagai dua objek terakhir, gunakan --last-scenes. Objek lain tetap
mengikuti urutan penemuan alfabet. Nama harus sama persis dengan folder dataset;
nama yang tidak ditemukan atau duplikat ditolak sebelum eksekusi.

~~~bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
export SCAFFOLD_PYTHON=/home/pande/miniconda3/envs/scaffold_gs/bin/python

# Periksa daftar dan urutannya tanpa training atau pembuatan output.
python -B research_pipeline/run_all.py \
  --config configs/togog_bali_scaffold_init.yaml \
  --last-scenes dewi_gangga detya \
  --start-from training --list-scenes

# Untuk melihat semua perintah, ganti --list-scenes dengan --dry-run.
# Jalankan antrean; hasil sukses dengan identitas sama dilewati.
python -B research_pipeline/run_all.py \
  --config configs/togog_bali_scaffold_init.yaml \
  --last-scenes dewi_gangga detya \
  --start-from training --resume
~~~

Opsi ini hanya mengubah urutan, bukan parameter atau experiment ID. Karena itu,
Ganesha yang sudah sukses dengan konfigurasi yang sama tetap dapat dilewati
oleh --resume. --list-scenes dan --dry-run menampilkan seluruh antrean;
pemeriksaan hasil sukses untuk resume dilakukan saat eksekusi. --last-scenes
tidak digabung dengan --scene. Jika beberapa YAML digunakan sekaligus, nama
pada --last-scenes harus tersedia di setiap dataset tersebut.

Penempatan terakhir tidak mengurangi kebutuhan memori suatu objek. Detya pernah
memerlukan --data_device cpu pada baseline; konfigurasi Scaffold ini masih
memakai data_device cuda. Pantau tahap Scaffold dan RaDe-GS sebelum menyimpulkan
semua objek dapat dijalankan dengan kebutuhan memori yang sama.

Urutan per objek pada konfigurasi perlakuan:
SCAFFOLD TRAIN -> SCAFFOLD EXPORT -> TRAIN -> RENDER -> IMAGE METRICS -> MESH
-> evaluasi geometri jika tersedia -> laporan -> objek berikutnya.

Tahap berikutnya tidak dijalankan jika training Scaffold atau ekspor gagal.
Keberadaan JSON dan checksum PLY diperiksa sebelum RaDe-GS dimulai.
resume melewati eksperimen yang seluruhnya sudah sukses dengan konfigurasi
sama. Percobaan gagal dimulai kembali dalam folder attempt baru; ini bukan
resume checkpoint per tahap. Jangan mengganti revisi repository Scaffold
di tengah antrean resume; buat eksperimen/output baru untuk revisi berbeda.

## Data dan hasil

Di bawah output setiap objek:

~~~text
scaffold_initialization/
  source/                    # input terpisah untuk Scaffold
  model/                     # hasil Scaffold, termasuk PLY final dan MLP
  scaffold_training.json     # perintah, revisi git, split, durasi program
  radegs_input/
    images/
    sparse/0/
      cameras.*              # salinan metadata asli
      images.*
      points3D.*             # metadata titik asli tetap disimpan
      points3D.ply           # anchor XYZ/RGB yang dibaca RaDe-GS
  initialization.json        # asal titik, filter, jumlah, split, SHA256
research/logs/
  scaffold_training.log
  scaffold_export.log
  training.log
~~~

Foto memakai hardlink jika memungkinkan, dan salinan jika hardlink tidak
didukung. Metadata kamera dan titik disalin, sehingga PLY sumber tidak ditimpa.
Simpan folder input turunan bersama hasil: cfg_args RaDe-GS menunjuk ke sana
untuk rendering/evaluasi berikutnya.

Pembaca RaDe-GS mengutamakan points3D.ply yang sudah ada. Karena itu PLY ekspor
dipakai sebagai geometri awal; points3D.bin/txt yang disalin bukan geometri
inisialisasi perlakuan.

Kedua tahap memakai --eval, urutan nama foto yang sama, dan split bawaan
index % 8 == 0 untuk test. Manifest mencatat nama foto train/test.
Training Scaffold memakai checkpoint iterasi final yang ditetapkan sebelumnya;
tidak memilih checkpoint atau titik berdasarkan skor test. Launcher memanggil fungsi
training upstream dengan evaluasi test dinonaktifkan. Pemilihan GPU otomatis,
viewer, alokasi model LPIPS evaluasi, dan rendering/evaluasi pascatraining dilewati.
Pilihan GPU mengikuti CUDA_VISIBLE_DEVICES milik proses pipeline. Rumus training,
model, dan strategi growing/pruning upstream dipertahankan.

summary.json menyimpan label metode, metadata inisialisasi, dan durasi/resource
tahap Scaffold dan ekspor. rincian_proses.csv menampilkan kedua tahap tambahan.
Kolom Training pada tabel utama tetap durasi program RaDe-GS; bandingkan biaya
total dengan menambahkan durasi persiapan. Durasi tahap Scaffold mencakup penyiapan data, pemeriksaan input, loading,
training, dan penyimpanan. Tahap ini tidak menjalankan evaluasi tambahan Scaffold.
Foto terdaftar dan metadata COLMAP di-hash sebelum training. Ekspor memeriksa
kecocokan receipt training, split, fingerprint input, dan checksum PLY anchor
sebelum RaDe-GS dimulai. Perubahan input saat training menyebabkan tahap gagal.

## Ekspor manual dari Scaffold yang sudah dilatih

~~~bash
python -B -m research_pipeline.scaffold_initialization export \
  --scene /data/patung \
  --anchors-ply /hasil/scaffold/point_cloud/iteration_30000/point_cloud.ply \
  --output /hasil/radegs_dengan_scaffold
python train.py \
  -s /hasil/radegs_dengan_scaffold/scaffold_initialization/radegs_input \
  -m /hasil/radegs_dengan_scaffold --eval --resolution 1920
~~~

Gunakan model Scaffold yang dilatih pada koordinat dan split dataset yang sama.
Ekspor manual tidak dapat membuktikan cara model eksternal sebelumnya dilatih.

## Batas validasi

Pengujian CPU memeriksa ekspor PLY ASCII/biner, pewarnaan tetangga terdekat,
filter, pose/data sumber tetap sama, checksum, split, urutan antrean, kegagalan
tahap, dan antarmuka proses training memakai fixture kecil tanpa GPU.
Keberhasilan training CUDA dan peningkatan kualitas mesh perlu divalidasi
dengan repositori Scaffold yang terpasang dan satu objek nyata di server.

## Pemulihan benturan impor utils pada launcher

Jika tahap Scaffold berhenti dengan pesan `No module named 'utils.loss_utils';
'utils' is not a package`, perbarui launcher integrasi. Folder utils milik
Scaffold merupakan namespace package tanpa __init__.py; file
research_pipeline/utils.py sebelumnya bisa terpilih ketika launcher dipanggil
sebagai skrip. Launcher kini menetapkan paket utils dari checkout Scaffold di
subprocess tersebut dan mengeluarkan jalur runner dari pencarian impor.
Kode training, model, loss, dan ekstensi CUDA upstream tidak diubah.

Pemeriksaan scaffold_environment kini juga memanggil launcher yang sebenarnya
memakai --help. Ini memeriksa impor lengkap tanpa training atau membuat output
model; kegagalan impor membuat ready bernilai false. Setelah pembaruan, jalankan
pemeriksaan environment dan ulangi pipeline satu objek dengan perintah yang sama.
Percobaan gagal tetap disimpan; runner menempatkan percobaan berikutnya pada
folder attempt baru bila output awal sudah ada. Tidak perlu menghapus hasil lama
atau memasang ulang CUDA untuk memperbaiki benturan nama modul ini.

Tes regresi tanpa training GPU:

~~~bash
PYTHONPATH=tests python -B -m unittest test_scaffold_launcher -v
~~~
