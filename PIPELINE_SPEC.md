Saya ingin kamu membuat sebuah Research Pipeline otomatis untuk repository RaDe-GS yang sedang terbuka ini.

PENTING:
Model penelitian yang digunakan HANYA RaDe-GS.

Jangan menggunakan 2DGS sebagai model penelitian.
Jangan mengubah algoritma utama RaDe-GS.
Jangan mengganti implementasi train.py, render.py, mesh_extract.py, CUDA rasterizer, renderer, Gaussian representation, atau bagian core lainnya kecuali perubahan kecil benar-benar diperlukan agar pipeline dapat berinteraksi dengan repository.

Research Pipeline harus dibuat sebagai orchestration/automation layer terpisah di atas RaDe-GS.

============================================================
1. INSPEKSI REPOSITORY TERLEBIH DAHULU
============================================================

SEBELUM MENULIS CODE:

Inspect seluruh repository RaDe-GS yang sedang terbuka.

Periksa secara khusus:

train.py
render.py
mesh_extract.py
argument parser
scene loader
metric implementation
struktur output training
struktur output rendering
struktur output mesh
cfg_args
checkpoint
point_cloud
nama file mesh final

Jangan menebak CLI.

Jika ada perbedaan antara instruksi saya dan implementation repository yang sebenarnya, pertahankan tujuan pipeline tetapi sesuaikan pemanggilan command dengan CLI yang benar-benar tersedia.

Khusus command penelitian berikut harus dipertahankan jika parser repository mendukungnya:

TRAIN:

python train.py -s <dataset_path> -m <output_path>

MESH:

python mesh_extract.py \
  -m <output_path> \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5


============================================================
2. TUJUAN PIPELINE
============================================================

Saya memiliki banyak Dataset Togog Bali yang sudah melalui preprocessing COLMAP.

Saya ingin cukup memasukkan seluruh dataset ke satu folder.

Contoh:

/home/pande/dataset_togog_bali/
├── togog_ganesha/
│   ├── images/
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
│
├── togog_arjuna/
│   ├── images/
│   └── sparse/
│       └── 0/
│
├── togog_bima/
│   ├── images/
│   └── sparse/
│       └── 0/
│
└── ...

Research Pipeline harus otomatis:

discover dataset
→ validate dataset
→ train RaDe-GS
→ monitor training
→ render
→ monitor rendering
→ evaluate image
→ resource check
→ extract mesh
→ monitor mesh extraction
→ analyze mesh
→ geometry evaluation jika GT tersedia
→ save semua hasil penelitian
→ lanjut dataset berikutnya

SEMUA DATASET WAJIB DIPROSES SATU PER SATU SECARA SERIAL.

Jangan melakukan parallel training atau parallel mesh extraction.


============================================================
3. NAMA DATASET MENJADI IDENTITAS OBJEK
============================================================

Nama folder dataset harus otomatis menjadi identitas scene/object.

Contoh:

/home/pande/dataset_togog_bali/togog_ganesha

menghasilkan:

scene_name = "togog_ganesha"
object_name = "togog_ganesha"

Nama ini harus digunakan pada:

output folder
CSV
JSON
log
training metrics
rendering metrics
mesh metrics
geometry metrics

Jangan meminta pengguna memasukkan nama scene secara manual jika dapat diperoleh dari nama folder.


============================================================
4. STRUKTUR TAMBAHAN REPOSITORY
============================================================

Tambahkan struktur seperti:

RaDe-GS/
│
├── train.py
├── render.py
├── mesh_extract.py
├── ...
│
├── research_pipeline/
│   ├── __init__.py
│   ├── run_all.py
│   ├── config.py
│   ├── scene_discovery.py
│   ├── runner.py
│   ├── process_monitor.py
│   ├── gpu_monitor.py
│   ├── resource_guard.py
│   ├── training_parser.py
│   ├── image_evaluator.py
│   ├── mesh_analyzer.py
│   ├── geometry_evaluator.py
│   ├── results.py
│   └── utils.py
│
├── configs/
│   └── togog_bali.yaml
│
├── research_results/
│   └── experiment_results.csv
│
├── docs/
│   └── research_notes.md
│
├── RESEARCH_PIPELINE.md
└── requirements-research.txt

Jangan memindahkan file original RaDe-GS.


============================================================
5. CONFIG FINAL
============================================================

Buat config:

configs/togog_bali.yaml

Default:

project:
  name: radegs_togog_bali
  dataset: Dataset Togog Bali

dataset:
  root: /home/pande/dataset_togog_bali

output:
  root: /home/pande/radegs_results/togog_bali

results:
  csv: research_results/experiment_results.csv

training:
  extra_args: []

rendering:
  enabled: true
  extra_args: []

mesh:
  enabled: true
  resolution: 2
  num_cluster: 8
  depth_ratio: 0.5
  extra_args: []

resource_monitor:
  interval_seconds: 1.0

resource_guard:
  enabled: true
  cooldown_before_mesh_seconds: 5
  minimum_free_ram_gb: 4
  minimum_free_vram_mb: 1024
  wait_if_memory_low: true
  check_interval_seconds: 10
  max_wait_seconds: 300

geometry_evaluation:
  enabled: false
  gt_root: null
  sample_points: 100000
  distance_threshold: null
  seed: 0


============================================================
6. AUTO DISCOVERY DATASET
============================================================

Pipeline harus scan direct child directory dari:

dataset.root

Contoh:

/home/pande/dataset_togog_bali/
├── togog_ganesha/
├── togog_arjuna/
└── togog_bima/

Pipeline otomatis menemukan:

togog_arjuna
togog_bima
togog_ganesha

Urutkan scene secara deterministic, misalnya alphabetical.

Dataset dianggap valid jika minimal memiliki:

images/

dan:

sparse/0/

serta file COLMAP:

cameras.bin
images.bin
points3D.bin

Jika repository mendukung format text COLMAP, format .txt yang ekuivalen juga boleh diterima.

Jika dataset invalid:

status = failed
failed_stage = dataset_validation

Catat error.

Lanjutkan ke dataset berikutnya.

Jangan menghentikan seluruh batch.


============================================================
7. TRAINING RaDe-GS
============================================================

Untuk setiap scene jalankan RaDe-GS original melalui subprocess.

Command utamanya:

python train.py \
  -s <dataset_path> \
  -m <scene_output>

Contoh:

python train.py \
  -s /home/pande/dataset_togog_bali/togog_ganesha \
  -m /home/pande/radegs_results/togog_bali/togog_ganesha

Jangan menambahkan parameter training lain secara diam-diam.

Parameter tambahan hanya boleh berasal dari:

training.extra_args

Gunakan subprocess argument list, bukan shell string jika tidak diperlukan.

Simpan SEMUA stdout dan stderr ke:

<scene_output>/research/logs/training.log


============================================================
8. DATA TRAINING YANG HARUS DICATAT
============================================================

Semua terminal output train.py harus tetap disimpan utuh sebagai raw log.

Selain itu parsing metric penting jika muncul.

Contoh output yang mungkin muncul:

Reading camera 481/481
cameras extent: 6.188...
Loading Training Cameras: 481
Loading Test Cameras: 0
Number of points at initialisation: 344382

7000/30000
65.65it/s
Loss=0.2119
loss_normal=0.2527
loss_ncc=0.2098

[ITER 7000] Evaluating train:
L1 0.053633...
PSNR 23.4509

15000/30000
Loss=0.1864
loss_normal=0.0690
loss_ncc=0.3198

30000/30000
Loss=0.1564
loss_normal=0.0594
loss_ncc=0.3191

Parse jika tersedia:

camera_count
training_camera_count
test_camera_count
camera_extent
initial_point_count

iteration
loss
loss_normal
loss_ncc
l1
psnr
iterations_per_second

Jangan mengasumsikan nilai selalu tersedia.

Jika tidak tersedia:

null / empty

BUKAN 0.


============================================================
9. TRAINING METRICS CSV
============================================================

Buat:

<scene_output>/research/training_metrics.csv

Format contoh:

iteration,loss,loss_normal,loss_ncc,l1,psnr,it_per_sec
7000,0.2119,0.2527,0.2098,0.053633,23.4509,65.65
15000,0.1864,0.0690,0.3198,,,41.28
30000,0.1564,0.0594,0.3191,0.040214,25.7926,48.00

Parser harus robust.

Jika output log berubah sedikit jangan membuat seluruh eksperimen gagal hanya karena satu metric tidak dapat diparse.

Log mentah tetap menjadi source of truth.


============================================================
10. MONITORING TRAINING
============================================================

Selama subprocess train.py berjalan catat:

training_duration_seconds

training_peak_vram_mb
training_average_vram_mb

training_peak_ram_mb
training_average_ram_mb

Gunakan nvidia-smi untuk GPU jika tersedia.

Gunakan psutil untuk RAM/process monitoring jika cocok.

Sampling:

resource_monitor.interval_seconds

Monitoring resource tidak boleh menyebabkan training gagal.

Jika nvidia-smi gagal:

training tetap berjalan
VRAM metrics = null
warning disimpan


============================================================
11. RENDERING
============================================================

Setelah training sukses jalankan render.py original.

Inspect CLI render.py yang sebenarnya terlebih dahulu.

Target utama adalah render yang dapat digunakan untuk evaluation.

Simpan stdout/stderr:

<scene_output>/research/logs/rendering.log

Monitor selama proses:

render_duration_seconds

render_peak_vram_mb
render_average_vram_mb

render_peak_ram_mb
render_average_ram_mb

Setelah rendering selesai tentukan:

render_image_count

render_seconds_per_image =
render_duration_seconds / render_image_count

render_fps =
render_image_count / render_duration_seconds

Jika image_count == 0:

fps = null
seconds_per_image = null

Jangan divide by zero.


============================================================
12. IMAGE QUALITY EVALUATION
============================================================

Setelah rendering hitung:

SSIM
PSNR
LPIPS

Gunakan hasil render dan ground-truth image.

Jika repository RaDe-GS memiliki metric implementation sendiri, reuse implementation tersebut sebisa mungkin.

Jangan membuat definisi metric berbeda tanpa alasan.

Simpan:

<scene_output>/research/image_metrics.json

Contoh:

{
  "ssim": 0.9123,
  "psnr": 28.43,
  "lpips": 0.1184,
  "image_count": 100
}

Jika evaluation gagal:

status = failed
failed_stage = image_metrics

Simpan log/error.

Lanjut scene berikutnya.


============================================================
13. RESOURCE RELEASE SEBELUM MESH
============================================================

Ini sangat penting karena pipeline dijalankan di WSL dan mesh extraction berpotensi OOM.

Setelah training/rendering selesai:

pastikan subprocess benar-benar exit

jangan ada train.py sebelumnya yang masih berjalan

jangan ada render.py sebelumnya yang masih berjalan

lakukan cooldown:

resource_guard.cooldown_before_mesh_seconds

Setelah itu cek:

free RAM
free VRAM

Jika:

free RAM < minimum_free_ram_gb

atau:

free VRAM < minimum_free_vram_mb

dan:

wait_if_memory_low = true

maka tunggu:

check_interval_seconds

lalu cek ulang sampai:

max_wait_seconds

Console contoh:

[RESOURCE] Preparing mesh extraction
[RESOURCE] Free RAM  : 3.1 GB
[RESOURCE] Free VRAM : 840 MB
[RESOURCE] Resource below threshold.
[RESOURCE] Waiting 10 seconds...

Jika tetap tidak memenuhi syarat setelah max_wait_seconds:

status = failed
failed_stage = mesh_resource_check

Jangan menjalankan mesh jika resource guard memutuskan resource tidak aman.


============================================================
14. MESH EXTRACTION FINAL
============================================================

Command mesh penelitian yang harus digunakan:

python mesh_extract.py \
  -m <scene_output> \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5

Contoh:

python mesh_extract.py \
  -m /home/pande/radegs_results/togog_bali/togog_ganesha \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5

Parameter berikut WAJIB dipertahankan:

-r 2
--num_cluster 8
--depth_ratio 0.5

Jangan otomatis menggantinya hanya karena memory tinggi.

Saya ingin parameter eksperimen tetap konsisten antar objek.

Sebelum menjalankannya pastikan parser repository benar-benar mendukung argument tersebut.


============================================================
15. OOM HANDLING
============================================================

Deteksi error seperti:

CUDA out of memory
OutOfMemoryError
std::bad_alloc
Killed
Cannot allocate memory

Jika mesh extraction terkena OOM:

status = failed
failed_stage = mesh_extraction_oom

Simpan seluruh stdout/stderr ke:

<scene_output>/research/logs/mesh_extraction.log

Jangan otomatis mencoba:

-r 4

atau:

--num_cluster lain

atau:

--depth_ratio lain

karena itu mengubah konfigurasi eksperimen.

Lebih baik eksperimen dicatat gagal daripada menghasilkan hasil dari parameter yang berbeda tanpa diketahui.


============================================================
16. MONITORING MESH EXTRACTION
============================================================

Selama mesh_extract.py berjalan catat:

mesh_extraction_duration_seconds

mesh_peak_vram_mb
mesh_average_vram_mb

mesh_peak_ram_mb
mesh_average_ram_mb

Setelah mesh berhasil dibuat cari final mesh berdasarkan output actual repository.

Kemungkinan:

recon_post.ply

dan mungkin:

recon_raw.ply

Jangan menebak.
Verifikasi dari source code.

Untuk final mesh catat:

mesh_vertex_count
mesh_face_count
mesh_file_size_mb

Gunakan trimesh atau library yang cocok untuk membaca informasi mesh.

Simpan:

<scene_output>/research/mesh_metrics.json

Contoh:

{
  "duration_seconds": 43.22,
  "peak_vram_mb": 4821,
  "average_vram_mb": 4012,
  "peak_ram_mb": 7341,
  "average_ram_mb": 6200,
  "vertex_count": 1245392,
  "face_count": 2481032,
  "file_size_mb": 83.4
}


============================================================
17. GEOMETRY EVALUATION
============================================================

Geometry evaluation bersifat optional.

Config:

geometry_evaluation:
  enabled: false
  gt_root: null
  sample_points: 100000
  distance_threshold: null
  seed: 0

Jika enabled = true:

cari Ground Truth berdasarkan scene_name.

Contoh:

geometry_gt/
├── togog_ganesha.ply
├── togog_arjuna.ply
└── togog_bima.ply

Bandingkan:

recon_post.ply
vs
ground truth

Hitung:

Chamfer Distance
Precision
Recall
F1-score

Definisi Chamfer:

mean(Reconstruction → Ground Truth)
+
mean(Ground Truth → Reconstruction)

Gunakan Euclidean distance.

Jangan squared distance.

Nearest neighbor:

scipy.spatial.cKDTree

Jika input mesh:

sampling point menggunakan trimesh.

sample_points berasal dari config.

Gunakan seed agar reproducible.

Precision:

jumlah reconstruction point dengan nearest GT distance < threshold
/
jumlah reconstruction point

Recall:

jumlah GT point dengan nearest reconstruction distance < threshold
/
jumlah GT point

F1:

2 * precision * recall
/
(precision + recall)

Gunakan STRICT:

distance < distance_threshold

BUKAN:

distance <= distance_threshold

Jika distance_threshold = null:

Chamfer tetap dihitung

Precision = null
Recall = null
F1 = null

berikan warning

scene tetap boleh success.


============================================================
18. OUTPUT PER SCENE
============================================================

Contoh:

/home/pande/radegs_results/togog_bali/
└── togog_ganesha/
    ├── point_cloud/
    ├── train/
    ├── test/
    ├── cfg_args
    ├── input.ply
    ├── recon_raw.ply
    ├── recon_post.ply
    │
    └── research/
        ├── logs/
        │   ├── training.log
        │   ├── rendering.log
        │   ├── image_metrics.log
        │   ├── mesh_extraction.log
        │   └── geometry_evaluation.log
        │
        ├── training_metrics.csv
        ├── image_metrics.json
        ├── mesh_metrics.json
        ├── geometry_metrics.json
        └── summary.json

Jangan merusak struktur output original RaDe-GS.

Folder research/ hanya berisi data tambahan penelitian.


============================================================
19. SUMMARY PER OBJEK
============================================================

Buat:

<scene_output>/research/summary.json

Contoh:

{
  "scene_name": "togog_ganesha",
  "object_name": "togog_ganesha",
  "status": "success",

  "training": {
    "initial_point_count": 344382,
    "final_iteration": 30000,
    "final_loss": 0.1564,
    "final_loss_normal": 0.0594,
    "final_loss_ncc": 0.3191,
    "final_l1": 0.040214,
    "final_psnr": 25.7926,
    "duration_seconds": 625,
    "peak_vram_mb": 10432,
    "average_vram_mb": 9310
  },

  "rendering": {
    "duration_seconds": 20.5,
    "image_count": 100,
    "seconds_per_image": 0.205,
    "fps": 4.87,
    "ssim": 0.91,
    "psnr": 28.4,
    "lpips": 0.12
  },

  "mesh_extraction": {
    "resolution": 2,
    "num_cluster": 8,
    "depth_ratio": 0.5,
    "duration_seconds": 43.2,
    "peak_vram_mb": 4821,
    "peak_ram_mb": 7341,
    "vertex_count": 1245392,
    "face_count": 2481032,
    "file_size_mb": 83.4
  },

  "geometry": {
    "chamfer_distance": null,
    "precision": null,
    "recall": null,
    "f1_score": null
  }
}

Nilai yang tidak tersedia gunakan null.


============================================================
20. MASTER CSV
============================================================

Buat:

research_results/experiment_results.csv

Satu scene + satu konfigurasi eksperimen = satu row.

Minimal kolom:

experiment_id
timestamp

scene_name
object_name
dataset_path
output_path

status
failed_stage
error_message

camera_count
training_camera_count
test_camera_count
initial_point_count
camera_extent

final_iteration
final_loss
final_loss_normal
final_loss_ncc
final_l1
final_training_psnr
final_it_per_second

training_duration_seconds
training_peak_vram_mb
training_average_vram_mb
training_peak_ram_mb
training_average_ram_mb

render_duration_seconds
render_peak_vram_mb
render_average_vram_mb
render_peak_ram_mb
render_average_ram_mb
render_image_count
render_seconds_per_image
render_fps

ssim
psnr
lpips

mesh_resolution
mesh_num_cluster
mesh_depth_ratio

mesh_extraction_duration_seconds
mesh_peak_vram_mb
mesh_average_vram_mb
mesh_peak_ram_mb
mesh_average_ram_mb

mesh_vertex_count
mesh_face_count
mesh_file_size_mb

chamfer_distance
precision
recall
f1_score
geometry_sample_points
distance_threshold

git_commit
git_dirty

python_version
pytorch_version
cuda_version
gpu_name


============================================================
21. EXPERIMENT ID
============================================================

Jangan menggunakan scene_name sebagai satu-satunya ID.

Buat deterministic experiment_id berdasarkan:

scene_name
training configuration
render configuration
mesh configuration
geometry configuration

Gunakan canonical JSON + SHA256 misalnya.

Tujuannya:

togog_ganesha + depth_ratio 0.5

berbeda dengan:

togog_ganesha + depth_ratio lain

Walaupun scene sama.


============================================================
22. CSV SAFETY
============================================================

Master CSV sangat penting.

Gunakan atomic write.

Contoh:

read existing CSV
→ update in memory
→ write temporary file
→ flush
→ fsync jika relevan
→ os.replace()

Jika master CSV tidak dapat ditulis:

CRITICAL ERROR

Hentikan seluruh batch.

Jangan melanjutkan penelitian tanpa pencatatan hasil yang reliable.


============================================================
23. FAILURE HANDLING
============================================================

Kegagalan satu objek tidak boleh menghentikan objek lainnya.

Contoh:

[1/4] togog_ganesha
SUCCESS

[2/4] togog_arjuna
MESH OOM
FAILED

[3/4] togog_bima
tetap dijalankan

Possible failed_stage:

dataset_validation
training
rendering
image_metrics
mesh_resource_check
mesh_extraction
mesh_extraction_oom
geometry_evaluation

Simpan error_message.

Setelah scene gagal:

write CSV
write summary
release resources
continue next scene


============================================================
24. COMMAND LINE INTERFACE
============================================================

Normal:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml

List scenes:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --list-scenes

Output contoh:

Discovered 3 scenes:

1. togog_arjuna
2. togog_bima
3. togog_ganesha

Single scene:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --scene togog_ganesha

Dry run:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --dry-run

Resume:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --resume

Force:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --force

--resume dan --force tidak boleh digunakan bersamaan.


============================================================
25. DRY RUN
============================================================

Dry-run WAJIB:

discover dataset
validate config
validate dataset
build commands
print commands

TIDAK BOLEH:

menjalankan CUDA
menjalankan training
menjalankan mesh
menulis fake metric
menulis status success ke CSV

Contoh output:

============================================================
RaDe-GS Research Pipeline
Dataset: Dataset Togog Bali
============================================================

Found 3 datasets.

[1/3] togog_ganesha

TRAIN:
python train.py \
  -s /home/pande/dataset_togog_bali/togog_ganesha \
  -m /home/pande/radegs_results/togog_bali/togog_ganesha

MESH:
python mesh_extract.py \
  -m /home/pande/radegs_results/togog_bali/togog_ganesha \
  -r 2 \
  --num_cluster 8 \
  --depth_ratio 0.5


============================================================
26. RESUME
============================================================

Resume hanya skip experiment jika:

experiment_id sama

dan:

status = success

Scene failed harus boleh dicoba kembali.

Jangan hanya membandingkan scene_name.


============================================================
27. FORCE
============================================================

--force selalu menjalankan experiment lagi.

Tetapi:

JANGAN otomatis delete output lama.

Jangan hapus model, checkpoint, mesh, atau log tanpa explicit action pengguna.


============================================================
28. REPRODUCIBILITY
============================================================

Catat:

timestamp
git commit
git dirty
Python version
PyTorch version
CUDA version
GPU name

Jika tersedia juga catat:

driver version

Jangan mengubah seed model RaDe-GS secara diam-diam.


============================================================
29. REQUIREMENTS
============================================================

requirements-research.txt hanya berisi dependency tambahan pipeline.

Jangan duplikasi seluruh requirements RaDe-GS.

Kemungkinan:

PyYAML
psutil
trimesh
scipy

Tambahkan dependency lain hanya jika diperlukan.


============================================================
30. CODE QUALITY
============================================================

Gunakan:

Python type hints
pathlib.Path
dataclass jika cocok
fungsi kecil
modular design
logging
explicit exception handling
tidak menggunakan bare except
tidak hardcode dataset scene
tidak hardcode jumlah dataset

Target utama:

Linux / WSL
NVIDIA GPU

Gunakan subprocess seperti:

subprocess.Popen([
    sys.executable,
    "train.py",
    "-s",
    str(dataset_path),
    "-m",
    str(output_path),
])

Hindari shell=True jika tidak diperlukan.


============================================================
31. DOKUMENTASI
============================================================

Buat:

RESEARCH_PIPELINE.md

Jelaskan:

tujuan pipeline
struktur Dataset Togog Bali
format COLMAP
dataset auto discovery
training
rendering
mesh extraction
OOM protection
metric collection
folder output
master CSV
dry-run
resume
force
failure handling
geometry evaluation
reproducibility


Buat:

docs/research_notes.md

Jelaskan metodologi:

Citra Dataset Togog Bali
        ↓
COLMAP preprocessing
        ↓
Dataset hasil COLMAP
        ↓
RaDe-GS Research Pipeline
        ↓
Training RaDe-GS
        ↓
Rendering
        ↓
Image Evaluation
        ↓
Mesh Extraction
        ↓
Geometry Evaluation
        ↓
Research Results

Jelaskan secara eksplisit:

COLMAP preprocessing berada SEBELUM Research Pipeline RaDe-GS.

Pipeline ini tidak mengubah algoritma RaDe-GS.

Pipeline hanya mengotomatisasi execution, monitoring, evaluation, dan result collection.


============================================================
32. URUTAN IMPLEMENTASI
============================================================

Kerjakan bertahap.

PHASE 1

Inspect repository.
Implement config.
Implement scene discovery.
Implement dataset validation.
Implement CLI.
Implement --list-scenes.
Implement --dry-run.

TEST.

PHASE 2

Implement training subprocess.
Implement raw logging.
Implement serial execution.
Implement failure handling.

TEST.

PHASE 3

Implement training resource monitor.
Implement training log parser.
Implement training_metrics.csv.

TEST.

PHASE 4

Implement rendering.
Implement render monitor.
Implement SSIM.
Implement PSNR.
Implement LPIPS.

TEST.

PHASE 5

Implement resource guard.
Implement WSL memory check.
Implement mesh extraction.
Implement OOM detection.
Implement mesh monitor.
Implement mesh analyzer.

TEST.

PHASE 6

Implement optional geometry evaluation.

TEST.

PHASE 7

Implement summary.json.
Implement master CSV.
Implement experiment_id.
Implement resume.
Implement force.
Implement atomic CSV writing.

TEST.

PHASE 8

Documentation.
Cleanup.
Final validation.


============================================================
33. FINAL VALIDATION
============================================================

Setelah implementasi selesai:

jalankan syntax check seluruh file.

Kemudian jalankan:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --list-scenes

Kemudian:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --dry-run

Pastikan:

semua Dataset Togog Bali ditemukan

nama folder menjadi scene_name

command train benar

command mesh persis menggunakan:

-r 2
--num_cluster 8
--depth_ratio 0.5

dry-run tidak menjalankan CUDA

dry-run tidak menulis fake results

pipeline tidak mengubah algoritma RaDe-GS

pipeline benar-benar sequential

resource guard aktif sebelum mesh

output tiap objek terpisah

raw logs tersimpan

structured metrics tersimpan

master CSV tersedia


============================================================
34. WORKFLOW AKHIR YANG SAYA INGINKAN
============================================================

Setelah pipeline selesai dibuat, workflow saya harus sesederhana:

conda activate radegs

cd ~/RaDe-GS

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --list-scenes

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --dry-run

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml

Jika proses sempat terhenti:

python research_pipeline/run_all.py \
  --config configs/togog_bali.yaml \
  --resume


Saya ingin hasil akhirnya:

Dataset Togog Bali
        ↓
auto discovery
        ↓
togog_ganesha
        ↓
train
render
metrics
mesh
evaluation
save
        ↓
togog_arjuna
        ↓
train
render
metrics
mesh
evaluation
save
        ↓
togog_bima
        ↓
...
        ↓
experiment_results.csv


Jangan berhenti setelah hanya membuat skeleton.

Implementasikan pipeline sampai dapat digunakan.

Namun jangan menjalankan training RaDe-GS nyata secara otomatis hanya untuk testing.

Gunakan dry-run untuk validation jika dataset/GPU nyata belum tersedia.

Setelah selesai, berikan ringkasan:

file apa saja yang dibuat
file apa saja yang diubah
alasan setiap perubahan
contoh config
contoh command menjalankan pipeline
hasil dry-run
hal yang masih perlu diuji menggunakan GPU sebenarnya