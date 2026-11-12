# Tiga objek baru: baseline dan Scaffold, hasil di drive D

Objek yang diproses: `hanoman`, `anggada`, dan `dewi_saraswati`.
Kode, environment, dan dataset tetap dibaca dari checkout Ubuntu-22.04-E:
`/home/pande/projects/RaDe-GS/dataset/togog_bali/<objek>/images/` dan `sparse/0/`.
Tidak ada perpindahan distro WSL atau dataset dalam langkah ini.

Preset baru hanya mengganti output.root dan results.csv. Resolusi 1920, split
--eval, iterasi 30000, evaluator metric.py, serta parameter mesh tetap mengikuti
preset sumber masing-masing. Konfigurasi lama tetap tersedia.

| Kebutuhan | Lokasi Windows |
| --- | --- |
| Model, mesh, render, dan log baseline | D:\RaDeGS_results\output\hasil_radegs_v2\togog_bali\<objek>\ |
| Model, mesh, render, dan log eksperimen | D:\RaDeGS_results\output\hasil_radegs_v2\scaffold_anchor_init\togog_bali\<objek>\ |
| CSV dan paket laporan baseline | D:\RaDeGS_results\research_results\hasil_radegs_v2\ |
| CSV dan paket laporan eksperimen | D:\RaDeGS_results\research_results\hasil_radegs_v2\scaffold_anchor_init\ |

Di Ubuntu, drive D diakses melalui /mnt/d. Izin menulis mengikuti hak akses akun
Windows pada folder tujuan. Performa akses drive Windows dari WSL dapat berbeda
dari filesystem Linux; uji satu objek sebelum memperkirakan waktu seluruh run.
Rujukan Microsoft: https://learn.microsoft.com/windows/wsl/filesystems dan
https://learn.microsoft.com/windows/wsl/file-permissions .

## Persiapan dan pemeriksaan penyimpanan di Ubuntu

Setelah kode terbaru masuk server:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
export SCAFFOLD_PYTHON=/home/pande/miniconda3/envs/scaffold_gs/bin/python

findmnt -T /mnt/d
df -h /mnt/d

python -B scripts/check_output_storage.py \
  --config configs/togog_bali_drive_d.yaml configs/togog_bali_scaffold_init_drive_d.yaml
```

Pemeriksaan memastikan /mnt/d benar-benar mount sebelum membuat folder, lalu
menguji tulis, baca, os.replace, fsync, dan hardlink/copy fallback menggunakan
fungsi pipeline yang sama. Pemeriksaan hanya memakai file kecil pada folder uji
unik dan membersihkan file ujinya; folder output/laporan dapat dibuat. File hasil
lama tidak diubah. Jika muncul kegagalan izin atau filesystem, berhenti sebelum
training dan periksa pesan error. Kelulusan uji bukan jaminan kapasitas seluruh
run; ruang kosong dicetak untuk diperiksa pengguna.

Scaffold menyiapkan dua input turunan. Dari filesystem Linux ke drive D, hardlink
antar-filesystem tidak tersedia sehingga foto dapat disalin. Sediakan ruang untuk
input turunan, render, model dan mesh; jangan mengasumsikan hanya PLY yang disimpan.
Folder sementara instalasi tetap gunakan filesystem Linux seperti sebelumnya,
bukan folder /mnt/d; pengaturan ini tidak memindahkan environment.

## Pastikan tiga dataset valid dan rencana benar

Pemeriksaan environment eksperimen pada ketiga objek:

```bash
(
  for objek in hanoman anggada dewi_saraswati
  do
    python -B -m research_pipeline.scaffold_environment \
      --config configs/togog_bali_scaffold_init_drive_d.yaml \
      --scene "$objek"
    tahap_status=$?
    if [ "$tahap_status" -ne 0 ]; then exit "$tahap_status"; fi
  done
)
```

Tampilkan perintah kedua metode, hanya untuk tiga objek tersebut:

```bash
(
  for objek in hanoman anggada dewi_saraswati
  do
    python -B research_pipeline/run_all.py \
      --config configs/togog_bali_drive_d.yaml configs/togog_bali_scaffold_init_drive_d.yaml \
      --scene "$objek" \
      --start-from training \
      --dry-run
    tahap_status=$?
    if [ "$tahap_status" -ne 0 ]; then exit "$tahap_status"; fi
  done
)
```

Satu pemanggilan memproses baseline dan kemudian Scaffold untuk objek yang sama.
Urutan keseluruhan: hanoman baseline -> hanoman Scaffold -> anggada baseline ->
anggada Scaffold -> dewi_saraswati baseline -> dewi_saraswati Scaffold.
Objek lain, termasuk Ganesha lama, tidak dipilih. Tidak memakai --last-scenes,
karena dewi_gangga dan detya bukan bagian dari tiga objek baru ini.

## Jalankan enam eksperimen secara bergantian

Jalankan setelah seluruh pemeriksaan dan dry-run lolos:

```bash
(
  for objek in hanoman anggada dewi_saraswati
  do
    python -B research_pipeline/run_all.py \
      --config configs/togog_bali_drive_d.yaml configs/togog_bali_scaffold_init_drive_d.yaml \
      --scene "$objek" \
      --start-from training \
      --resume
    tahap_status=$?
    case "$tahap_status" in
      0|1) ;;
      *) exit "$tahap_status" ;;
    esac
  done
)
```

Kegagalan eksperimen biasa dicatat dan objek berikutnya diproses. Error kritis
(misalnya CSV tidak dapat ditulis) atau Ctrl+C menghentikan loop. --resume membaca
CSV yang ditetapkan pada preset drive D. CSV baru tidak otomatis berisi catatan
sukses dari lokasi lama, sehingga pemilihan --scene di loop adalah pembatas agar
hanya tiga dataset baru diproses. Identitas eksperimen tidak berubah hanya karena
output/CSV berpindah, tetapi file catatan sukses pada lokasi lama tidak disalin.

Pemakaian VRAM tetap bergantung pada jumlah foto dan anchor tiap objek. Pilih
run satu objek dahulu bila ingin memeriksa kapasitas sebelum menjalankan loop.

## Membuka laporan dan gambar

Buka CSV ringkasan atau MULAI_DI_SINI.html pada dua folder laporan di tabel atas.
HTML adalah navigasi; tabel tetap CSV dan pasangan foto berupa PNG. Model Gaussian
ada di point_cloud/iteration_30000/point_cloud.ply, mesh di recon.ply dan
recon_post.ply pada folder run. Percobaan berikutnya dapat berada di runs/<id>/.

Untuk mengumpulkan laporan dari kedua lokasi baru:

```bash
python -B scripts/collect_research_reports.py \
  --config configs/togog_bali_drive_d.yaml configs/togog_bali_scaffold_init_drive_d.yaml \
  --pairs 5
```

Collector ini membaca seluruh hasil yang tersimpan di root baru. Jangan menyalin
hasil lama ke root baru tanpa juga memeriksa path sumber, cfg_args, summary.json,
dan asal metriknya. Langkah di panduan ini tidak menggabungkan atau memindahkan
arsip lama secara otomatis.
