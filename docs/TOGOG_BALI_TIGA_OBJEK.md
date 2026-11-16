# Tiga objek baru: baseline dan Scaffold, hasil di folder proyek

Objek yang diproses: `hanoman`, `anggada`, dan `dewi_saraswati`.
Kode, environment, dan dataset dibaca dari checkout Ubuntu-22.04-E:
`/home/pande/projects/RaDe-GS/dataset/togog_bali/<objek>/images/` dan `sparse/0/`.

Gunakan konfigurasi utama `configs/togog_bali.yaml` dan
`configs/togog_bali_scaffold_init.yaml`. Semua hasil berikut berada di dalam
folder proyek, seperti penyimpanan sebelum dipindahkan. Path dihitung dari root
repositori, sehingga mengikuti filesystem tempat checkout proyek disimpan.

| Kebutuhan | Lokasi relatif terhadap root proyek |
| --- | --- |
| Model, mesh, render, dan log baseline | output/hasil_radegs_v2/togog_bali/<objek>/ |
| Model, mesh, render, dan log eksperimen | output/hasil_radegs_v2/scaffold_anchor_init/togog_bali/<objek>/ |
| CSV dan paket laporan baseline | research_results/hasil_radegs_v2/ |
| CSV dan paket laporan eksperimen | research_results/hasil_radegs_v2/scaffold_anchor_init/ |

Resolusi 1920, split --eval, iterasi 30000, evaluator metric.py, serta parameter
mesh mengikuti konfigurasi utama masing-masing.

## Persiapan dan pemeriksaan penyimpanan di Ubuntu

Setelah kode terbaru masuk server:

```bash
conda activate radegs
cd /home/pande/projects/RaDe-GS
export SCAFFOLD_PYTHON=/home/pande/miniconda3/envs/scaffold_gs/bin/python

df -h .

python -B scripts/check_output_storage.py \
  --config configs/togog_bali.yaml configs/togog_bali_scaffold_init.yaml
```

Pemeriksaan menguji tulis, baca, os.replace, fsync, dan hardlink/copy fallback
menggunakan fungsi pipeline yang sama. Pemeriksaan hanya memakai file kecil pada
folder uji unik dan membersihkan file ujinya; folder output/laporan dapat dibuat.
File hasil lama tidak diubah. Jika muncul kegagalan izin atau filesystem,
berhenti sebelum training dan periksa pesan error. Kelulusan uji bukan jaminan
kapasitas seluruh run; ruang kosong dicetak untuk diperiksa pengguna.

Scaffold menyiapkan dua input turunan. Foto memakai hardlink jika tersedia,
atau salinan jika hardlink tidak dapat digunakan. Sediakan ruang untuk input
turunan, render, model, dan mesh.

## Pastikan tiga dataset valid dan rencana benar

Pemeriksaan environment eksperimen pada ketiga objek:

```bash
(
  for objek in hanoman anggada dewi_saraswati
  do
    python -B -m research_pipeline.scaffold_environment \
      --config configs/togog_bali_scaffold_init.yaml \
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
      --config configs/togog_bali.yaml configs/togog_bali_scaffold_init.yaml \
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
      --config configs/togog_bali.yaml configs/togog_bali_scaffold_init.yaml \
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
CSV yang ditetapkan pada konfigurasi utama dan melewati eksperimen sukses dengan
konfigurasi yang cocok. Pemilihan --scene di loop membatasi pemrosesan pada tiga
objek tersebut. Catatan hasil mengikuti CSV di folder proyek.

Pemakaian VRAM tetap bergantung pada jumlah foto dan anchor tiap objek. Pilih
run satu objek dahulu bila ingin memeriksa kapasitas sebelum menjalankan loop.

## Membuka laporan dan gambar

Buka CSV ringkasan atau MULAI_DI_SINI.html pada dua folder laporan di tabel atas.
HTML adalah navigasi; tabel tetap CSV dan pasangan foto berupa PNG. Model Gaussian
ada di point_cloud/iteration_30000/point_cloud.ply, mesh di recon.ply dan
recon_post.ply pada folder run. Percobaan berikutnya dapat berada di runs/<id>/.

Untuk mengumpulkan laporan baseline dan Scaffold di folder proyek:

```bash
python -B scripts/collect_research_reports.py \
  --config configs/togog_bali.yaml configs/togog_bali_scaffold_init.yaml \
  --pairs 5
```

Collector ini membaca seluruh hasil pada output.root kedua konfigurasi utama.
Pemilihan tiga objek berlaku pada perintah run di atas; collector dapat memasukkan
objek lain yang sudah tersimpan pada root yang sama. Arsip dari lokasi lain tidak
digabungkan atau dipindahkan secara otomatis.
