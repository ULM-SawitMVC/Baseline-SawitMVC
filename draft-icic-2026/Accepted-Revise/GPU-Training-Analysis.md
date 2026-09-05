# Analisis hasil GPU YOLO11m — 5 September 2026

`git pull --ff-only` berhasil dari `a175c606` ke **`2d73b6ed`**, tanpa konflik.
Commit `fb975ca4` memuat evaluasi a12; `2d73b6ed` mengarsipkan checkpoint dan
artefak training. Tidak ada training ulang atau push pada audit lokal ini.

## Hasil yang telah diverifikasi

Semua angka berikut memakai **Ridge+F0**, dilatih pada 716 pohon train,
kemudian dievaluasi pada 141 pohon test yang sama.

| Input | Class ±1 | Tree ±1 | Macro MAE | Total MAE | Total ±1 |
|---|---:|---:|---:|---:|---:|
| YOLO11m | 73,76% | 26,24% | 1,087 | 1,950 | 43,97% |
| YOLO26m | 76,06% | 28,37% | 1,053 | 1,929 | 48,23% |
| GT | 97,70% | 90,78% | 0,275 | 0,972 | 78,01% |

YOLO11m − YOLO26m: **−2,30 pp**, CI 95% **−4,96 hingga +0,35**, exact paired
p = **0,118**. Skor YOLO11m lebih rendah secara deskriptif, tetapi belum ada
bukti signifikan untuk mengurutkan kedua pipeline pada Class ±1. Ini juga
bukan bukti bahwa keduanya ekuivalen. Tree ±1 p = 0,648; macro MAE p = 0,157.

GT − YOLO11m: **23,94 pp**, CI 95% **20,57–27,30**, exact p < 0,001.
Kesimpulan bahwa ada gap besar antara input GT dan input detektor bertahan
pada keluarga detektor tambahan ini. Angka YOLO26m **76,06%** di atas memakai
F0; headline **77,48%** di naskah memakai F_all, sehingga kedua angka tersebut
tidak boleh diperlakukan sebagai hasil konfigurasi yang sama.

## Apa yang terjadi selama training

- Training lengkap 60 epoch; waktu kumulatif log **4.658,94 detik ≈ 77,6 menit**.
- Batch 32, imgsz 640, seed 42, patience 60, deterministic, workers **12**.
- Checkpoint mencatat Ultralytics **8.4.140** dan 20.056.092 parameter.
- Validation mAP50–95 maksimum pada **epoch 16: 0,25077**; mAP50 pada epoch
  yang sama **0,52450**, precision **0,50574**, recall **0,56570**.
- Pada epoch 60, mAP50–95 turun ke **0,21351** dan mAP50 ke **0,49433**.
  Loss training terus turun sementara validation box/DFL loss meningkat pada
  bagian akhir. Pola ini konsisten dengan overfitting; penyebab tunggalnya
  tidak dapat dipastikan dari kurva saja.
- Evaluasi a12 menggunakan **best.pt**, bukan last.pt. Tidak perlu mengganti
  hasil paper dengan epoch terakhir. Menambah epoch saja belum didukung oleh
  tren validation yang tersedia.

Checkpoint tersimpan telah di-strip sehingga field `epoch` bernilai −1;
epoch terbaik diinferensikan dari kecocokan fitness/mAP dengan kurva,
bukan dibaca sebagai epoch dari field tersebut.

## Sumber error yang masih terlihat

Pada 2.612 appearance GT di test set:

| Hasil matching | YOLO11m | YOLO26m |
|---|---:|---:|
| Matched, kelas benar | 1.237 | 1.252 |
| Matched, kelas salah | 399 | 396 |
| Appearance GT tidak matched | 976 | 964 |
| Deteksi tidak matched | 463 / 2.099 | 706 / 2.354 |

YOLO11m menghasilkan lebih sedikit deteksi unmatched, tetapi jumlah appearance
yang hilang/salah kelas hampir sama. Penurunan unmatched detections saja tidak
menjamin counting membaik. Matching ini memakai IoU ≥ 0,5 dan confidence 0,25;
angka recall matching test berbeda protokol dari recall pada log validation.

| Kelas | Class ±1 YOLO11m | Class ±1 YOLO26m | MAE YOLO11m |
|---|---:|---:|---:|
| B1 | 94,33% | 95,74% | 0,411 |
| B2 | 79,43% | 80,14% | 1,121 |
| B3 | 51,77% | 56,03% | 1,638 |
| B4 | 69,50% | 72,34% | 1,177 |

B3 tetap paling sulit untuk counting. Confusion YOLO11m mencatat 172 appearance
B2 diprediksi B3 dan 526 appearance B3 tidak matched. Angka ini merupakan
diagnostik, bukan pembagian kausal dari error counter.

## Integritas dan implikasi revisi

[a13_gpu_result_audit.py](../../experiments/revision/a13_gpu_result_audit.py)
memeriksa hash checkpoint terhadap seluruh cache, identitas pohon, sisi gambar,
split, confidence, kelas dan koordinat finite. Cakupan lengkap: **953 pohon /
3.992 gambar**, terbagi **716/96/141 pohon** dan **3.000/404/588 gambar**.
Counter direfit dari cache; seluruh metrik, p-value, CI, prediksi counting dan
confusion matrix cocok dengan commit GPU. Hasil audit tersimpan pada
[a13_gpu_audit.json](../../results/revision/a13_gpu_audit.json).

SHA-256 best checkpoint:
`306f93948c9ec1fb5a305644cae3f38e9c29e99c2b695392f6e33c96a9215838`.
Provenance JSON asli belum memiliki `train_workers`; arsip args dan checkpoint
sama-sama mencatat workers=12. Audit mempertahankan hasil asli dan mencatat
keterangan ini secara terpisah. Path `/workspace/...` adalah path mesin GPU.

Permintaan reviewer tentang **arsitektur tambahan sekarang terpenuhi** untuk
eksperimen ini. Naskah, Tabel VI, dan respons reviewer telah diperbarui. Namun,
YOLO26 menggunakan versi training Ultralytics 8.4.49; versi/default arsitektur
dan satu run per model membatasi atribusi perbedaan hanya kepada arsitektur.
Test set juga tetap dataset historis yang sama. Pitch kamera nyata dan kebun
independen belum diuji oleh eksperimen ini.
