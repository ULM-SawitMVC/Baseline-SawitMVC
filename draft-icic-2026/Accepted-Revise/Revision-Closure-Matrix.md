# Status revisi ICIC 2026 — 5 September 2026

Naskah aktif: [main-new.pdf](../main-new.pdf), dengan sumber
[main-new.tex](../main-new.tex). Respons per reviewer:
[Response-to-Reviewers.md](Response-to-Reviewers.md).
Versi anonim: [main-blind.pdf](../main-blind.pdf).

Revisi mencakup substansi, statistik, dan tata letak. **Belum semua permintaan
eksperimen selesai secara empiris:** arsitektur detektor kedua dan pitch kamera
nyata masih memerlukan eksekusi/data tambahan. Klaim naskah sudah dibatasi sesuai
bukti yang tersedia; paket ini belum diunggah ke konferensi atau GitHub.

## Pemetaan seluruh komentar

| Komentar | Perubahan dan bukti | Status |
|---|---|---|
| R1 — implikasi, manfaat, kekurangan, arah riset | Kesimpulan dan keterbatasan menjelaskan implikasi counting, detector recall/confusion, asosiasi, dan validasi lanjutan. | Selesai pada naskah |
| R2 — satu detektor | Empat checkpoint YOLO26; batas generalisasi dinyatakan eksplisit, a12 disiapkan untuk keluarga kedua. | Parsial; keluarga kedua belum dievaluasi |
| R2 — ukuran test terbatas | Matched repeated CV pada 237 dan 141 pohon; ukuran dan ketergantungan fold dijelaskan. | Analisis tambahan selesai; tidak menambah data independen |
| R2 — signifikansi | Bootstrap berpasangan, exact paired permutation, semua 10 pasangan counter per kondisi, koreksi Holm; a10. | Selesai |
| R2 — dekomposisi error | Matching appearance, confusion matrix, objek hilang lintas-view dan aturan diagnostik; a2, Tabel I, Fig. 2. | Selesai sebagai diagnosis; bukan atribusi kausal aditif |
| R3.1 — ketidakpastian/signifikansi | CI dan endpoint statistik dibedakan. a11 menambah seleksi model/fitur pada validation. | Selesai |
| R3.2 — lingkup kesimpulan detektor | Temuan dibatasi pada konfigurasi SawitMVC yang diuji; checkpoint sensitivity bukan ranking arsitektur independen. | Selesai sebagai pembatasan klaim |
| R3.3 — missed/FP/confusion | 964 appearance tidak matched, 396 salah kelas, 706 deteksi unmatched; 317/1.397 tandan tidak matched pada semua view. | Selesai |
| R3.4 — MAE/RMSE dan total count | Metrik per kelas, bias dan error total; definisi Tree ±1 dibedakan dari total ±1. | Selesai |
| R3.5 — CV atau kebun independen | CV berpasangan dan counter-transfer dua kebun; detector frozen dan ketidakseimbangan train diungkap. | Opsi CV selesai; kebun independen belum tersedia |
| R3.6 — definisi fitur | Seluruh 67 dimensi, normalisasi, empty-class defaults dan formula dijelaskan pada teks/Tabel II. | Selesai |
| R3.7 — RF | Analisis slope, rentang prediksi dan kelompok count-load; interpretasi dibatasi. | Selesai |
| R3.8 — generalisasi berlebihan | Tidak ada klaim batas mutlak counter, ekuivalensi model, atau keunggulan fitur yang tidak signifikan. | Selesai |
| R3.9 — detector vs association | Enam aturan pada Fig. 2; tiga aturan berbasis identitas dijelaskan sebagai estimator diagnostik. Klaim bound 8,69 pp dihapus. | Selesai sebagai klarifikasi; atribusi kausal butuh modul asosiasi eksplisit |
| R3.10 — proofreading | Caption singkat, font tabel 8 pt, angka sejajar, persamaan dan urutan sitasi diperiksa. M01 dikeluarkan karena provenance development belum jelas. | Selesai |
| R4 — layout Tabel III/V | Keenam tabel diperbaiki, detail caption dipindah ke catatan atau paragraf terkait. | Selesai |
| R4 — arsitektur tambahan | Script a12, notebook GPU dan ZIP portabel. Validasi split/data selesai; belum ada hasil training baru. | Menunggu GPU/eksekusi |
| R4 — pitch nyata | Sentinel feature diperbaiki, uji affine/noise diulang. Klaim hanya feature sensitivity; bukan rotasi fisik. | Parsial; perlu gambar dengan pitch terukur |

## Hasil baru: model dipilih pada validation

Eksperimen [a11](../../experiments/revision/a11_validation_selection.py) melatih
40 kombinasi counter/fitur per kondisi pada 716 pohon train. Pemilihan memakai
96 pohon validation: Class ±1 tertinggi, macro MAE terendah, dimensi terendah,
lalu nama model/fitur. Skor test tidak masuk aturan pemilihan.

| Kondisi | Konfigurasi terpilih | Test Class ±1 | Test Tree ±1 | Macro MAE |
|---|---|---:|---:|---:|
| GT | Ridge + F0 | 97,70% | 90,78% | 0,275 |
| Detektor tetap | SVM + F0 + confidence | 74,11% | 27,66% | 1,053 |

Gap berpasangan: **23,58 pp**, CI 95% **20,21–27,13**, exact p < 0,001.
Hasil dan prediksi tersedia pada
[a11_validation_selected.json](../../results/revision/a11_validation_selected.json)
dan [a11_selected_predictions.csv](../../results/revision/a11_selected_predictions.csv).
Ini tetap memakai test historis yang sudah pernah dianalisis, sehingga bukan
replikasi pada test baru. Headline lama 98,05%/77,48% tetap dilabeli skor terbaik
yang diamati pada test, dengan keterbatasan seleksinya dijelaskan.

## Gambar dan tabel

Fig. 2 memakai dua panel sejajar selebar dua kolom. Panel (a) memiliki enam bar,
termasuk deployed counter sebagai bar sendiri. Panel (b) memiliki 11 titik
threshold; legenda dipisahkan dari area data dan anotasi yang berdesakan dihapus.
Tidak ada angka eksperimen yang diubah untuk memperindah gambar.

Sesuai permintaan, GPT Image Gen digunakan untuk membuat referensi layout:
[hasil Image Gen](../../figures/paper/fig04_imagegen.png),
[prompt lengkap](../../figures/paper/imagegen-figure-prompt.txt).
Karena panjang bar pada raster generatif tidak seluruhnya presisi, naskah memakai
[plot PDF vektor](../../figures/paper/fig04_attribution_sweep.pdf) dari data asli,
dibuat oleh [script plotting](../../scripts/generate_revision_figure.py).
Caption keenam tabel dipersingkat; definisi tetap ada dalam catatan atau teks.

## Melengkapi eksperimen yang tersisa

Untuk arsitektur kedua, buka [Reviewer-4-GPU.ipynb](Reviewer-4-GPU.ipynb) di
Colab dengan GPU dan unggah [reviewer-gpu-bundle.zip](reviewer-gpu-bundle.zip).
Notebook memuat langkah autentikasi dataset, validasi data, training, inference,
evaluasi berpasangan dan ekspor hasil. Notebook belum dijalankan.
Alternatif CLI pada mesin yang sudah memiliki GPU/dependensi/dataset:

```powershell
python experiments/revision/a12_second_detector.py --prepare-only
python experiments/revision/a12_second_detector.py --device 0
```

Persiapan lokal sudah memvalidasi **3.992 gambar**: train 3.000 / val 404 /
test 588, dengan split **716/96/141 pohon**. Training YOLO11m direncanakan
60 epoch, batch 32, imgsz 640, seed 42; checkpoint dipilih memakai validation.
Counter pembanding ditetapkan Ridge+F0. Default training khusus arsitektur
disimpan, sehingga perbandingan tidak diklaim mengisolasi pengaruh arsitektur saja.
Hasil a12 baru boleh dimasukkan ke Tabel VI setelah training/evaluasi selesai.

Pitch nyata memerlukan gambar berulang dari pohon yang sama dengan sudut kamera
terukur, lalu inference dan evaluasi ulang. Data baru itu belum tersedia.
Validasi full pipeline pada kebun independen juga memerlukan kebun yang tidak
pernah digunakan untuk training detektor.

## Verifikasi dan batas administrasi

- Enam pengujian audit statistik/seleksi lulus.
- Main dan blind berhasil dibangun; pemeriksaan overflow horizontal dan referensi
  terdefinisi dilakukan oleh `scripts/build_icic_revision.py`.
- PDF utama dan anonim masing-masing 8 halaman. Ini target layout saat ini,
  bukan konfirmasi batas halaman resmi ICIC 2026.
- Seluruh 22 font pada masing-masing PDF terbenam; 23 sitasi mengikuti urutan
  kemunculan pertama. Seluruh halaman utama diperiksa secara visual.
- Sel Python pada notebook GPU lolos pemeriksaan sintaks; training/inference
  YOLO11 belum diuji end-to-end pada GPU.
- Dokumen belum melalui PDF eXpress atau diunggah sebagai camera-ready.
- Urutan penulis pada sumber berbeda dari daftar EasyChair yang dilampirkan;
  metadata penulis tidak diubah otomatis. Cocokkan sebelum submit.

Audit tahap pertama tetap tersedia di
[Review-Audit-2026-09-05.md](Review-Audit-2026-09-05.md) sebagai rekam perubahan.
