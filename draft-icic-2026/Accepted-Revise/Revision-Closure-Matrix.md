# Status revisi ICIC 2026 — 5 September 2026

Naskah aktif: [main-new.pdf](../main-new.pdf), dengan sumber
[main-new.tex](../main-new.tex). Respons per reviewer:
[Response-to-Reviewers.md](Response-to-Reviewers.md).
Versi anonim: [main-blind.pdf](../main-blind.pdf).

Revisi mencakup substansi, statistik, dan tata letak. **Arsitektur detektor kedua
sudah dievaluasi:** hasil GPU YOLO11m ditarik hingga commit `2d73b6ed` dan
direproduksi secara lokal. Pitch kamera nyata masih memerlukan data tambahan.
Pembaruan naskah dari audit GPU ini belum di-push atau diunggah ke konferensi.

## Pemetaan seluruh komentar

| Komentar | Perubahan dan bukti | Status |
|---|---|---|
| R1 — implikasi, manfaat, kekurangan, arah riset | Kesimpulan dan keterbatasan menjelaskan implikasi counting, detector recall/confusion, asosiasi, dan validasi lanjutan. | Selesai pada naskah |
| R2 — satu detektor | Empat checkpoint YOLO26 dan training YOLO11m pada split resmi; gap GT tetap 23,94 pp dengan Ridge+F0. | Selesai untuk evaluasi keluarga tambahan; generalisasi tetap dibatasi |
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
| R4 — arsitektur tambahan | Training YOLO11m 60 epoch, best checkpoint, cache 953 pohon, paired evaluation a12, dan reproduksi CPU a13 tersedia. | Selesai |
| R4 — pitch nyata | Uji affine/noise, keterbatasan data pitch, batas generalisasi dan rencana pengambilan gambar bersudut terukur dijelaskan pada III-E, III-G dan IV. | Klarifikasi ditanggapi; validasi pitch fisik menjadi future work |

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

Pada pembaruan 6 September, seluruh tanda strip di Tabel III/VI dilengkapi
dengan perhitungan [a14](../../experiments/revision/a14_complete_table_metrics.py).
Global divisor dikalibrasi terpisah per kondisi pada 716 pohon train (k GT
1,891; fixed 1,793); hasil fixed **70,21% / 22,70% / MAE 1,188**. Jumlah deteksi
checkpoint dihitung pada 64 pohon evaluasi yang sesuai. GT berisi jumlah
appearance anotasi (2.612 pada 141 test; 1.203 pada 64 evaluasi), bukan jumlah
tandan unik. Recall threshold dihitung ulang dari cache low-confidence setelah
filtering. Recall 0,25 filtered **0,441** berbeda dari cache asli **0,442**.
Hasil lengkap, confusion counts dan prediksi divisor tersedia di
[a14_completed_table_metrics.json](../../results/revision/a14_completed_table_metrics.json),
[a14_appearance_metrics.csv](../../results/revision/a14_appearance_metrics.csv) dan
[a14_global_divisor_predictions.csv](../../results/revision/a14_global_divisor_predictions.csv).

## Hasil GPU dan eksperimen yang tersisa

Training YOLO11m telah selesai 60 epoch pada mesin GPU penulis. Audit memverifikasi
hash checkpoint, **953 pohon / 3.992 gambar**, split **716/96/141**, dan reproduksi
seluruh metrik, paired tests, prediksi counting serta confusion matrix.
Dengan Ridge+F0 yang sama, Class ±1 YOLO11m **73,76%**, YOLO26m **76,06%**,
dan GT **97,70%**. Gap GT–YOLO11m **23,94 pp** (CI 20,57–27,30; p < 0,001).
Selisih antardetektor belum signifikan (−2,30 pp; CI −4,96–0,35; p = 0,118).
Hasil ini sudah masuk Tabel VI dan respons reviewer.

Lihat [GPU-Training-Analysis.md](GPU-Training-Analysis.md) untuk diagnosis kurva,
error per kelas dan provenance. Best checkpoint cocok dengan epoch 16 menurut
fitness validation; evaluasi menggunakan best.pt, bukan last.pt. Perbedaan versi
software/default dan satu training run membatasi klaim ranking arsitektur.
Notebook/ZIP lama tetap menjadi rekam persiapan; run selesai menggunakan script
a12 pada mesin GPU. Reproduksi audit tanpa training:

```powershell
python experiments/revision/a13_gpu_result_audit.py
```

Pitch nyata memerlukan gambar berulang dari pohon yang sama dengan sudut kamera
terukur, lalu inference dan evaluasi ulang. Data baru itu belum tersedia.
Validasi full pipeline pada kebun independen juga memerlukan kebun yang tidak
pernah digunakan untuk training detektor.

## Verifikasi dan batas administrasi

- Enam pengujian audit statistik/seleksi lulus.
- Main dan blind berhasil dibangun; pemeriksaan overflow horizontal dan referensi
  terdefinisi dilakukan oleh `scripts/build_icic_revision.py`.
- PDF utama dan anonim masing-masing **6 halaman**, mengikuti batas yang
  diminta penulis. Keenam tabel, kedua gambar dan 23 referensi tetap dipertahankan.
  Rincian pemadatan dan cakupan reviewer ada di
  [Six-Page-Revision-Audit.md](Six-Page-Revision-Audit.md).
- Seluruh 21 font pada masing-masing PDF terbenam dan ter-subset; 23 sitasi
  mengikuti urutan kemunculan pertama. Seluruh halaman diperiksa secara visual.
- Kepatuhan format ICIC/IEEE Xplore diverifikasi ulang pada 7 September 2026 dan
  kini diasersi otomatis oleh `scripts/build_icic_revision.py`: ukuran halaman
  A4 595x842 pt, nol anotasi tautan, nol bookmark, maksimum enam halaman, dan
  tidak ada em dash. Sebelum pemeriksaan ini naskah masih terkompilasi pada
  US Letter dan memuat 49 tautan serta 17 bookmark dari `hyperref`.
- Foto pada Gambar 1 kini tertanam pada 424 DPI efektif, naik dari 212 DPI,
  memenuhi anjuran IEEE 300 DPI untuk foto. Ukuran PDF 1,5 MB, di bawah batas
  2 MB.
- Training/inference YOLO11 selesai pada GPU; hasilnya direproduksi oleh audit
  CPU a13. Seluruh artefak GPU asli dipertahankan.
- Dokumen belum melalui PDF eXpress atau diunggah sebagai camera-ready.
- Urutan penulis pada sumber berbeda dari daftar EasyChair yang dilampirkan;
  metadata penulis tidak diubah otomatis. Cocokkan sebelum submit.

Audit tahap pertama tetap tersedia di
[Review-Audit-2026-09-05.md](Review-Audit-2026-09-05.md) sebagai rekam perubahan.
