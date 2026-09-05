# Audit revisi ICIC 2026 — 5 September 2026

> **Catatan versi:** dokumen ini merekam audit tahap pertama. Hasil terbaru ada
> di [Revision-Closure-Matrix.md](Revision-Closure-Matrix.md) dan
> [Response-to-Reviewers.md](Response-to-Reviewers.md). Pada tahap lanjutan, M01
> dikeluarkan dari perbandingan naskah, a11 menambah seleksi berbasis validation,
> keenam tabel dan Fig. 2 didesain ulang, dan paket a12 disiapkan untuk GPU.
> Pengujian audit kini berjumlah enam. Keterangan M01 dan layout di bawah adalah
> rekam keadaan sebelum perubahan lanjutan tersebut.

Naskah aktif adalah `../main-new.tex` dan `../main-new.pdf`. `../main.tex`
tetap merupakan versi lama. Audit ini memeriksa permintaan empat reviewer,
kode a1–a9, keluaran eksperimen, dan kesesuaian klaim di revisi sebelumnya.

## Temuan penting dan perbaikan

| Temuan pada revisi sebelumnya | Perbaikan |
|---|---|
| “Maksimum gain counter/asosiasi 8,69 pp” | Dihapus sebagai bound. Oracle memakai identitas dan kelas GT serta membuang FP; counter statistik dapat mengestimasi objek yang luput. |
| Kerugian 28,19 pp vs 1,95 pp dipresentasikan sebagai atribusi kausal | Dijelaskan sebagai keluaran estimator diagnostik berbeda, bergantung urutan, bukan error budget aditif. Grafik diperbaiki. |
| Klaim semua pasangan counter sudah diuji, padahal a1 hanya memuat sebagian | a10 menghitung seluruh 10 pasangan per kondisi; exact paired permutation dan koreksi Holm. |
| Bootstrap tail fraction disebut p-value tanpa pembedaan | Nilai inferensial di naskah diganti exact paired permutation. CI bootstrap tetap dipakai; a1 lama dipertahankan sebagai rekam analisis. |
| CV GT pada 953 pohon dibandingkan dengan fixed pada 237 pohon | Ditambah CV pasangan pada pohon/fold/train size/model/fitur yang sama; juga subset 141 test-only. |
| 237/64 pohon disebut benar-benar tidak pernah dipakai detektor | Diperjelas: tidak masuk gradient training, tetapi mencakup validation yang dipakai saat pengembangan detektor. |
| Transfer kebun diklaim validasi lokasi independen | Dibatas sebagai transfer counter; detektor dilatih pada kedua kebun, ukuran train tidak setara, LONSUM hanya 24 pohon evaluasi. |
| YOLO26n/s/m disebut memenuhi permintaan arsitektur baru | Statusnya eksplisit parsial; seluruhnya satu keluarga YOLO26. |
| Pitch affine mengubah sentinel mean_cy=0,5 saat kelas kosong | Sentinel dipertahankan; rerun a5. Disebut uji feature shift, bukan rotasi kamera terkalibrasi. |
| Recall diklaim langsung mengurutkan penurunan akurasi counting | Dikoreksi: B3 mengalami kerugian terbesar meski recall test lebih tinggi dari B2/B4. |
| Bias B4 langsung disimpulkan berasal dari B3→B4 | Dihapus atribusi tunggal yang tidak dibuktikan. |
| 2.353 deteksi sweep dicampur dengan recall cache 2.354 deteksi | Provenance dipisahkan; recall filtered cache tidak diisi dengan angka cache asli. |
| M01 masih dijelaskan sebagai tiga jadwal divisor tetap | Dijelaskan sesuai adaptive/visibility/ensemble dan redistribusi B2/B3; asal development snapshot 228 pohon diberi caveat. |
| `unique_bunches` di a2 selalu 0 | Diisi jumlah tandan unik sebenarnya per kelas dan hasil dihitung ulang. |
| Versi software 26.0.0 tidak didukung | Dihapus; training log menyebut Ultralytics 8.4.49. |
| Build Tectonic mengganti body Times ke Latin Modern | Ditambah T1 font encoding; PDF memakai Nimbus Roman (keluarga Times) dengan font terbenam. |
| Blind hanya menghapus judul Acknowledgment tetapi menyisakan pendanaan | Generator baru menghapus seluruh paragraf; tautan anonymous mirror yang tidak diverifikasi diganti keterangan withheld. |

## Angka yang sudah diperiksa ulang

- Headline tetap: GT **98,05% / 92,20%**, fixed **77,48% / 32,62%**.
- Gap Class ±1: **20,57 pp**, CI 95% **17,20–23,94**, exact p < 0,001.
- Ridge+F0 pada kedua kondisi: gap **21,63 pp**, CI **18,44–25,00**.
- Fixed EN vs RF: raw p **0,0628**, Holm **0,628**, bukan p=0,05 yang diklaim sebagai batas tepat.
- Fixed F_all vs F0: raw p **0,201**, Holm **1,000**; gain 1,42 pp tidak signifikan.
- Matched CV 237: GT **97,72 ± 1,02%**, fixed **72,70 ± 2,35%**; gap **25,03 pp**.
- Matched CV 141 test-only: GT **97,16 ± 1,23%**, fixed **74,08 ± 2,96%**; gap **23,08 pp**.
- Pitch proxy yang diperbaiki: penurunan maksimal **1,42 pp**; noise σ=0,05 **0,53 pp**.
- MAE/RMSE serta metrik utama a10 dicocokkan dengan a1 melalui assertion numerik.

Simpangan baku CV adalah variasi fold, bukan CI; fold berulang tidak independen.
Model/fitur terbaik sebelumnya disorot setelah membandingkan skor test. CI dan
p-value tidak mengoreksi proses seleksi ini. Konfigurasi baseline sengaja
dipertahankan, termasuk scaling Ridge/ElasticNet sebelum internal CV; data
outer-test tetap tidak ikut fitting. Ini dijelaskan, bukan disembunyikan.

## Status permintaan yang belum sepenuhnya terpenuhi

1. **Reviewer 4: arsitektur detektor berbeda.** Tersedia hanya checkpoint YOLO26n/s/m.
   CUDA tidak tersedia pada runtime lokal. Belum ada model dari keluarga lain yang
   dilatih pada split yang sesuai. Jangan menyebut permintaan ini sudah selesai.
   Eksperimen yang dibutuhkan: latih detector family kedua pada 716 train,
   gunakan 96 val untuk pemilihan, bekukan sebelum evaluasi 141 test, dan fit
   counter yang sama. Tidak ada training baru atau biaya cloud dijalankan dalam audit ini.
2. **Reviewer 4: pitch kamera nyata.** Memerlukan data dengan sudut terukur atau
   kamera terkalibrasi; perturbasi mean_cy tidak menguji perubahan oklusi/deteksi.
3. **Independent plantation/full-pipeline validation.** Belum tersedia perkebunan
   yang sama sekali dikecualikan dari training detektor.
4. **Provenance M01.** Keanggotaan 228-tree development snapshot terhadap split
   test kini belum diverifikasi; baseline heuristik dilabeli historical reference.

## Surat penerimaan dan administrasi

PDF `Email-Acceptancec.pdf` **bisa dibaca secara visual** setelah dirender;
kegagalan ekstraksi teks bukan berarti dokumennya tidak dapat diperiksa.

- Paper 57 diterima untuk presentasi oral.
- Camera-ready: **25 September 2026**.
- Early-bird: **12 September 2026**; registrasi reguler: **25 September 2026**.
- IEEE PDF eXpress conference ID: **71916X**.
- Konferensi: **8–9 Oktober 2026**, hybrid, Hotel Sahid Raya Yogyakarta.
- Surat **tidak menyebut batas halaman**. Website ICIC 2026 yang dicoba
  mengembalikan HTTP 403; aturan ICIC 2025 tidak dipakai sebagai aturan 2026.
  Draf dipertahankan 8 halaman sebagai target kerja revisi sebelumnya,
  bukan klaim bahwa batas 8 halaman sudah disahkan panitia.
- Urutan penulis di `main-new.tex` berbeda dari daftar EasyChair dalam Revise.txt.
  Nama penulis tidak diubah otomatis; perlu konsistensi metadata saat submit.
- PDF saat ini berukuran US Letter mengikuti source lama. Ukuran kertas dan
  batas halaman perlu dicocokkan dengan ketentuan camera-ready 2026 yang berlaku.
- Belum dilakukan upload camera-ready, pengecekan PDF eXpress, registrasi,
  atau publikasi artefak revisi ke GitHub.

## Reproduksi audit

```powershell
python experiments/revision/a10_revision_audit.py
python experiments/revision/a2_error_decomposition.py
python experiments/revision/a5_spatial_robustness.py
python experiments/revision/test_revision_audit.py
python scripts/generate_revision_figure.py
python scripts/build_icic_revision.py
```

Audit a10 menyimpan prediksi per pohon, tabel p-value lengkap, fold CV, dan
versi dependensi. `test_revision_audit.py` membandingkan exact null dengan
enumerasi SciPy dan memeriksa koreksi Holm serta integritas CV berpasangan.
Build memeriksa horizontal overflow, referensi tak terdefinisi, dan duplikasi
label; inspeksi visual diperlukan untuk layout akhir. Peringatan underfull
serta fallback font small-caps italic dapat muncul dari template; jangan
menyamakan kompilasi lokal dengan kelulusan resmi PDF eXpress.

Sumber metodologi untuk mekanisme pertukaran berpasangan:
[SciPy permutation_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
Dokumentasi resmi mengelompokkan n/s/m sebagai ukuran dalam keluarga YOLO26:
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).

## Hasil verifikasi akhir

- Empat pengujian audit statistik lulus.
- PDF utama dan anonim: masing-masing 8 halaman.
- Seluruh 28 font pada PDF utama terbenam.
- Urutan 23 sitasi sesuai daftar pustaka; seluruh label rujukan terdefinisi.
- Tidak ada overfull horizontal box pada kedua build. Peringatan kecil vertical box dari penyeimbangan kolom tetap tercatat di log; inspeksi halaman tidak menemukan konten terpotong.
- Seluruh delapan halaman naskah utama diperiksa secara visual; tabel I–VI dan grafik terbaca.
- Nama/email penulis serta seluruh paragraf pendanaan dihapus dari companion anonim; sitasi ilmiah dataset tetap dipertahankan.
