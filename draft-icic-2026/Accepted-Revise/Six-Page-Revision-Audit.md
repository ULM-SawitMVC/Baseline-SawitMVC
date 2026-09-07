# Audit penyajian enam halaman

Pembaruan 6 September 2026. Naskah utama dan anonim masing-masing **6 halaman**.
Audit ulang terhadap commit `f9eb4e1c` menunjukkan bahwa versi ringkas pertama
terlalu banyak memangkas kalimat penjelas. Retensi angka tabel tidak cukup untuk
menyimpulkan bahwa alur argumentasi tetap utuh. Versi sekarang memulihkan alasan
perbandingan GT/detektor, hubungan antarpengujian, interpretasi statistik, dan
penjelasan diagnosis kesalahan. Eksperimen yang menanggapi reviewer tetap
dibahas dalam naskah; batas interpretasinya tidak dipindahkan hanya ke respons.

Ini tetap penyuntingan enam halaman, bukan pemulihan seluruh kalimat dari versi
delapan halaman. Uraian yang mengulang tabel dan rangkuman hasil yang berulang
antarbagian masih diringkas. Pemetaan di bawah membedakan cakupan reviewer dari
retensi kata per kata.

## Perubahan penyajian

- Penjelasan yang berulang antara abstrak, metode, hasil dan kesimpulan disatukan.
  Hasil lengkap tetap berada pada tabel dan pembahasan terkait.
- Vektor target dan estimator naive dijelaskan secara inline; dua rumus akurasi
  disajikan bersama. Definisi MAE, RMSE, bias dan total-count error tetap lengkap.
- Blok afiliasi menggabungkan universitas/lokasi yang sama. Nama, urutan penulis,
  ketiga departemen, fakultas, email dan keterangan pendanaan dipertahankan.
- Tiga foto Gambar 1 disusun dalam satu kolom. Sumber foto, crop dan bounding
  box tetap sama; ruang vertikal dan label disesuaikan agar terbaca pada ukuran
  tersebut. Gambar 2 tetap dua kolom dan tidak diubah.
- Pengantar kembali menjelaskan masalah sensus, pengulangan penampakan, posisi
  pendekatan regresi, dan alasan membandingkan dua kondisi input. Definisi
  Class/Tree accuracy kembali disertai penjelasan verbal sebelum persamaan.
- Hasil utama dibagi menjadi paragraf kondisi GT, kondisi detektor, dan pola
  per kelas. Diagnosis menjelaskan prosedur matching, identitas tandan,
  aturan voting, dan alasan selisih estimator bukan dekomposisi kausal.
- Bahasa diperbaiki, termasuk ungkapan seperti "spread and floor" menjadi
  "SD and minimum", serta penjelasan RF yang semula menggunakan ungkapan
  "thinly populated leaves" menjadi penjelasan langsung tentang sedikitnya
  contoh dengan jumlah tandan tinggi.
- Em dash di abstrak, keywords dan label gambar dihapus. Tanda minus serta
  rentang angka tetap memiliki fungsi matematisnya.

## Cakupan komentar reviewer dalam naskah

| Komentar | Lokasi dan informasi yang dipertahankan |
|---|---|
| R1: kesimpulan, implikasi, manfaat dan kekurangan | III-G dan IV: penggunaan untuk BBC, kematangan versus inventaris, bias, reproduksibilitas, dua kebun yang tidak seimbang, batas akuisisi dan arah penelitian. |
| R2 dan R3.1: signifikansi/ketidakpastian | II-E; III-A sampai III-C: 10.000 bootstrap pohon, paired exact tests, Holm, endpoint McNemar/Wilcoxon, CI/p-value, batas inferensi dan seleksi validation. |
| R2, R3.2 dan R4: detektor tambahan/lingkup | II-B, III-E, Tabel VI: YOLO11m, checkpoint YOLO26, protokol split, versi software, threshold dan paired GT gap. |
| R2, R3.3 dan R3.9: diagnosis detector/association | III-D, Tabel I dan Gambar 2: matching IoU, missed/wrong/FP, unique-bunch misses, tiga aturan identitas, nonadditivitas dan tidak adanya association bound. |
| R3.4: metrik tambahan | II-D, III-C dan Tabel V: MAE/RMSE/bias per kelas, total MAE dan total accuracy; perbedaan Tree versus total ±1. |
| R3.5: CV atau kebun independen | II-E dan III-E: matched folds, 237/141 pohon, ukuran training per fold, frozen detector, transfer 641/75 ke 24/213 pohon, serta batas validasi lokasi. |
| R3.6: definisi fitur | II-C dan Tabel II: seluruh 67 dimensi, formula, normalisasi, epsilon, confidence threshold, SD termasuk empty views, dan empty-class defaults. |
| R3.7: RF | III-F: cara leaf averaging bekerja, batas extrapolation, rentang prediksi, slope, SD, MAE dan kelompok high-count; interpretasi tetap dibatasi. |
| R3.8: generalisasi | Abstrak, II-E, III-B sampai III-G dan IV: GT adalah referensi empiris, bukan bound; nonsignificance bukan equivalence; keterbatasan seleksi/test/site/pitch tetap eksplisit. |
| R3.10 dan R4: tata letak | Enam tabel dengan seluruh sel terisi, angka dan label tetap, foto/grafik terbaca, sitasi urut, tanpa em dash. |
| R4: pitch kamera | III-E, III-G dan IV: affine/noise feature test, sentinel, batas simulasi, ketiadaan kalibrasi, dan pengujian sudut terukur sebagai future work. |

## Pemeriksaan retensi dan layout

- Urutan seluruh token angka pada masing-masing dari enam tabel sama dengan
  versi delapan halaman sebelum penyuntingan ini.
- Terdapat 23 referensi; isi daftar pustaka tidak berubah dan urutan sitasi sesuai
  kemunculan pertama.
- Diff generator mengonfirmasi sumber foto, koordinat crop, dan bounding box
  Gambar 1 tetap. Raster tertanam dirender ulang pada ukuran layout baru;
  tidak diklaim memiliki hash yang sama. Gambar 2 identik dengan commit acuan.
- Angka bias 0,071, 0,078 dan 0,177 tidak diulang pada paragraf karena sudah
  tercantum dengan tanda yang benar pada Tabel V. Pembahasannya tetap ada.
- Versi utama memuat 4.934 kata hasil ekstraksi PDF, dibandingkan 4.326 pada
  versi ringkas pertama dan 6.496 pada versi delapan halaman. Versi anonim
  memuat 4.867 kata. Jumlah mencakup tabel, label, afiliasi dan referensi,
  sehingga bukan ukuran tunggal kualitas atau retensi penjelasan.
- Main dan blind: 6 halaman A4, 21 font terbenam dan ter-subset, tidak ada em
  dash pada teks PDF, tidak ada overflow horizontal atau rujukan/sitasi yang
  tidak terdefinisi, tidak ada anotasi tautan maupun bookmark.
- Ukuran huruf isi dan tabel tidak diperkecil: template IEEEtran tetap memakai
  isi 10 pt dan tabel 8 pt. Margin dan jarak antarbaris isi tidak diubah.
- Keenam halaman pada kedua PDF telah dirender dan diperiksa secara visual.

## Penyuntingan lanjutan 7 September 2026

Pemeriksaan ulang setelah audit enam halaman menemukan tujuh cacat penyuntingan
dan tiga pelanggaran format. Seluruhnya diperbaiki dan naskah dibangun ulang.

Cacat penyuntingan:

- III-E: kalimat transformasi fitur `cy` tidak memiliki subjek untuk kata kerja
  `leaves`. Dipecah menjadi dua kalimat; jumlah kombinasi (a, b) yang diuji
  ikut disebutkan.
- Pendahuluan: subjek majemuk berkonjungsi `or` diikuti predikat jamak `differ`.
  Konjungsi diubah menjadi `and`.
- II-B: rumusan "Counters use the 590 trees ... and 64 trees" tidak membedakan
  data pelatihan dari data evaluasi counter. Diperjelas menjadi fitted/evaluated.
- III-D: dua parentesis berturut pada rerata confidence diurai; anteseden `its`
  pada aturan diagnostik diganti `that`.
- III-G: `its contribution` diganti `their contribution` karena antesedennya
  dua pasangan kelas.
- Tabel VI: label `YOLO26m` dipakai untuk dua checkpoint berbeda. Baris blok
  Family kini bernama YOLO26m dan baris blok Checkpoint yang memakai split baru
  diberi keterangan; catatan tabel menyebutkan baris lain memakai split lama.
- Nama berkas bobot internal `y26mv2` dihapus dari seluruh naskah. Detektor
  disebut menurut arsitektur dan, bila perlu, menurut split pelatihan.

Pelanggaran format terhadap aturan ICIC 2026 dan IEEE Xplore:

- Kelas dokumen tidak menyertakan opsi `a4paper`, sehingga PDF terkompilasi pada
  US Letter 612x792 pt. ICIC mensyaratkan A4.
- `hyperref` menyisipkan 49 anotasi tautan dan 17 bookmark. Aturan IEEE Xplore
  melarang keduanya. Paket diganti `url`.
- Foto Gambar 1 tertanam pada 212 DPI efektif, di bawah anjuran 300 DPI.
  Generator kini menyimpan PDF pada dpi 200 sehingga hasil akhir 424 DPI.

Tanda afiliasi diubah dari simbol catatan kaki bawaan `\IEEEauthorrefmark`
menjadi superskrip angka, memakai kotak berketinggian nol yang sama sehingga
jarak antarbaris blok penulis tidak berubah.

Ketiga pemeriksaan format kini diasersi oleh `scripts/build_icic_revision.py`,
sehingga build gagal bila salah satunya kembali muncul.

Respons lengkap tetap tersedia di [Response-to-Reviewers.md](Response-to-Reviewers.md).
Pengujian pitch kamera fisik dan kebun independen tetap merupakan keterbatasan,
bukan eksperimen yang diklaim sudah selesai.
