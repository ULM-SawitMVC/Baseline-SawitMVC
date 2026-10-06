# ICIC 2026 — Paper #057 Artifacts & Presentation Workspace

Direktori ini berisi seluruh berkas terkait publikasi paper dan materi presentasi untuk **The Tenth International Conference on Informatics and Computing (ICIC 2026)**.

**Judul Paper:**  
*Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector*

**Penulis:**  
Muhammad Zainal Muttaqin, Fatma Indriani, Setyo Wahyu Saputro, Alia Rahmi, Triando Hamonangan Saragih, Hartoni, Dwi Kartini, Naufal Said  
*(Universitas Lambung Mangkurat, Banjarbaru, Indonesia)*

---

## Struktur Direktori

```text
draft-icic-2026/
├── presentation/         # Materi presentasi lisan (slides, script, video)
├── camera-ready/         # Berkas final naskah kamera siap & sertifikasi IEEE
├── latex/                # Source code LaTeX kamera siap & script pipeline DOCX
├── archive/              # Arsip draft awal dan versi blind review
├── Accepted-Revise/      # Dokumen respon reviewer, audit revisi & analisis GPU
└── README.md             # Dokumentasi indeks berkas ini
```

---

## Rincian Berkas per Direktori

### 1. `presentation/` — Materi Presentasi
| Berkas | Deskripsi |
|---|---|
| `057_MUTTAQIN.pptx` | Slide deck presentasi (7 slide, 4 bagian: introduction, background, results, conclusion); naskah tersimpan juga pada catatan pembicara. |
| `Script.md` | Naskah presentasi (635 kata, sekitar 4,5 sampai 5 menit) dan Q&A cheat sheet. |
| `ICIC_057_MUTTAQIN.mp4` | Rekaman video presentasi resmi (66 MB), direkam dengan slide deck versi pertama. |

### 2. `camera-ready/` — Berkas Final Submission
| Berkas | Deskripsi |
|---|---|
| `057_MUTTAQIN.pdf` | Paper PDF final kamera siap (6 halaman IEEE format). |
| `2026318348.pdf` | Paper PDF yang telah lolos validasi IEEE PDF eXpress (PID: 2026318348). |
| `57_MUTTAQIN.docx` | Naskah paper final format Microsoft Word sesuai template IEEE ICIC. |
| `Response_057_MUTTAQIN.pdf` | Dokumen respon formal terhadap tanggapan semua reviewer. |
| `Similarity_057_MUTTAQIN.pdf` | Laporan resmi uji similaritas Turnitin. |

### 3. `latex/` — Source Code & Build Pipeline
| Berkas / Direktori | Deskripsi |
|---|---|
| `main-new.tex` | Source LaTeX utama untuk versi kamera siap (path gambar relatif ke `../../figures/paper/`). |
| `main-new.pdf` | Hasil kompilasi PDF lokal dari `main-new.tex`. |
| `main-new.docx` | Dokumen Word hasil konversi LaTeX. |
| `IEEEtran.cls` | Class file LaTeX standar IEEE. |
| `template/` | Template asli IEEE ICIC 2026 (Word A4 & LaTeX bundle). |
| `tex_to_docx.py` | Script Python konversi LaTeX ke format Word 2-kolom IEEE. |
| `place_floats.ps1` | Script PowerShell untuk penataan otomatis floating table & figure pada Word. |

### 4. `archive/` — Arsip Draft Terdahulu
| Berkas | Deskripsi |
|---|---|
| `main.tex` / `main.pdf` | Draft awal naskah sebelum proses peer-review. |
| `main-blind.tex` / `main-blind.pdf` | Naskah yang disubmit untuk tahap blind review. |
| `main-blind-old.pdf` | Versi kompilasi lama naskah blind review. |
| `presentation-v1/` | Materi presentasi versi pertama: slide deck 13 slide, `Script.md`, naskah Word dan PDF, dan `make_docx.py`. |

### 5. `Accepted-Revise/` — Catatan Revisi & Audit Reviewer
Berisi file Markdown dan notebook yang mendokumentasikan tindak lanjut atas masukan reviewer (termasuk validasi GPU Reviewer 4, closure matrix, dan audit 6 halaman).
