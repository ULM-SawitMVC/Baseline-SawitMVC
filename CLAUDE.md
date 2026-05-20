# SawitMVC Baseline — CLAUDE.md

---

## RERUN PLAN: y26mv2 Full Evaluation (75/10/15 split)
> Status: **PENDING** — hapus bagian ini setelah selesai run di cloud

### Tujuan
Jalankan inference dengan model `y26me60p60b32s42v2.pt` pada split baru (716/96/141),
lalu evaluasi semua counter (SVM, LR, RF, M01) dan collect metrics berikut:
per-class MAE, macro class-MAE, exact profile accuracy, total count MAE,
total ±1 accuracy, per-class bias.

### Step 0 — Setup cloud
```bash
git clone https://github.com/ULM-SawitMVC/Baseline-SawitMVC.git
cd Baseline-SawitMVC
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 1 — Download dataset dari Hugging Face
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ULM-DS-Lab/SawitMVC-YOLO',
    repo_type='dataset',
    local_dir='./SawitMVC-YOLO',
    token=True
)
"
```

### Step 2 — Siapkan weights
```bash
cp models/yolo/y26me60p60b32s42v2.pt models/yolo/y26mv2.pt
```

### Step 3 — Inference + evaluasi semua counter
```bash
python pipeline/run_e2e_pipeline.py \
    --name y26mv2 \
    --weights models/yolo/y26mv2.pt \
    --data SawitMVC-YOLO/ \
    --counters svm lr rf m01
```
Output: `predictions/y26mv2_per_tree/` (953 JSONs) +
`results/e2e_per_tree/y26mv2_{svm,lr,rf,m01}/metrics.json`

Estimasi waktu: ~3 menit inference (GPU A100) + <5 menit evaluasi.

### Step 4 — Print metrics
```bash
python scripts/report_metrics.py y26mv2 test
python scripts/report_metrics.py y26mv2 val
```

### Step 5 — Ambil hasil & push
```bash
tar -czf y26mv2_results.tar.gz \
    predictions/y26mv2_per_tree/ \
    results/e2e_per_tree/y26mv2_svm/ \
    results/e2e_per_tree/y26mv2_lr/ \
    results/e2e_per_tree/y26mv2_rf/ \
    results/e2e_per_tree/y26mv2_m01/
# download y26mv2_results.tar.gz ke lokal, extract, lalu:
git add predictions/y26mv2_per_tree/ results/e2e_per_tree/y26mv2_*/
git commit -m "add y26mv2 inference predictions and evaluation results"
git push
```

### Checklist
- [ ] Clone + install di cloud
- [ ] Dataset HF berhasil didownload
- [ ] `cp` weights ke `y26mv2.pt`
- [ ] `run_e2e_pipeline.py` selesai tanpa error
- [ ] `report_metrics.py` menampilkan angka untuk test + val
- [ ] Hasil di-download dan di-push ke repo

---

## Project overview

Research baseline untuk menghitung dan mengklasifikasi kematangan tandan buah segar (TBS) kelapa sawit dari banyak sudut kamera (multi-view). Masalah utama: satu tandan muncul di beberapa foto dari sudut berbeda, sehingga penjumlahan naif overcounts ~83%.

**Dataset**: 953 pohon, 3,992 gambar, 9,823 tandan unik — dari dua kebun (DAMIMAS & LONSUM) di Kabupaten Tanah Laut, Kalimantan Selatan.

---

## Tech stack

- **Python** ≥ 3.10
- **YOLOv26** (Ultralytics) — detektor objek, 3 ukuran: nano/small/medium
- **scikit-learn** — SVM (RBF kernel + GridSearchCV), Random Forest, Linear Regression
- **numpy** — feature extraction
- GPU hanya dibutuhkan saat retraining YOLO; semua evaluasi bisa di CPU

---

## Domain concepts

### Kelas kematangan TBS
| Kelas | Visual | Posisi | Nilai |
|-------|--------|--------|-------|
| B1 | Merah, besar, bulat | Paling bawah | Nilai komersial tertinggi |
| B2 | Hitam + sisa merah | Di atas B1 | Target panen berikutnya |
| B3 | Hitam solid, berduri | Di atas B2 | Jadwal panen selanjutnya |
| B4 | Terkecil, hijau-hitam | Paling atas | Inventaris masa depan |

### Tiga track eksperimen
- **Track A** — 5 heuristik deterministik langsung pada ground truth → best: **87.62% Acc±1** (M01)
- **Track B** — Counter ML (SVM/RF/LR) pada fitur 13-dim dari deteksi YOLO → best: **70.79% Acc±1** (y26s + SVM)
- **Track B'** — Per-image variant (max/mean/sum aggregation) → best: **64.21%** (y26m + max)
- **Track C** — Upper bound: counter ML pada fitur dari ground truth → best: **97.37% Acc±1** (LR)

Gap Track B vs Track C = **26.58 poin** → sumber utama error adalah detektor, bukan counter.

### Metric utama
**Acc±1**: fraksi pohon di mana prediksi ≤ 1 dari ground truth, di-macro-average atas 4 kelas.
Supporting: Macro MAE, total-count MAE, exact-profile accuracy.

---

## Repository layout (ringkas)

```
algorithms/          # 5 heuristik deterministik (M01..M05)
pipeline/            # Script counter ML + end-to-end pipeline
benchmarks/          # Entry points: run_benchmark.py, check_release_claims.py
ground_truth/        # 953 JSON anotasi per pohon + split_manifest.csv
predictions/         # Cached YOLO outputs (y26n/s/m × per_tree/per_image)
results/             # Output evaluasi ter-precompute (36 folder)
models/yolo/         # Bobot YOLO: y26n.pt, y26s.pt, y26m.pt
models/counters/     # Artefak counter: svm.pkl, rf.pkl, lr.pkl
SawitMVC-YOLO/       # Gambar dataset (YOLO format, images + labels)
docs/                # Dokumentasi panjang (algorithms.md, evaluation.md, dll)
scripts/             # Shell scripts: reproduce_all.sh, dll
```

---

## Key files

| File | Peran |
|------|-------|
| `algorithms/M01_selector_b2b3.py` | Algoritma heuristik terbaik (87.62%) |
| `algorithms/__init__.py` | Registry semua algoritma |
| `pipeline/build_counting_features.py` | Ekstraksi fitur 13-dim per pohon |
| `pipeline/run_e2e_pipeline.py` | End-to-end Track B (per-tree) |
| `pipeline/run_e2e_per_image.py` | End-to-end Track B' (per-image) |
| `pipeline/run_e2e_inference.py` | Inference YOLO per pohon |
| `pipeline/run_counting_{svm,rf,lr}.py` | Trainer counter ML |
| `benchmarks/run_benchmark.py` | Runner benchmark Track A |
| `benchmarks/check_release_claims.py` | Verifikasi angka klaim di README |
| `ground_truth/split_manifest.csv` | Split 763/95/95 (train/val/test) |

---

## Common commands

```bash
# Track A: 5 heuristik pada 953 pohon
python benchmarks/run_benchmark.py

# Track B: end-to-end satu detektor
python pipeline/run_e2e_pipeline.py --name y26s --skip-inference

# Track B': per-image variant
python pipeline/run_e2e_per_image.py --name y26s --skip-inference

# Track C: upper bound
bash scripts/reproduce_upper_bound.sh

# Reproduksi semua (dari cached predictions, tanpa GPU)
bash scripts/reproduce_all.sh

# Verifikasi klaim di README
python benchmarks/check_release_claims.py

# Download dataset dari Hugging Face (hanya untuk retraining)
python -c "from huggingface_hub import snapshot_download; snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset', local_dir='./SawitMVC-YOLO', token=True)"
```

---

## Algorithm interface

Semua algoritma heuristik (M01..M05) mengekspos fungsi yang sama:

```python
def predict(detections: list[dict]) -> dict[str, int]:
    # detections: list dari bounding box, tiap dict harus punya:
    #   "class": "B1"/"B2"/"B3"/"B4"
    #   "x_norm": posisi x dinormalisasi (0..1)
    #   "side_index": indeks sudut kamera (0..7)
    # returns: {"B1": int, "B2": int, "B3": int, "B4": int}
```

Untuk menambah algoritma baru:
1. Implementasi `predict()` di `algorithms/M{NN}_{family}_{descriptor}.py`
2. Daftarkan di `algorithms/__init__.py`
3. Jalankan `python benchmarks/run_benchmark.py --save` dan commit CSV yang diperbarui
4. Algoritma baru harus melampaui M05 (86.99% Acc±1) untuk diterima

---

## Feature vector (13 dimensi)

```python
# dari pipeline/build_counting_features.py
[naive_sum_B1, naive_sum_B2, naive_sum_B3, naive_sum_B4,   # 4: total deteksi
 max_per_side_B1, ..., max_per_side_B4,                     # 4: max di satu sudut
 mean_per_side_B1, ..., mean_per_side_B4,                   # 4: rata-rata per sudut
 n_sides]                                                    # 1: jumlah sudut (4 atau 8)
```

---

## Data conventions

- **tree_id / tree_name**: identifier pohon, key di semua JSON
- **split**: "train" / "val" / "test" — selalu baca dari `ground_truth/split_manifest.csv` kolom `new_split`
- **Anotasi GT**: `ground_truth/annotations/{tree_id}.json` — struktur: `images`, `bunches`, `_confirmedLinks`, `summary`
- **Prediksi inference**: `predictions/y26{n,s,m}_per_tree/{tree_id}.json`
- **Hasil evaluasi**: `results/{folder}/metrics.json` + `predictions.csv`
- Gambar tidak dibundle di repo — hanya diperlukan saat retraining

---

## Conventions

- `seed=42` di semua script
- YOLO training: 60 epoch, `batch=32`, `imgsz=640`, `patience=60`, `seed=42`, deterministic
- SVM: `GridSearchCV` atas C dan gamma, `n_jobs=-1` (CV scores bisa drift <0.5%, test split stabil)
- RF: `n_estimators=200`, `max_depth=10`, `random_state=42`
- Counter artifacts bisa diload ulang: `--load-model models/counters/{svm,rf,lr}.pkl`
- Semua script mendukung `--skip-inference` untuk pakai cached predictions
