# SawitMVC Baseline: Multi-View Oil Palm Bunch Counting and Maturity Classification

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

Reproducible research baseline for counting and grading fresh fruit bunches (TBS) on oil palm trees from four to eight camera angles per tree. The core problem is **multi-view deduplication**: the same bunch routinely appears in two or more views, causing naive summation to overcount by ~83%.

---

## Dataset

**953 trees** — 3,992 images, 9,823 unique bunches — from two plantations (DAMIMAS & LONSUM) in Kabupaten Tanah Laut, Kalimantan Selatan.

Split: **716 train / 96 val / 141 test** (75/10/15, stratified by variety × dominant class).

### Maturity Classes

| Class | Visual | Position | Role |
|:-----:|--------|----------|------|
| B1 | Red, large, round | Lowest | Highest commercial value |
| B2 | Black with red transition | Above B1 | Imminent harvest target |
| B3 | Solid black, spiky | Above B2 | Next harvest schedule |
| B4 | Smallest, dark green-black | Highest | Future inventory |

---

## Detector: y26mv2

Single YOLOv26-medium detector fine-tuned on the SawitMVC-YOLO training split (60 epochs, batch=32, imgsz=640, patience=60, seed=42).

**mAP50 on val set** (best epoch):

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|:-----:|----------:|----------:|-------:|------:|---------:|
| **all** | 1887 | 0.504 | 0.570 | **0.521** | 0.243 |
| B1 | 201 | 0.606 | 0.801 | 0.746 | 0.379 |
| B2 | 388 | 0.478 | 0.433 | 0.425 | 0.213 |
| B3 | 959 | 0.505 | 0.656 | 0.550 | 0.243 |
| B4 | 339 | 0.427 | 0.389 | 0.363 | 0.137 |

Weights: [`models/yolo/y26mv2.pt`](models/yolo/y26mv2.pt) (42 MB). Training log: [`models/yolo/train_logs/y26m_e60_p60_b32_s42_v2.txt`](models/yolo/train_logs/y26m_e60_p60_b32_s42_v2.txt).

---

## Counting Pipeline

Detections from one tree are folded into a **13-dimensional feature vector**:

```
[ naive_sum_B1..B4,  max_per_side_B1..B4,  mean_per_side_B1..B4,  n_sides ]
```

Three ML counters operate on this vector: SVM (RBF + GridSearchCV), Random Forest (`n_estimators=200, max_depth=10`), and Linear Regression (+ StandardScaler). A deterministic heuristic (M01) serves as a non-learning baseline.

---

## Results

### Track B — End-to-End Counting (y26mv2 + ML counter)

#### Test set (141 trees)

**Accuracy (Acc±1)**:

| Counter | Macro Acc±1 | Total ±1 Acc | Exact Profile | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|:-------:|------------:|-------------:|--------------:|---------:|---------:|---------:|---------:|
| **LR** | **75.71%** | 48.94% | 0.71% | 96.45% | 79.43% | 55.32% | 71.63% |
| SVM | 74.82% | 43.97% | 1.42% | 95.74% | 76.60% | 55.32% | 71.63% |
| RF | 73.23% | 43.26% | 1.42% | 95.74% | 73.76% | 56.03% | 67.38% |
| M01 | 70.57% | 34.75% | 2.84% | 90.78% | 74.47% | 51.06% | 65.96% |

**MAE**:

| Counter | Macro MAE | Total Count MAE | MAE B1 | MAE B2 | MAE B3 | MAE B4 |
|:-------:|----------:|----------------:|-------:|-------:|-------:|-------:|
| **LR** | **1.048** | **1.894** | 0.369 | 1.057 | 1.546 | 1.220 |
| SVM | 1.043 | 2.057 | 0.383 | 1.035 | 1.560 | 1.191 |
| RF | 1.110 | 2.071 | 0.383 | 1.156 | 1.617 | 1.284 |
| M01 | 1.183 | 2.745 | 0.511 | 1.071 | 1.844 | 1.305 |

**Per-class Bias (mean error, + = overcount, − = undercount)**:

| Counter | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|:-------:|--------:|--------:|--------:|--------:|
| LR | +0.014 | −0.064 | −0.142 | +0.057 |
| SVM | +0.014 | −0.582 | −0.284 | −0.312 |
| RF | +0.028 | −0.163 | −0.227 | +0.121 |
| M01 | +0.255 | −0.418 | +0.411 | −0.908 |

#### Val set (96 trees)

**Accuracy (Acc±1)**:

| Counter | Macro Acc±1 | Total ±1 Acc | Exact Profile | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|:-------:|------------:|-------------:|--------------:|---------:|---------:|---------:|---------:|
| **LR** | **69.79%** | 39.58% | 2.08% | 92.71% | 68.75% | 51.04% | 66.67% |
| SVM | 69.53% | 41.67% | 1.04% | 92.71% | 68.75% | 54.17% | 62.50% |
| RF | 67.19% | 45.83% | 1.04% | 87.50% | 63.54% | 55.21% | 62.50% |
| M01 | 66.67% | 36.46% | 1.04% | 85.42% | 68.75% | 45.83% | 66.67% |

**MAE**:

| Counter | Macro MAE | Total Count MAE | MAE B1 | MAE B2 | MAE B3 | MAE B4 |
|:-------:|----------:|----------------:|-------:|-------:|-------:|-------:|
| **LR** | 1.172 | **2.104** | 0.531 | 1.198 | 1.667 | 1.292 |
| SVM | **1.133** | 2.135 | 0.510 | 1.156 | 1.604 | 1.260 |
| RF | 1.190 | 2.135 | 0.542 | 1.313 | 1.583 | 1.323 |
| M01 | 1.375 | 3.146 | 0.688 | 1.260 | 2.292 | 1.260 |

**Per-class Bias (mean error, + = overcount, − = undercount)**:

| Counter | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|:-------:|--------:|--------:|--------:|--------:|
| LR | +0.073 | −0.177 | +0.083 | +0.208 |
| SVM | +0.052 | −0.552 | −0.021 | −0.135 |
| RF | +0.063 | −0.188 | +0.167 | +0.260 |
| M01 | +0.417 | −0.344 | +0.854 | −0.698 |

### Track A — Heuristic Counting on Ground Truth

> Metrics computed directly from GT annotations (no detector noise). Two Acc±1 variants are reported: **Macro** = per-class average over B1–B4; **Joint** = fraction of trees where all 4 classes are simultaneously within ±1.

#### Naive Sum Baseline (no deduplication)

| Set | n | Macro Acc±1 | Joint Acc±1 | Macro MAE | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|-----|--:|------------:|------------:|----------:|---------:|---------:|---------:|---------:|
| Full | 953 | 46.88% | 3.78% | 2.2867 | 65.37% | 51.21% | 9.23% | 61.70% |
| Test | 141 | 50.00% | 6.38% | 2.1418 | 70.92% | 51.77% | 11.35% | 65.96% |

Per-class MAE:

| Set | MAE B1 | MAE B2 | MAE B3 | MAE B4 |
|-----|-------:|-------:|-------:|-------:|
| Full | 1.1312 | 1.7933 | 4.8625 | 1.3599 |
| Test | 0.9362 | 1.7021 | 4.7092 | 1.2199 |

#### M01 Heuristic (best deterministic deduplication)

| Set | n | Macro Acc±1 | Joint Acc±1 | Macro MAE | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|-----|--:|------------:|------------:|----------:|---------:|---------:|---------:|---------:|
| Full | 953 | 95.41% | 87.62% | 0.3746 | 97.59% | 95.59% | 90.87% | 97.59% |
| Test | 141 | **95.92%** | **87.23%** | **0.3404** | 97.87% | 97.16% | 89.36% | 99.29% |

Per-class MAE:

| Set | MAE B1 | MAE B2 | MAE B3 | MAE B4 |
|-----|-------:|-------:|-------:|-------:|
| Full | 0.1658 | 0.3337 | 0.7062 | 0.2928 |
| Test | 0.1489 | 0.2766 | 0.6809 | 0.2553 |

Full heuristic ranking (29 methods): [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).

### Track C — Upper Bound (ML counter on perfect GT features, 95 test trees)

| Counter | Acc±1 | Macro MAE |
|:-------:|------:|----------:|
| LR | **97.37%** | 0.276 |
| SVM | 96.58% | 0.300 |
| RF | 95.79% | 0.361 |

### Cross-Track Summary

| Track | Best result | Acc±1 |
|-------|-------------|------:|
| A: heuristic on GT | M01 | 87.62% |
| C: counter on GT features | LR | **97.37%** |
| **B: y26mv2 + ML counter** | **LR** | **75.71%** |
| Naive sum | — | 3.78% |

**Gap Track B → Track C: 21.66 pp** — this gap is detector error, not counter error. Improving recall on B3 and B4 is the highest-leverage target.

---

## Key Findings

- **LR is the best counter** consistently across test and val splits.
- **B3 is the hardest class** (~51–56% Acc±1) — partially occluded, solid-black appearance blends with B2.
- **B1 is the easiest** (>95% Acc±1) — large, red, visually distinctive.
- All counters tend to **undercount B2 and B3** (negative bias).
- The 21.66-point gap between Track B and Track C confirms the bottleneck is the detector, not the counting algorithm.

---

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: copy weights
cp models/yolo/y26me60p60b32s42v2.pt models/yolo/y26mv2.pt

# Step 2: inference + evaluate all counters
python pipeline/run_e2e_pipeline.py \
    --name y26mv2 \
    --weights models/yolo/y26mv2.pt \
    --data SawitMVC-YOLO/ \
    --counters svm lr rf m01

# Step 3: print metrics
python scripts/report_metrics.py y26mv2 test
python scripts/report_metrics.py y26mv2 val

# Track A: heuristics on full GT
python benchmarks/run_benchmark.py

# Track C: upper bound
bash scripts/reproduce_upper_bound.sh

# Verify all headline claims
python benchmarks/check_release_claims.py
```

Dataset download (required for inference from scratch):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset', local_dir='./SawitMVC-YOLO', token=True)
"
```

---

## Repository Layout

```
algorithms/          # Track A: 5 deterministic heuristics (M01..M05)
pipeline/            # Track B/C: feature extraction + ML counter scripts
benchmarks/          # run_benchmark.py, check_release_claims.py
ground_truth/        # 953 GT JSONs + split_manifest.csv
predictions/         # Cached YOLO outputs (y26mv2_per_tree/)
results/             # Pre-computed evaluation (e2e_per_tree, heuristics_953, e2e_upper_bound)
models/yolo/         # y26mv2.pt + training log
models/counters/     # svm.pkl, rf.pkl, lr.pkl
docs/                # algorithms.md, evaluation.md, findings.md, ...
scripts/             # reproduce_all.sh, reproduce_upper_bound.sh, ...
```

---

## Citation

```bibtex
@dataset{sawitmvc2026,
  title   = {SawitMVC-YOLO: Multi-View Oil Palm Bunch Counting Dataset},
  author  = {ULM-DS-Lab},
  year    = {2026},
  url     = {https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO},
  license = {CC BY-NC 4.0}
}
```

---

## License

[CC BY-NC 4.0](LICENSE) — non-commercial use with attribution.
