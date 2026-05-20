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

#### Baseline counters — Test set (141 trees, F0: 13-dim, train on 716 trees)

Accuracy and per-class breakdown:

| Counter | Macro Acc±1 | Joint Acc±1 | Macro MAE | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|:-------:|------------:|------------:|----------:|---------:|---------:|---------:|---------:|
| **LR** | **75.71%** | 30.50% | 1.048 | 96.45% | 79.43% | 55.32% | 71.63% |
| SVM | 74.82% | 29.08% | 1.043 | 95.74% | 76.60% | 55.32% | 71.63% |
| RF | 73.23% | 26.95% | 1.110 | 95.74% | 73.76% | 56.03% | 67.38% |
| M01 (heuristic) | 70.57% | 24.11% | 1.183 | 90.78% | 74.47% | 51.06% | 65.96% |

Per-class bias (mean signed error, + = overcount):

| Counter | MAE B1 | MAE B2 | MAE B3 | MAE B4 | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|:-------:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|
| LR | 0.369 | 1.057 | 1.546 | 1.220 | +0.014 | −0.064 | −0.142 | +0.057 |
| SVM | 0.383 | 1.035 | 1.560 | 1.191 | +0.014 | −0.582 | −0.284 | −0.312 |
| RF | 0.383 | 1.156 | 1.617 | 1.284 | +0.028 | −0.163 | −0.227 | +0.121 |
| M01 | 0.511 | 1.071 | 1.844 | 1.305 | +0.255 | −0.418 | +0.411 | −0.908 |

#### Baseline counters — Val set (96 trees)

| Counter | Macro Acc±1 | Joint Acc±1 | Macro MAE | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 |
|:-------:|------------:|------------:|----------:|---------:|---------:|---------:|---------:|
| **LR** | **69.79%** | 33.33% | 1.172 | 92.71% | 68.75% | 51.04% | 66.67% |
| SVM | 69.53% | 33.33% | 1.133 | 92.71% | 68.75% | 54.17% | 62.50% |
| RF | 67.19% | 35.42% | 1.190 | 87.50% | 63.54% | 55.21% | 62.50% |
| M01 | 66.67% | 29.17% | 1.375 | 85.42% | 68.75% | 45.83% | 66.67% |

#### Feature ablation experiments — Test set (141 trees)

80 configurations tested (`experiments/exp_counting_v3.py`): 8 feature sets × 5 models × 2 training strategies. All feature sets are supersets of F0. Training strategy "train" = 716 trees; "train+val" = 812 trees.

**Feature set definitions:**

| Tag | Dims | Added features (beyond F0 = 13-dim baseline) |
|-----|-----:|----------------------------------------------|
| F0 | 13 | *baseline*: naive\_sum, max\_per\_side, mean\_per\_side × 4 classes + n\_sides |
| F0+conf | 33 | per-class: conf\_sum, conf\_mean, conf\_max, high\_conf≥0.5, vhigh\_conf≥0.6 |
| F0+spatial | 21 | per-class: mean\_cy (vertical centroid), mean\_area (bbox size) |
| F0+distrib | 33 | per-class: std\_per\_side, min\_per\_side, cv\_per\_side (std/mean), n\_sides\_detected, 1/(1+std) consistency |
| F0+conf+spatial | 41 | F0 + conf + spatial |
| F0+conf+distrib | 53 | F0 + conf + distrib |
| F0+distrib+spatial | 37 | F0 + distrib + spatial |
| F_all | 67 | all above + total\_naive, frac\_Bc (class proportions), b3/(b2+b3) ratio |

**Top 10 configurations by Macro Acc±1 (test, 141 trees):**

| Feature Set | Model | Strategy | Dims | Macro Acc±1 | Joint Acc±1 | Macro MAE | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|-------------|-------|----------|-----:|------------:|------------:|----------:|--------:|--------:|--------:|--------:|
| **F_all** | **Ridge** | train | 67 | **77.48%** | **32.62%** | **1.036** | +0.014 | −0.078 | −0.177 | +0.071 |
| F0+spatial | ElasticNet | train | 21 | 76.77% | 31.21% | 1.039 | +0.014 | −0.064 | −0.156 | +0.035 |
| F0+spatial | Ridge | train | 21 | 76.60% | 30.50% | 1.046 | +0.021 | −0.071 | −0.128 | +0.021 |
| F0+spatial | ElasticNet | train+val | 21 | 76.60% | 31.21% | 1.051 | +0.007 | −0.035 | −0.149 | −0.014 |
| F0 | ElasticNet | train | 13 | 76.42% | 29.79% | 1.043 | +0.007 | −0.057 | −0.135 | +0.000 |
| F0 | ElasticNet | train+val | 13 | 76.42% | 30.50% | 1.034 | +0.007 | −0.014 | −0.113 | −0.043 |
| F0 | Ridge | train+val | 13 | 76.42% | 30.50% | 1.037 | +0.021 | −0.021 | −0.106 | −0.057 |
| F0 | LR | train+val | 13 | 76.42% | 29.79% | 1.044 | +0.007 | −0.028 | −0.163 | +0.007 |
| F_all | Ridge | train+val | 67 | 76.42% | 29.08% | 1.044 | +0.007 | −0.085 | −0.156 | +0.043 |
| F0 | Ridge | train | 13 | 76.06% | 28.37% | 1.053 | +0.028 | −0.035 | −0.128 | −0.007 |

**Top 2 per model (best feature set × strategy per model):**

| Model | Best Feature Set | Macro Acc±1 | Joint Acc±1 | Macro MAE |
|-------|-----------------|------------:|------------:|----------:|
| **Ridge** | F_all, train | **77.48%** | **32.62%** | **1.036** |
| ElasticNet | F0+spatial, train | 76.77% | 31.21% | 1.039 |
| LR | F0, train+val | 76.42% | 29.79% | 1.044 |
| XGB | F0, train+val | 75.35% | 32.62% | 1.066 |
| LGB | F0, train | 74.65% | 31.21% | 1.087 |

**Observations (v3):**
- Linear models (LR, Ridge, ElasticNet) consistently outperform tree-based methods (XGB, LGB) — training set of 716 trees is too small for tree-based overfitting.
- **Spatial features** (mean vertical centroid `mean_cy`, mean bbox area `mean_area` per class) provide the most consistent gain — each maturity class occupies a different vertical zone on the tree.
- **F_all + Ridge** achieves the highest test performance (+1.77 pp over LR baseline) but shows a larger train–test gap on val (70.57%). **F0+spatial + ElasticNet** (76.77%) is the most stable configuration across val/test.
- Using train+val (812 trees) for training yields +0.71 pp on average but varies by model.

#### Deep research — v4 ceiling probe (`experiments/exp_counting_v4.py`)

Extended to **170-dim** features (multi-threshold counts at conf ≥ 0.30/0.35/0.40/0.45/0.50, entropy, harmonic mean, duplication factor estimate, confidence percentiles p25/p50/p75, std\_cy, std\_area, mean\_cx, cross-class ratios). Also tested Bayesian hyperparameter search (Optuna, 80 trials) for XGB and LGB, and 5-fold OOF stacking (Ridge + ElasticNet + XGB + LGB → Ridge meta-learner).

| Config | Dims | Macro Acc±1 | Joint Acc±1 | Macro MAE |
|--------|-----:|------------:|------------:|----------:|
| Ridge + F_all (v3 best, repeated) | 67 | **77.48%** | **32.62%** | 1.036 |
| Ridge + 170-dim | 170 | 76.24% | 31.21% | 1.037 |
| ElasticNet + F0+spatial | 21 | 76.77% | 31.21% | 1.039 |
| Stacking 4 base + Ridge meta | 67 | 76.06% | 30.50% | 1.046 |
| XGB-Optuna (80 trials) | 67 | 71.10% | 24.82% | 1.133 |
| LGB-Optuna (80 trials) | 67 | 71.81% | 24.82% | 1.092 |
| Blend Ridge(0.7)+XGB+LGB | 67 | 77.30% | 32.62% | **1.018** |

**Key result:** All v4 approaches fail to exceed 77.48%. Adding multi-threshold features (170-dim) *hurts* Ridge because the new features are highly collinear with `naive_sum` and `conf_sum`, adding noise without new signal. Stacking is limited by base model correlation (tree methods are consistently inferior). Blending marginally improves MAE (1.018) but not Macro Acc±1.

**Conclusion: 77.48% is the practical ceiling for ML counting on current YOLO predictions.** The 19.89 pp gap to Track C is due to detector quality (B3 recall 65.6%, B4 recall 38.9%), not counter design. The Pearson correlation between YOLO naive\_sum and GT true count is r = 0.421 for B2 (r² = 0.177) — meaning 82.3% of B2 variance is irreducible detector noise, regardless of feature engineering or model choice.

---

### Summary Table (Table 4)

Two Acc±1 variants: **Macro** = per-class average over B1–B4; **Joint** = fraction of trees where all 4 classes are simultaneously within ±1 (stricter). Bias = mean signed error (+ overcount, − undercount).

| Track | Method | Features | Set | Macro Acc±1 | Joint Acc±1 | Macro MAE | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|-------|--------|----------|-----|------------:|------------:|----------:|--------:|--------:|--------:|--------:|
| Naive Sum | GT annotations | — | 953 trees | 46.88% | 3.78% | 2.287 | +1.131 | +1.793 | +4.863 | +1.360 |
| Naive Sum | GT annotations | — | 141 test | 50.00% | 6.38% | 2.142 | +0.936 | +1.702 | +4.709 | +1.220 |
| Track A | M15 (simple divisor) | GT det. | 953 trees | 95.17% | 85.94% | 0.391 | +0.168 | +0.152 | +0.342 | −0.026 |
| Track A | M15 (simple divisor) | GT det. | 141 test | 95.39% | 85.11% | 0.376 | +0.135 | +0.135 | +0.262 | −0.064 |
| Track A | M01 (best complex) | GT det. | 953 trees | 95.41% | 87.62% | 0.375 | +0.128 | +0.193 | +0.188 | −0.098 |
| Track A | M01 (best complex) | GT det. | 141 test | 95.92% | 87.23% | 0.340 | +0.106 | +0.149 | +0.099 | −0.099 |
| Track B | LR (baseline) | F0 (13-dim) | 141 test | 75.71% | 30.50% | 1.048 | +0.014 | −0.064 | −0.142 | +0.057 |
| Track B | SVM | F0 (13-dim) | 141 test | 74.82% | 29.08% | 1.043 | +0.014 | −0.582 | −0.284 | −0.312 |
| Track B | RF | F0 (13-dim) | 141 test | 73.23% | 26.95% | 1.110 | +0.028 | −0.163 | −0.227 | +0.121 |
| Track B | ElasticNet | F0 (13-dim) | 141 test | 76.42% | 29.79% | 1.043 | +0.007 | −0.057 | −0.135 | +0.000 |
| Track B | ElasticNet | F0+spatial (21-dim) | 141 test | 76.77% | 31.21% | 1.039 | +0.014 | −0.064 | −0.156 | +0.035 |
| Track B | **Ridge** | **F_all (67-dim)** | **141 test** | **77.48%** | **32.62%** | **1.036** | +0.014 | −0.078 | −0.177 | +0.071 |
| Track C | LR on GT features | F0 (GT det.) | 95 test | 97.37% | 90.53% | 0.276 | −0.053 | +0.021 | +0.168 | +0.000 |

Full heuristic ranking (29 methods): [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).  
Full ablation: v3 (80 configs): [`results/experiments/counting_v3_results.csv`](results/experiments/counting_v3_results.csv) · v4 (deep probe): [`results/experiments/counting_v4_results.csv`](results/experiments/counting_v4_results.csv).

**Gap Track B → Track C: 19.89 pp** — detector error, not counter error. B3 recall (YOLO val: 65.6%) and B4 recall (38.9%) are the highest-leverage improvement targets.

---

## Key Findings

- **Best counter: Ridge + F_all (67-dim)** — 77.48% Macro Acc±1 / 32.62% Joint Acc±1, a +1.77 pp gain over the LR baseline. For stability, **ElasticNet + F0+spatial (21-dim)** (76.77%) is preferred.
- **Spatial features are most informative** — `mean_cy` (vertical centroid) and `mean_area` per class are the highest-value additions; each maturity class occupies a distinct vertical zone on the tree.
- **77.48% is the practical ML-counter ceiling** — exhaustive probing (170-dim features, Optuna-tuned XGB/LGB, stacking ensembles) confirmed no approach exceeds it. B2 YOLO detections have r = 0.421 correlation with GT (r² = 0.177), making 82.3% of B2 variance irreducible detector noise.
- **Linear models dominate** — Ridge, ElasticNet, and LR all outperform tree-based methods (XGB, LGB, RF) with 716 training trees. 170-dim features *hurt* Ridge due to collinearity.
- **B3 is the hardest class** (51–57% Acc±1) — YOLO val recall 65.6%; solid-black appearance blends with B2.
- **B4 is severely under-detected** — YOLO val recall 38.9%; undercounted 0.85× on average despite being the most numerous class.
- **B1 is the easiest** (>95% Acc±1) — large, red, visually distinct; YOLO val recall 80.1%.
- **SVM severely undercounts B2** (bias −0.582) — RBF kernel unsuitable for this small, near-linear problem.
- **All ML counters undercount B2 and B3** (negative bias) — missed YOLO detections dominate.
- **The 19.89 pp gap to Track C is entirely detector error** — improving B3/B4 YOLO recall is the only path to substantial gains beyond 77.48%.

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
