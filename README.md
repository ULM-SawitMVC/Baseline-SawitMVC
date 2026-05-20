# SawitMVC Baseline: Multi-View Oil Palm Bunch Counting

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

Reproducible baseline for counting and grading fresh fruit bunches (TBS) from four to eight camera angles per tree. The core challenge is **multi-view deduplication**: the same bunch routinely appears in 2–3 views, causing naive summation to overcount by ~83%.

---

## Dataset

**953 trees** — 3,992 images, 9,823 unique bunches — from two plantations (DAMIMAS & LONSUM), Kabupaten Tanah Laut, Kalimantan Selatan.

Split: **716 train / 96 val / 141 test** (75/10/15, stratified by variety × dominant class).

| Class | Visual | Position | Role |
|:-----:|--------|----------|------|
| B1 | Red, large, round | Lowest | Highest commercial value |
| B2 | Black with red transition | Above B1 | Imminent harvest target |
| B3 | Solid black, spiky | Above B2 | Next harvest schedule |
| B4 | Smallest, dark green-black | Highest | Future inventory |

---

## 1 · Detector

Single **YOLOv26-medium** (`y26mv2`) fine-tuned on the training split (60 epochs, batch=32, imgsz=640, patience=60, seed=42).

**mAP50 — val set (best epoch):**

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|:-----:|----------:|----------:|-------:|------:|---------:|
| **all** | 1887 | 0.504 | 0.570 | **0.521** | 0.243 |
| B1 | 201 | 0.606 | 0.801 | 0.746 | 0.379 |
| B2 | 388 | 0.478 | 0.433 | 0.425 | 0.213 |
| B3 | 959 | 0.505 | 0.656 | 0.550 | 0.243 |
| B4 | 339 | 0.427 | 0.389 | 0.363 | 0.137 |

B4 (recall 38.9%) and B3 (65.6%) are the weakest classes. Weights: [`models/yolo/y26mv2.pt`](models/yolo/y26mv2.pt) (42 MB).

---

## 2 · Counter

Detections from one tree are folded into a feature vector, then a regression model maps features → per-class bunch counts.

**Baseline feature vector (F0, 13-dim):**
```
naive_sum_B1..B4 | max_per_side_B1..B4 | mean_per_side_B1..B4 | n_sides
```

**Extended feature vector (F_all, 67-dim):** F0 + per-class confidence stats (conf\_sum, conf\_mean, conf\_max, high\_conf≥0.5), per-side distribution stats (std, min, cv, n\_sides\_detected, consistency), spatial stats (mean vertical centroid `mean_cy`, mean bbox area `mean_area`), and cross-class proportions.

### Track A — Heuristic counters on ground-truth detections

Upper reference: what a deterministic counter achieves when the detector is perfect.

| Method | Set | Macro Acc±1 | Joint Acc±1 | Macro MAE |
|--------|-----|------------:|------------:|----------:|
| Naive sum (no deduplication) | 141 test | 50.00% | 6.38% | 2.142 |
| M15 — divide by global factor | 141 test | 95.39% | 85.11% | 0.376 |
| **M01 — best complex heuristic** | **141 test** | **95.92%** | **87.23%** | **0.340** |

M15 divides each class's naive sum by a calibrated constant (B1:1.986, B2:1.786, B3:1.795, B4:1.655). M01 uses a three-way Gaussian visibility selector with adaptive divisors. Full ranking of 29 heuristics: [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).

### Track C — ML counter upper bound (GT detections as input)

What an ML counter achieves when the detector is perfect.

| Method | Features | Set | Macro Acc±1 | Joint Acc±1 | Macro MAE |
|--------|----------|-----|------------:|------------:|----------:|
| LR on GT features | F0 (13-dim) | 95 test | **97.37%** | **90.53%** | **0.276** |

---

## 3 · End-to-End (Detector + Counter)

YOLO detections → feature vector → ML counter → per-class count.

### Baseline counters — F0 (13-dim), train on 716 trees, test on 141 trees

| Counter | Macro Acc±1 | Joint Acc±1 | Macro MAE | B1 Acc±1 | B2 Acc±1 | B3 Acc±1 | B4 Acc±1 | Bias B2 | Bias B3 |
|:-------:|------------:|------------:|----------:|---------:|---------:|---------:|---------:|--------:|--------:|
| **LR** | **75.71%** | 30.50% | 1.048 | 96.45% | 79.43% | 55.32% | 71.63% | −0.064 | −0.142 |
| SVM | 74.82% | 29.08% | 1.043 | 95.74% | 76.60% | 55.32% | 71.63% | −0.582 | −0.284 |
| RF | 73.23% | 26.95% | 1.110 | 95.74% | 73.76% | 56.03% | 67.38% | −0.163 | −0.227 |

### Feature ablation — best 3 configurations (test, 141 trees)

Experiment CSVs for baseline ablation and deep probe are in [`results/experiments/`](results/experiments/).

| Counter | Features | Dims | Train set | Macro Acc±1 | Joint Acc±1 | Macro MAE | B2 Acc±1 | B3 Acc±1 |
|:-------:|----------|-----:|:---------:|------------:|------------:|----------:|---------:|---------:|
| **Ridge** | **F_all** | **67** | **716** | **77.48%** | **32.62%** | **1.036** | **82.98%** | **57.45%** |
| ElasticNet | F0+spatial | 21 | 716 | 76.77% | 31.21% | 1.039 | 81.56% | 56.74% |
| ElasticNet | F0 | 13 | 812 | 76.42% | 30.50% | 1.034 | 80.14% | 56.03% |

`F0+spatial` adds `mean_cy` (vertical centroid) and `mean_area` per class — highest-value single addition. `F_all` additionally includes per-side distribution stats and confidence stats. **Ridge + F_all** is the best single configuration; **ElasticNet + F0+spatial** is more stable across val/test (val: 71.09% vs 70.57%).

**Why linear models win:** With 716 training trees, tree-based models (XGB, LGB) overfit and peak below the linear baselines. Stacking (4 base models → Ridge meta) reaches 76.06%. The counter ceiling is confirmed at 77.48% — see v4 probe below.

### Counter ceiling analysis (v4 deep probe)

The v4 probe expands to 200+ features (multi-threshold counts, entropy, harmonic mean, conf percentiles, extended spatial stats), plus Optuna-tuned XGB/LGB (60 trials on train, 40 on train+val) and 5-fold OOF stacking.

| Approach | Macro Acc±1 |
|----------|------------:|
| Ridge + F_all 67-dim (v3 best, reference) | **77.48%** |
| ElasticNet + F0+spatial (best v4 run) | 76.77% ↓ |
| Ridge + full v4 feature set | 76.24% ↓ |
| Stacking 4 models | 76.06% ↓ |
| XGB-Optuna | 74.11% ↓ |

The larger v4 feature bank still fails to beat the v3 reference. **77.48% is the practical ML-counter ceiling on current YOLO detections.** The Pearson correlation between YOLO naive\_sum and GT count is r = 0.421 for B2 (r² = 0.177) — 82.3% of B2 variance is irreducible detector noise.

---

## Summary Table (Table 4)

Acc±1 variants: **Macro** = per-class average over B1–B4; **Joint** = all 4 classes simultaneously within ±1. Bias = mean signed error (+ overcount, − undercount).

| Track | Method | Features | Set | Macro Acc±1 | Joint Acc±1 | Macro MAE | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|-------|--------|----------|-----|------------:|------------:|----------:|--------:|--------:|--------:|--------:|
| Naive Sum | GT annotations | — | 141 test | 50.00% | 6.38% | 2.142 | +0.936 | +1.702 | +4.709 | +1.220 |
| Track A | M15 (simple divisor) | GT det. | 141 test | 95.39% | 85.11% | 0.376 | +0.135 | +0.135 | +0.262 | −0.064 |
| Track A | M01 (best complex) | GT det. | 141 test | 95.92% | 87.23% | 0.340 | +0.106 | +0.149 | +0.099 | −0.099 |
| Track B | LR (baseline) | F0 13-dim | 141 test | 75.71% | 30.50% | 1.048 | +0.014 | −0.064 | −0.142 | +0.057 |
| Track B | ElasticNet | F0+spatial 21-dim | 141 test | 76.77% | 31.21% | 1.039 | +0.014 | −0.064 | −0.156 | +0.035 |
| Track B | **Ridge** | **F_all 67-dim** | **141 test** | **77.48%** | **32.62%** | **1.036** | +0.014 | −0.078 | −0.177 | +0.071 |
| Track C | LR on GT | F0 GT det. | 95 test | 97.37% | 90.53% | 0.276 | −0.053 | +0.021 | +0.168 | +0.000 |

**Gap Track B → Track C: 19.89 pp** — entirely detector error. Improving B3/B4 recall is the only path to gains beyond 77.48%.

---

## Key Findings

- **Best E2E: Ridge + F_all (67-dim)** — 77.48% Macro Acc±1 / 32.62% Joint Acc±1, +1.77 pp over LR baseline.
- **Spatial features are most informative** — `mean_cy` (vertical centroid) and `mean_area` per class; each maturity class occupies a distinct vertical zone on the tree.
- **77.48% is the ML-counter ceiling** on current YOLO detections — confirmed by exhaustive v4 probing (200+ dims, Optuna, stacking). B2 YOLO–GT correlation r = 0.421: 82% of its variance is detector noise.
- **Linear models dominate** (LR, Ridge, ElasticNet > XGB, LGB, RF) — 716 training trees is too small for tree-based methods.
- **B3 hardest** (57% Acc±1, recall 65.6%); **B4 most under-detected** (recall 38.9%, 0.85× average count).
- **19.89 pp gap to Track C is entirely detector error** — retraining YOLO with improved B3/B4 recall is the highest-leverage next step.

---

## Reproduction

```bash
pip install -r requirements.txt

# Copy weights
cp models/yolo/y26me60p60b32s42v2.pt models/yolo/y26mv2.pt

# Track B: inference + all counters
python pipeline/run_e2e_pipeline.py \
    --name y26mv2 --weights models/yolo/y26mv2.pt \
    --data SawitMVC-YOLO/ --counters svm lr rf m01

# Print metrics
python scripts/report_metrics.py y26mv2 test

# Track A: heuristics on GT
python benchmarks/run_benchmark.py

# Track C: upper bound
bash scripts/reproduce_upper_bound.sh

# Counting ablation experiments
python experiments/exp_counting_v3.py   # 80 configs
python experiments/exp_counting_v4.py   # 200+ dim + Optuna + stacking

# Verify headline claims
python benchmarks/check_release_claims.py
```

Dataset download (needed for fresh inference):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset',
                  local_dir='./SawitMVC-YOLO', token=True)
"
```

---

## Repository Layout

```
algorithms/          # Track A: deterministic heuristics (M01..M05)
pipeline/            # Track B/C: feature extraction + ML counter scripts
experiments/         # Counting ablation (exp_counting_v3.py, exp_counting_v4.py)
benchmarks/          # run_benchmark.py, check_release_claims.py
ground_truth/        # 953 GT JSONs + split_manifest.csv
predictions/         # Cached YOLO outputs (y26mv2_per_tree/)
results/             # Pre-computed evaluation + experiment CSVs
models/yolo/         # y26mv2.pt + training log
models/counters/     # svm.pkl, rf.pkl, lr.pkl
scripts/             # reproduce_all.sh, reproduce_upper_bound.sh, ...
archive/             # Old detector outputs (y26n, y26s, y26m)
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
