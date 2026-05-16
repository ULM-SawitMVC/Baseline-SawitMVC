# Pipeline — End-to-End Replication Scripts

Two E2E approaches available (produce identical ML-counter results since features are the same):

| Approach | Script | Description |
|----------|--------|-------------|
| **Per-tree** | `run_e2e_pipeline.py` | Groups all 4-8 images per tree → 1 JSON → ML counting |
| **Per-image** | `run_e2e_per_image.py` | Each image processed independently → group → ML counting |

Both approaches support the same 7 counters: `max`, `mean`, `sum`, `m01`, `svm`, `rf`, `lr`.
The per-image approach additionally supports simple aggregation (`max`/`mean`/`sum`) before ML.

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `run_e2e_inference.py` | YOLO inference grouped per tree (4-8 images → 1 JSON) |
| `run_e2e_per_image.py` | **Complete per-image pipeline**: YOLO per image → group → max/mean/sum/M01/SVM/RF/LR |
| `build_counting_features.py` | Extract 13-dim feature vectors from inference JSONs |
| `run_counting_svm.py` | SVM counter (RBF kernel, GridSearchCV) |
| `run_counting_rf.py` | Random Forest counter (n=200, max_depth=10) |
| `run_counting_lr.py` | **Linear Regression** counter (interpretable baseline) |
| `run_e2e_pipeline.py` | Unified harness: inference → SVM + RF + LR + M01 |

---

## Quick Start (Full Pipeline in One Command)

**Per-tree approach** (groups all images per tree before counting):
```bash
# From repo root — download dataset first (see docs/dataset.md)
python pipeline/run_e2e_pipeline.py \
    --name y26n \
    --weights models/y26n.pt

# Output:
#   predictions/y26n_inference/        ← 953 JSON files (skipped if exists)
#   benchmarks/e2e/e2e_y26n_svm/       ← SVM metrics.json + predictions.csv
#   benchmarks/e2e/e2e_y26n_rf/        ← RF  metrics.json + predictions.csv
#   benchmarks/e2e/e2e_y26n_lr/        ← LR  metrics.json + predictions.csv
#   benchmarks/e2e/e2e_y26n_m01/       ← M01 metrics.json
```

**Per-image approach** (each image processed independently, then grouped):
```bash
# Using pre-computed per-tree predictions (no GPU needed — derives per-image data automatically)
python pipeline/run_e2e_per_image.py \
    --name y26n \
    --weights models/y26n.pt \
    --data ./SawitMVC-YOLO \
    --skip-inference

# Output (7 counters):
#   predictions/y26n_per_image/                   ← 3992 per-image JSONs
#   benchmarks/e2e/e2e_y26n_per_image_max/        ← simple max agg
#   benchmarks/e2e/e2e_y26n_per_image_mean/
#   benchmarks/e2e/e2e_y26n_per_image_sum/
#   benchmarks/e2e/e2e_y26n_per_image_m01/        ← M01 heuristic
#   benchmarks/e2e/e2e_y26n_per_image_svm/        ← SVM counter
#   benchmarks/e2e/e2e_y26n_per_image_lr/         ← LR counter
#   benchmarks/e2e/e2e_y26n_per_image_rf/         ← RF counter
```

---

## Step-by-Step (Manual)

### Step 1 — YOLO Inference

> **Skip this step** if using pre-computed predictions in `predictions/`.

```bash
python pipeline/run_e2e_inference.py \
    --name y26n \
    --weights models/y26n.pt
# → predictions/y26n_inference/{tree_id}.json (953 files)
```

### Step 2 — Feature Extraction

Extracts a 13-dim feature vector per tree from the YOLO inference JSONs.

```bash
python pipeline/build_counting_features.py \
    --inference-dir predictions/y26n_inference/
# → used internally by run_counting_svm.py / run_counting_rf.py
```

### Step 3 — Train Counter and Evaluate

```bash
# SVM (RBF kernel, GridSearchCV)
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26n_inference/

# Random Forest (n=200, max_depth=10)
python pipeline/run_counting_rf.py \
    --inference-dir predictions/y26n_inference/
```

---

## Pre-computed Predictions

The `predictions/` folder contains YOLO inference outputs for all 3 trained models
(953 trees × 3 detectors = 2,859 JSON files, ~17 MB total).

**You can skip Step 1** entirely and run Steps 2–3 directly on existing predictions:

```bash
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26m_inference/
```

Available prediction sets:

| Folder | Detector | mAP50 |
|--------|----------|:-----:|
| `y26m_inference/` | YOLO26m | **0.528** |
| `y26n_inference/` | YOLO26n | 0.515 |
| `y26s_inference/` | YOLO26s | 0.511 |

---

## Feature Vector Format (13 dimensions)

| Feature | Description |
|---------|-------------|
| `naive_sum_B1–B4` | Raw detection count per class (4 features) |
| `max_per_side_B1–B4` | Max detections on any single camera side per class (4 features) |
| `mean_per_side_B1–B4` | Mean detections per side per class (4 features) |
| `n_sides` | Number of camera sides (4 or 8) |

Target: `[gt_B1, gt_B2, gt_B3, gt_B4]` — unique GT bunch count per class.

---

## Designing a New Ablation

### New detector (e.g., your retrained model)

```bash
python pipeline/run_e2e_pipeline.py \
    --name my_new_detector \
    --weights path/to/my_weights.pt
# Results → benchmarks/e2e/e2e_my_new_detector_{svm,rf,lr,m01}/
```

Then compare with pre-computed baselines in `benchmarks/e2e/`.

### New counting algorithm

1. Add your algorithm as `algorithms/M{NN}_your_algo.py` with `predict(detections) -> dict`
2. Edit `pipeline/run_e2e_pipeline.py` to include it alongside the SVM/RF/M01 block
3. Re-run using existing predictions (no re-inference needed):

```bash
python pipeline/run_e2e_pipeline.py \
    --name y26n \
    --weights models/y26n.pt \
    --skip-inference
```

### New feature engineering

Edit `pipeline/build_counting_features.py` to add features (e.g., bbox aspect ratio,
spatial density, inter-side class agreement) and re-run the SVM/RF steps.
All 2,859 pre-computed prediction JSONs remain valid inputs.
