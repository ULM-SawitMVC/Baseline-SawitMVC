# Pipeline: End-to-End Replication Scripts

This folder houses every script in Tracks B, B', and C of the SawitMVC
baseline. All scripts read the bundled ground truth at
[`ground_truth/annotations/`](../ground_truth/annotations/) and the canonical
split at [`ground_truth/split_manifest.csv`](../ground_truth/split_manifest.csv).
They write results under [`results/`](../results/) and YOLO predictions under
[`predictions/`](../predictions/).

## Two end-to-end approaches

| Approach | Driver | Output folder |
|----------|--------|---------------|
| Per-tree (Track B) | [`run_e2e_pipeline.py`](run_e2e_pipeline.py) | [`results/e2e_per_tree/`](../results/e2e_per_tree/) |
| Per-image (Track B') | [`run_e2e_per_image.py`](run_e2e_per_image.py) | [`results/e2e_per_image/`](../results/e2e_per_image/) |

Per-tree groups the 4–8 images of a tree before inference and produces one
JSON per tree. Per-image runs inference image-by-image and groups afterwards.
Both pipelines derive identical 13-dimensional feature vectors and therefore
yield numerically identical ML-counter results; the per-image pipeline adds
three simple aggregators (`max`, `mean`, `sum`) that the per-tree pipeline
does not need.

## Script inventory

| Script | Purpose |
|--------|---------|
| [`run_e2e_inference.py`](run_e2e_inference.py) | YOLO inference grouped per tree (one JSON per tree). |
| [`run_e2e_pipeline.py`](run_e2e_pipeline.py) | Unified Track B harness: inference plus SVM, RF, LR, and M01. |
| [`run_e2e_per_image.py`](run_e2e_per_image.py) | Track B' harness with seven counters: `max`, `mean`, `sum`, `m01`, `svm`, `rf`, `lr`. |
| [`build_counting_features.py`](build_counting_features.py) | 13-dim feature library; importable or runnable. |
| [`run_counting_svm.py`](run_counting_svm.py) | SVM counter (RBF + `GridSearchCV`); supports `--save-model` / `--load-model`. |
| [`run_counting_rf.py`](run_counting_rf.py) | Random Forest counter (`n=200`, `max_depth=10`, `random_state=42`). |
| [`run_counting_lr.py`](run_counting_lr.py) | Linear regression with `StandardScaler`; emits a coefficients CSV. |

Each script pins `random`, `numpy.random`, and `PYTHONHASHSEED` to `42`.

## Quick start

Reproduce the per-tree pipeline for the best detector, reusing the cached
predictions and the bundled ground truth:

```bash
python pipeline/run_e2e_pipeline.py --name y26s --skip-inference
# Writes:
#   results/e2e_per_tree/y26s_svm/{metrics.json, predictions.csv}
#   results/e2e_per_tree/y26s_rf/{metrics.json, predictions.csv, feature_importance.csv}
#   results/e2e_per_tree/y26s_lr/{metrics.json, predictions.csv, coefficients.csv}
#   results/e2e_per_tree/y26s_m01/{metrics.json, predictions.csv}
```

Per-image variant (also CPU-only when `--skip-inference` is set):

```bash
python pipeline/run_e2e_per_image.py --name y26s --skip-inference
# Writes seven folders under results/e2e_per_image/y26s_{max,mean,sum,m01,svm,rf,lr}/.
```

## Step-by-step

### Step 1: YOLO inference

Required only when retraining or rerunning detection. The repository ships
pre-computed per-tree predictions for every detector in
[`predictions/`](../predictions/).

```bash
python pipeline/run_e2e_inference.py --name y26s \
    --weights models/yolo/y26s.pt --data ./SawitMVC-YOLO/
# Output: predictions/y26s_per_tree/{tree_id}.json (953 files)
```

### Step 2: Feature extraction

The library reconstructs a 13-dim feature vector per tree.

```bash
python pipeline/build_counting_features.py --inference-dir predictions/y26s_per_tree/
```

### Step 3: Counter training and evaluation

Either refit the counter (default behaviour) or reload an artifact from
[`models/counters/`](../models/counters/) and skip training.

```bash
# Refit and save
python pipeline/run_counting_svm.py --inference-dir predictions/y26s_per_tree \
    --save-model models/counters/svm.pkl

# Reload and evaluate only
python pipeline/run_counting_svm.py --inference-dir predictions/y26s_per_tree \
    --load-model models/counters/svm.pkl
```

The same `--save-model` / `--load-model` pair exists in `run_counting_rf.py`
and `run_counting_lr.py`.

## Feature vector specification

The 13 dimensions, in order:

```
naive_sum_B1,  naive_sum_B2,  naive_sum_B3,  naive_sum_B4,
max_per_side_B1, max_per_side_B2, max_per_side_B3, max_per_side_B4,
mean_per_side_B1, mean_per_side_B2, mean_per_side_B3, mean_per_side_B4,
n_sides
```

Target: `[gt_B1, gt_B2, gt_B3, gt_B4]`, the unique GT bunch count per class
read from `summary.by_class` in each ground-truth JSON.

## Cached predictions

| Folder | Detector | Files | mAP50 |
|--------|----------|------:|------:|
| [`predictions/y26m_per_tree/`](../predictions/y26m_per_tree/) | YOLOv26 medium | 953 | 0.528 |
| [`predictions/y26n_per_tree/`](../predictions/y26n_per_tree/) | YOLOv26 nano | 953 | 0.515 |
| [`predictions/y26s_per_tree/`](../predictions/y26s_per_tree/) | YOLOv26 small | 953 | 0.511 |
| [`predictions/y26m_per_image/`](../predictions/y26m_per_image/) | derived from per-tree | 3,992 | - |
| [`predictions/y26n_per_image/`](../predictions/y26n_per_image/) | derived | 3,992 | - |
| [`predictions/y26s_per_image/`](../predictions/y26s_per_image/) | derived | 3,992 | - |

## Designing a new ablation

### New detector

```bash
python pipeline/run_e2e_pipeline.py --name my_detector \
    --weights path/to/my_weights.pt --data ./SawitMVC-YOLO/
```

The script will inference, extract features, train each counter, and write
results to `results/e2e_per_tree/my_detector_{svm,rf,lr,m01}/`.

### New counting algorithm

1. Implement `predict(detections) -> dict[str, int]` in
   `algorithms/M{NN}_{family}_{descriptor}.py`.
2. Register it in [`algorithms/__init__.py`](../algorithms/__init__.py).
3. Re-run the heuristic benchmark:
   ```bash
   python benchmarks/run_benchmark.py --save
   ```
4. Compare against the existing rows in
   [`results/heuristics_953/accuracy_full.csv`](../results/heuristics_953/accuracy_full.csv).

### New features

Extend [`build_counting_features.py`](build_counting_features.py) by adding to
`FEATURE_NAMES` and to `extract_features(...)`. All 2,859 cached prediction
JSONs remain valid inputs; only the SVM / RF / LR runs need to be repeated.
