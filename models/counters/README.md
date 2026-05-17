# Counter Artifacts

This folder stores the trained machine-learning counters used by Track B of the
end-to-end pipeline. Each `.pkl` file is a pickled scikit-learn estimator that
maps a 13-dimensional feature vector (built from per-tree YOLO detections by
[`pipeline/build_counting_features.py`](../../pipeline/build_counting_features.py))
to the per-class bunch count vector `[B1, B2, B3, B4]`.

## Files

| File | Estimator | Training set | Best test Acc±1 |
|------|-----------|--------------|-----------------|
| [`svm.pkl`](svm.pkl) | `Pipeline(StandardScaler, MultiOutputRegressor(SVR(rbf)))` with `GridSearchCV` over `C ∈ {0.1, 1, 10}` and `gamma ∈ {scale, 0.01, 0.1}`, 3-fold CV. | y26s per-tree predictions, train split (n=763). | 70.79% (see [`results/e2e_per_tree/y26s_svm/metrics.json`](../../results/e2e_per_tree/y26s_svm/metrics.json)). |
| [`rf.pkl`](rf.pkl) | `RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)`. | Same. | 64.21% (see [`results/e2e_per_tree/y26s_rf/metrics.json`](../../results/e2e_per_tree/y26s_rf/metrics.json)). |
| [`lr.pkl`](lr.pkl) | `Pipeline(StandardScaler, LinearRegression())`. | Same. | 68.68% (see [`results/e2e_per_tree/y26s_lr/metrics.json`](../../results/e2e_per_tree/y26s_lr/metrics.json)). |

Each estimator predicts continuous values which the counting scripts then clip
to `[0, +inf)` and round to the nearest non-negative integer.

## Regenerating the artifacts

Run the following from the repository root after a fresh clone. Each command
prints a metrics summary and replaces the corresponding `.pkl` file:

```
python pipeline/run_counting_svm.py --inference-dir predictions/y26s_per_tree --save-model models/counters/svm.pkl
python pipeline/run_counting_rf.py  --inference-dir predictions/y26s_per_tree --save-model models/counters/rf.pkl
python pipeline/run_counting_lr.py  --inference-dir predictions/y26s_per_tree --save-model models/counters/lr.pkl
```

Random seeds are pinned to `42` in each script, but the SVM grid search uses
`n_jobs=-1` (multiprocess). Across machines this can perturb CV-MAE by less
than 0.5%, which leaves the headline test-split numbers unchanged.

## Loading an artifact for evaluation only

The three counting scripts accept `--load-model` to skip training entirely:

```
python pipeline/run_counting_svm.py --load-model models/counters/svm.pkl \
    --inference-dir predictions/y26s_per_tree --out results/e2e_per_tree/y26s_svm
```

This regenerates `metrics.json` and `predictions.csv` for the chosen detector
in seconds, with no GridSearchCV in the loop. Useful when swapping the
inference set (for example, evaluating the same counter on `y26n_per_tree` or
`y26m_per_tree` predictions without retraining).

## Why train on y26s features?

The y26s detector is the best-performing per-tree pipeline at 70.79% test
Acc±1 (see [`results/e2e_per_tree/`](../../results/e2e_per_tree/)). Fitting the
counters on these features yields the strongest published numbers; the same
counter, evaluated against features from a different detector, still recovers
useful counts but does not generalise the same way as a counter trained on
matching features. To reproduce a fully matched detector/counter pair, refit
the counter on the chosen detector’s predictions and pass `--save-model` with
a distinct path.
