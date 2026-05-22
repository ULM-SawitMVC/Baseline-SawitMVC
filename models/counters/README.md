# Counter Artifacts

This folder stores reusable scikit-learn counters for the current per-tree
counting pipeline.

## Files

| File | Estimator |
|------|-----------|
| [`svm.pkl`](svm.pkl) | `Pipeline(StandardScaler, MultiOutputRegressor(SVR(rbf)))` |
| [`rf.pkl`](rf.pkl) | `RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)` |
| [`lr.pkl`](lr.pkl) | `Pipeline(StandardScaler, LinearRegression())` |

They map the 13-dim feature vector produced by
[`pipeline/build_counting_features.py`](../../pipeline/build_counting_features.py)
to the per-class bunch count `[B1, B2, B3, B4]`.

## Regenerating on the current baseline

```bash
python pipeline/run_counting_svm.py --inference-dir predictions/y26mv2_per_tree --save-model models/counters/svm.pkl
python pipeline/run_counting_rf.py  --inference-dir predictions/y26mv2_per_tree --save-model models/counters/rf.pkl
python pipeline/run_counting_lr.py  --inference-dir predictions/y26mv2_per_tree --save-model models/counters/lr.pkl
```

## Evaluating with a saved artifact

```bash
python pipeline/run_counting_svm.py \
    --load-model models/counters/svm.pkl \
    --inference-dir predictions/y26mv2_per_tree \
    --out results/e2e_per_tree/y26mv2_svm
```

The latest published baseline metrics in the repo are:

- `y26mv2_lr`: 75.71% Class ±1 Acc
- `y26mv2_svm`: 74.82% Class ±1 Acc
- `y26mv2_rf`: 73.23% Class ±1 Acc

Counter-improvement experiments beyond these stored artifacts are tracked in
[`results/experiments/`](../../results/experiments/).
