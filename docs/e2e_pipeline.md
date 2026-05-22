# End-to-End Pipeline: Current Baseline

This note documents the active end-to-end path used by the latest release:

```
y26mv2 detections -> 13-dim per-tree features -> counter -> per-class count
```

The canonical split is:

- train: 716 trees
- val: 96 trees
- test: 141 trees

## Track B: current stored baseline

Current cached predictions live in
[`predictions/y26mv2_per_tree/`](../predictions/y26mv2_per_tree/), and the
stored baseline counters are in [`results/e2e_per_tree/`](../results/e2e_per_tree/).

| Counter | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|--------|-----|------------:|------------:|----------:|
| LR | 141 test | 75.71% | 30.50% | 1.048 |
| SVM | 141 test | 74.82% | 29.08% | 1.043 |
| RF | 141 test | 73.23% | 26.95% | 1.110 |
| M01 | 141 test | 70.57% | 24.11% | 1.183 |

## Track B: counter improvement experiments

The best single configuration on the same detector outputs comes from the
experiment CSVs in [`results/experiments/`](../results/experiments/):

| Method | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|--------|----------|-----|------------:|------------:|----------:|
| **Ridge** | **F_all (67-dim)** | **141 test** | **77.48%** | **32.62%** | **1.036** |
| ElasticNet | F0+spatial (21-dim) | 141 test | 76.77% | 31.21% | 1.039 |

This is the current Track B ceiling for the repo’s published `y26mv2`
detections.

## Track C: ground-truth upper bound

When the same counters are trained on GT-derived features instead of YOLO
predictions:

| Counter | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|--------|-----|------------:|------------:|----------:|
| LR | 141 test | 97.52% | 90.07% | 0.277 |
| SVM | 141 test | 97.87% | 91.49% | 0.266 |
| RF | 141 test | 95.92% | 84.40% | 0.365 |
| Ridge | 141 test | 97.70% | 90.78% | 0.275 |
| **ElasticNet** | **141 test** | **98.05%** | **92.20%** | **0.277** |

The Track B to Track C gap is therefore **20.57 pp** in Class ±1 Acc. That gap
is detector error, not counter design.

## Reproduction

```bash
# Track B: current stored baseline
python pipeline/run_e2e_pipeline.py --name y26mv2 --skip-inference

# Track C: GT upper bound
bash scripts/reproduce_upper_bound.sh

# Verify README headline numbers
python benchmarks/check_release_claims.py
```

## Historical E2E variants

Older `y26n`, `y26s`, `y26m`, and per-image experiments are archived under
[`archive/`](../archive/). They are kept for reference only and should not be
treated as the latest version of the baseline.
