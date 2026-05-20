# Pipeline: Current Baseline Scripts

This folder contains the active Track B and Track C scripts used by the latest
`y26mv2` baseline. All of them read the bundled ground truth at
[`ground_truth/annotations/`](../ground_truth/annotations/) and the canonical
split at [`ground_truth/split_manifest.csv`](../ground_truth/split_manifest.csv).

## Active scripts

| Script | Purpose |
|--------|---------|
| [`run_e2e_inference.py`](run_e2e_inference.py) | YOLO inference grouped per tree (one JSON per tree). |
| [`run_e2e_pipeline.py`](run_e2e_pipeline.py) | Track B harness: inference plus SVM, RF, LR, and M01 for `y26mv2`. |
| [`build_counting_features.py`](build_counting_features.py) | 13-dim feature extraction from per-tree detections. |
| [`run_counting_svm.py`](run_counting_svm.py) | SVM counter on 13-dim features. |
| [`run_counting_rf.py`](run_counting_rf.py) | Random Forest counter on 13-dim features. |
| [`run_counting_lr.py`](run_counting_lr.py) | Linear regression counter on 13-dim features. |

## Quick start

```bash
python pipeline/run_e2e_pipeline.py --name y26mv2 --skip-inference
```

This reuses the cached predictions in
[`predictions/y26mv2_per_tree/`](../predictions/y26mv2_per_tree/) and writes:

- `results/e2e_per_tree/y26mv2_svm/`
- `results/e2e_per_tree/y26mv2_rf/`
- `results/e2e_per_tree/y26mv2_lr/`
- `results/e2e_per_tree/y26mv2_m01/`

## Feature vector

The current counter input is the 13-dim per-tree feature vector:

```
naive_sum_B1, naive_sum_B2, naive_sum_B3, naive_sum_B4,
max_per_side_B1, max_per_side_B2, max_per_side_B3, max_per_side_B4,
mean_per_side_B1, mean_per_side_B2, mean_per_side_B3, mean_per_side_B4,
n_sides
```

## Track C

To fit the counter directly on GT-derived features:

```bash
bash scripts/reproduce_upper_bound.sh
```

This writes:

- `results/e2e_upper_bound/gt_svm/`
- `results/e2e_upper_bound/gt_rf/`
- `results/e2e_upper_bound/gt_lr/`

## Archived pipeline variants

The per-image experimental pipeline and other legacy variants were moved to
[`archive/pipeline/`](../archive/pipeline/). They are historical experiments
and are not part of the latest baseline path.
