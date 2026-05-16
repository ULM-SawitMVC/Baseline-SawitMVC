# Benchmarks — Reproducible Evaluation

This folder contains pre-computed results and a runnable benchmark script for the
SawitMVC deduplication algorithms.

---

## Quick Run

```bash
# From repository root
python benchmarks/run_benchmark.py --data ./SawitMVC-YOLO/json/
```

Expected output (953 trees):

```
SawitMVC Baseline — Benchmark Results (953 trees)
==================================================
Rank  Algorithm                       Acc±1    MAE   Total MAE   Fail
-----------------------------------------------------------------
  1🥇  M01_selector_b2b3             87.62%  0.3746     1.3305    118
  2🥈  M02_selector_trifurc          87.62%  0.3757     1.3305    118
  3🥉  M03_blend_geometric           86.99%  0.3767     1.3410    124
  4    M04_blend_floor_clamped       86.99%  0.3848     1.3421    124
  5    M05_blend_vis_divide          86.99%  0.3875     1.3463    124
-----------------------------------------------------------------
       Naive sum (reference)          3.78%  2.2867     9.1469
```

Save results to CSV:

```bash
python benchmarks/run_benchmark.py --data ./SawitMVC-YOLO/json/ --save
```

---

## Metrics Definitions

### Acc ±1 (Primary metric)

Percentage of trees where every class prediction is within ±1 of ground truth,
macro-averaged across the 4 maturity classes.

For a single tree:
- `within1 = all(|pred[c] - gt[c]| ≤ 1 for c in {B1, B2, B3, B4})`

Overall: `Acc±1 = count(within1) / total_trees × 100%`

### Macro class-MAE

Mean absolute error averaged uniformly across the 4 classes:
```
Macro MAE = mean(|pred[B1]-gt[B1]|, |pred[B2]-gt[B2]|, |pred[B3]-gt[B3]|, |pred[B4]-gt[B4]|)
```
Then averaged over all trees.

### Total-count MAE

MAE of the total bunch count per tree:
```
Total MAE = mean(|sum(pred) - sum(gt)|)
```

### Exact-profile accuracy

Percentage of trees where the predicted vector `[B1, B2, B3, B4]` exactly matches
ground truth. This is a stricter metric than Acc±1 and reflects the full 4-class joint accuracy.

### Total ±1 accuracy

Percentage of trees where the total predicted count is within ±1 of the ground-truth total.

---

## Pre-computed Results

### `results/accuracy_953.csv`

Full ranking of all 29 heuristic algorithms evaluated on 953 trees.

Columns:
- `method` — algorithm name
- `acc_within1_pct` — Acc±1 (%)
- `macro_class_mae` — macro-averaged MAE
- `total_count_mae` — MAE of total count
- `total_within1_pct` — total ±1 accuracy (%)
- `exact_profile_pct` — exact-profile accuracy (%)
- `mae_B1`, `mae_B2`, `mae_B3`, `mae_B4` — per-class MAE
- `n_fail` — number of trees outside ±1

### `results/per_tree.csv`

Row per tree (953 rows) with predictions from all 29 methods.

Columns: `tree_id`, `split`, `n_dets`, `n_sides`, then for each method:
`{method}_B1`, `{method}_B2`, `{method}_B3`, `{method}_B4`, `{method}_total`

### `results/totals.csv`

Aggregate bunch counts per method (total B1/B2/B3/B4 summed across all 953 trees).

### `results/mean_per_tree.csv`

Mean predicted counts per tree per method.

### `e2e/y26m_svm_metrics.json`

Best end-to-end result: `y26m_vanilla_local` detector + SVM counter.

Key metrics:
- `macro_acc_within1`: 0.716 (71.6%)
- `macro_class_mae`: 1.118
- `total_count_mae`: 2.432
- Per-class: B1=92.6%, B2=63.2%, B3=60.0%, B4=70.5%

---

## Dataset Requirements

The benchmark script requires the `Brand-New-Dataset-YOLO` JSON ground-truth files.
Each `.json` file corresponds to one tree and follows this minimal schema:

```json
{
  "tree_id": "DAMIMAS_A21B_0001",
  "images": {
    "side_1": {
      "side_index": 0,
      "annotations": [
        {"class_name": "B3", "bbox_yolo": [0.512, 0.341, 0.08, 0.12]}
      ]
    }
  },
  "summary": {
    "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
  }
}
```

Download the full dataset:
```python
from huggingface_hub import snapshot_download
snapshot_download("ULM-DS-Lab/SawitMVC-YOLO", repo_type="dataset", local_dir="./SawitMVC-YOLO")
```

---

## Reproducing the Full 29-method Benchmark

The pre-computed CSVs already contain results for all 29 methods. To reproduce them
from scratch using the original research code:

```bash
# Clone the research repository
git clone https://github.com/muhammad-zainal-muttaqin/research-method-dedup.git
cd research-method-dedup

# Run the canonical 953-tree benchmark
python scripts/dedup_brand_new_953.py
# → outputs to reports/dedup_brand_new_953/accuracy_953.csv
```
