# Evaluation: Metrics and Protocol

This document defines all metrics used in the SawitMVC Baseline benchmark and
describes the evaluation protocol to ensure fair, reproducible comparisons.

---

## Primary Metric: Acc ±1

**Definition:** Percentage of trees where every class prediction is within ±1 of
the ground truth, computed as the macro average across all 4 maturity classes.

For a single tree with prediction `P = {B1, B2, B3, B4}` and ground truth `G = {B1, B2, B3, B4}`:

```
correct(tree) = 1  if |P[c] - G[c]| ≤ 1 for ALL c ∈ {B1, B2, B3, B4}
                0  otherwise
```

Overall:
```
Acc±1 = sum(correct(tree) for all trees) / total_trees × 100%
```

**Why ±1 tolerance?** Exact prediction on 4 simultaneous classes is extremely
hard (only ~27% of trees are exactly correct even for the best algorithm). A ±1
tolerance acknowledges annotation uncertainty and practical utility in agricultural
settings where ±1 bunch does not affect harvesting decisions.

---

## Secondary Metrics

### Macro class-MAE

Unweighted mean absolute error across the 4 classes, averaged over all trees:

```
Macro MAE = (1/N) × Σ_tree [ (|P[B1]-G[B1]| + |P[B2]-G[B2]| + |P[B3]-G[B3]| + |P[B4]-G[B4]|) / 4 ]
```

This metric penalizes large errors proportionally and is sensitive to systematic
class-level bias. M07 has the lowest Macro MAE (0.368) even though it is ranked 7th
by Acc±1, different optima for different metrics.

### Total-count MAE

MAE of the sum of all 4 classes per tree:

```
Total MAE = mean(|sum(P) - sum(G)|)  for all trees
```

This is relevant for practical harvest estimation where the total bunch count matters
more than per-class breakdown.

### Total ±1 accuracy

```
Total±1 = count(trees where |sum(P) - sum(G)| ≤ 1) / total_trees × 100%
```

### Exact-profile accuracy

Percentage of trees where the full 4-class prediction vector is exactly correct:

```
Exact = count(trees where P[c] == G[c] for ALL c) / total_trees × 100%
```

The best algorithm achieves only 27.07% exact-profile accuracy, highlighting the
inherent difficulty of the 4-class joint prediction.

### Per-class MAE and bias

Per-class MAE: `MAE[c] = mean(|P[c] - G[c]|)` for each class c.

Per-class bias: `bias[c] = mean(P[c] - G[c])`, positive means systematic over-count,
negative means systematic under-count.

---

## Evaluation Dataset

All benchmarks use the **SawitMVC-YOLO** canonical 953-tree dataset
(post-GT-fix 2026-05-16).

| Subset | Trees | Notes |
|--------|------:|-------|
| Full dataset | 953 | Primary benchmark |
| Train split | 758 | Never used for algorithm tuning |
| Val split | - | |
| Test split | 95 | Secondary reference; identical trend to full |

**All algorithms are evaluated on all 953 trees**: not just the test split, because
the heuristic algorithms have no trainable parameters. The full dataset provides a
more stable estimate of generalization performance.

---

## Ground Truth Source

The ground truth for counting is `summary.by_class` in each tree's JSON file. This
is derived from the `_confirmedLinks` field by running Union-Find connected components:
boxes linked across camera sides are the same physical bunch and counted once.

Ground truth has been validated for:
- Zero same-side uniqueness violations
- Zero geometric adjacency violations
- 100% image–label–JSON file integrity

---

## Evaluation Protocol

1. **Load** all 953 JSON files from `ground_truth/annotations/`
2. **Extract** detections for each tree: one dict per bounding box with keys
   `class`, `x_norm`, `y_norm`, `side_index`
3. **Run** the algorithm's `predict(detections)` function
4. **Compare** predictions to `summary.by_class` using the metrics above
5. **Report** Acc±1 %, Macro MAE, Total MAE, and (optionally) per-class breakdown

The benchmark runner (`benchmarks/run_benchmark.py`) implements this protocol exactly.

---

## Reporting New Results

When contributing a new algorithm or reporting results in a paper, please include:

| Metric | Required |
|--------|:--------:|
| Acc ±1 (%) | ✅ |
| Macro class-MAE | ✅ |
| Total-count MAE | ✅ |
| n_fail (trees outside ±1) | ✅ |
| Per-class MAE (B1–B4) | Recommended |
| Exact-profile accuracy | Optional |
| Evaluation dataset | ✅ (must be 953-tree SawitMVC-YOLO) |
| GT version date | ✅ (must be post-fix 2026-05-16 or later) |

Results on the historical 228/478/727/882-tree subsets are for reference only and
**must not** be compared directly to the 953-tree benchmark.

---

## YOLO Detection Metrics

For detection models, report:

| Metric | Notes |
|--------|-------|
| mAP50 | Primary detection metric |
| mAP50-95 | Secondary (COCO standard) |
| Inference speed (ms) | At batch=1, imgsz=640, on the test hardware |
| Model size (MB) | Weights file size |
| Best epoch | For training reproducibility |

Evaluate on the `test` split of `ground_truth/data.yaml`:
```bash
yolo detect val model=<weights.pt> data=ground_truth/data.yaml split=test
```
