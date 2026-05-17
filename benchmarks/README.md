# Benchmarks — Reproducible Evaluation

This folder contains the two top-level scripts that drive every evaluation in
the repository. All pre-computed numerical artifacts live one level up under
[`results/`](../results/).

| Script | Purpose |
|--------|---------|
| [`run_benchmark.py`](run_benchmark.py) | Track A — runs the five heuristic deduplicators across all 953 trees and prints a ranked table. |
| [`check_release_claims.py`](check_release_claims.py) | Release guard — verifies that the headline numbers in [`README.md`](../README.md) still match the artifacts in [`results/`](../results/). |

## Quick run

```bash
# From the repository root
python benchmarks/run_benchmark.py
```

Default `--data` resolves to [`ground_truth/annotations/`](../ground_truth/annotations/),
the 953 GT JSONs bundled in this repository. To save the printed table to
[`results/heuristics_953/benchmark_top5.csv`](../results/heuristics_953/) add `--save`.

Expected output:

```
SawitMVC Baseline - Benchmark Results (953 trees)
==================================================
Rank  Algorithm                       Acc+/-1    MAE   Total MAE   Fail
----------------------------------------------------------------------
   1  M01_selector_b2b3                87.62%  0.3746     1.3305    118
   2  M02_selector_trifurc             87.62%  0.3757     1.3305    118
   3  M03_blend_geometric              86.99%  0.3767     1.3410    124
   4  M04_blend_floor_clamped          86.99%  0.3848     1.3421    124
   5  M05_blend_vis_divide             86.99%  0.3875     1.3463    124
----------------------------------------------------------------------
      Naive sum (reference)             3.78%  2.2867     9.1469
```

## Headline-claims guard

Before publishing a new release or amending the README tables:

```bash
python benchmarks/check_release_claims.py
```

The script reads
[`results/heuristics_953/accuracy_full.csv`](../results/heuristics_953/accuracy_full.csv)
and every
[`results/e2e_per_tree/*/metrics.json`](../results/e2e_per_tree/), then compares
against the values quoted in the README. It exits 0 on success and prints the
offending rows on failure.

## Metric definitions

The exhaustive specification lives at [`docs/evaluation.md`](../docs/evaluation.md);
the short version follows.

- **Acc±1** (primary): fraction of trees where the predicted count differs
  from the ground truth by at most one bunch, macro-averaged across the four
  classes.
- **Macro class-MAE**: mean absolute error per class, then averaged across the
  four classes.
- **Total-count MAE**: mean absolute error of the tree-level total.
- **Exact-profile accuracy**: fraction of trees where all four classes match
  exactly. Strictest of the four.
- **Total ±1 accuracy**: fraction of trees where the tree-level total is
  within ±1 of the ground-truth total.

## Result artefacts

| Path | What it contains |
|------|------------------|
| [`results/heuristics_953/accuracy_full.csv`](../results/heuristics_953/accuracy_full.csv) | Full ranking of the 29-method heuristic exploration. |
| [`results/heuristics_953/per_tree.csv`](../results/heuristics_953/per_tree.csv) | One row per tree (n = 953) with predictions from every method. |
| [`results/heuristics_953/totals.csv`](../results/heuristics_953/totals.csv) | Aggregate per-class counts per method. |
| [`results/heuristics_953/mean_per_tree.csv`](../results/heuristics_953/mean_per_tree.csv) | Mean per-tree predictions per method. |
| [`results/e2e_per_tree/`](../results/e2e_per_tree/) | 12 folders, one per detector × counter (Track B). |
| [`results/e2e_per_image/`](../results/e2e_per_image/) | 21 folders covering simple aggregation and ML counters on per-image inference (Track B'). |
| [`results/e2e_upper_bound/`](../results/e2e_upper_bound/) | 3 folders for SVM, RF, LR fitted on ground-truth-derived features (Track C). |

## Ground-truth schema

Every JSON in [`ground_truth/annotations/`](../ground_truth/annotations/)
documents one tree. The annotation schema, split layout, and class taxonomy
are described in [`ground_truth/README.md`](../ground_truth/README.md). For
the broader collection methodology see
[`docs/dataset.md`](../docs/dataset.md).
