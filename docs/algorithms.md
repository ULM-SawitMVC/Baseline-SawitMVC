# Algorithms — Design and Rationale

This document explains the multi-view deduplication problem and how the top-5
algorithms solve it without any training.

---

## Problem Formulation

Given a tree photographed from N sides (N=4 or N=8), each side produces a set of
bounding-box detections with class labels (B1–B4). The same physical bunch appears
in up to 3 consecutive views (4-side trees) or 6 consecutive views (8-side trees).

**Goal:** predict `{B1: int, B2: int, B3: int, B4: int}` — the unique count of
bunches per maturity class.

**Naive sum:** simply adding up all detections overcounts by a factor of ~1.83.
The algorithms in this repository reduce that error to ≤ 13%.

---

## Three Core Building Blocks

All top-5 algorithms combine variants of three estimators:

### 1. `visibility_count` (used by M01–M04, M06)

Assigns a weight to each detection based on its horizontal position in the frame.

**Intuition:** A bunch at the center of the frame (`x_norm ≈ 0.5`) is photographed
head-on from that camera angle and is likely unique. A bunch at the edge of the frame
is partially visible and may also appear in the next camera's frame.

```
weight = 1 / (1 + exp(-((x_norm - 0.5)² / (2 × 0.3²))))
```

Per-class count = sum of weights (rounded to nearest integer).

### 2. `adaptive_corrected` (used by M01–M05)

Divides the naive count by a class-specific duplication factor that decreases as the
tree gets denser.

```
dup_rate = clip(2.05 - 0.014 × n_total, 1.45, 2.10)
count[c] = round(naive[c] / (BASE_FACTORS[c] × (dup_rate / 1.79)))
```

`BASE_FACTORS` are the median naive/GT ratios derived from a 228-tree development set:
`{B1: 1.986, B2: 1.786, B3: 1.795, B4: 1.655}`.

**Intuition:** Dense trees have a lower per-class duplication rate because bunches are
harder to capture from multiple angles when the canopy is full.

### 3. `max_per_side` (used by M01–M04, M07)

A hard physical floor: the unique count cannot be less than the maximum count seen on
any single side.

```
floor[c] = max(count of class c detections on any single side_index)
```

**Intuition:** If side 2 has 5 B3 bunches visible, there must be at least 5 B3 bunches
on that tree — they can't all be duplicates of each other from the same viewpoint.

---

## Algorithm Families

### Selector (M01, M02)

Routes each tree to a different base estimator depending on its detection profile.
The trifurcation logic in M01/M02 identifies three regimes:

| Condition | Regime | Estimator |
|-----------|--------|-----------|
| `b3_frac ≥ 0.60` AND `n_total ≥ 25` | Dense B3-dominated | `median3_floor` |
| `naive_B1 ≥ 3` AND `b3_frac < 0.45` AND `naive_B4 < 10` | B1-rich, balanced | `adaptive_corrected` |
| Otherwise | Default | `geometric_mean_blend` |

**Why selectors win:** A single global rule cannot be optimal for all tree types.
Trees with heavy canopy, many B3 bunches, and high total detections behave differently
from sparse trees with diverse class distributions.

M01 adds a second stage: after routing, it reallocates B2 and B3 counts while
preserving their sum. The naive B2:B3 ratio is a reliable signal for the split even
when individual class predictions drift.

### Blend (M03, M04, M05)

Combine two estimators (visibility and adaptive) via a fixed formula:

| Algorithm | Formula |
|-----------|---------|
| M03 | `sqrt(visibility × adaptive)` |
| M04 | `round(0.6 × visibility + 0.4 × adaptive)` + max_per_side floor |
| M05 | `round(0.6 × visibility + 0.4 × adaptive)` |

Geometric mean (M03) is more conservative than linear blending and handles outliers
better when one estimator produces a very high or very low value.

---

## Why Heuristics Beat End-to-End ML

On this dataset, the heuristic M01 achieves 87.62% Acc±1 while the best end-to-end
pipeline (YOLO26m → SVM) achieves only 71.6%. The gap is entirely explained by
**detector error propagation**:

- ML counting on perfect (GT) detections achieves 96.1% Acc±1 (SVM on 13-dim features)
- The same SVM on YOLO detections drops to ~71% because misdetected class labels and
  missed detections corrupt the feature vectors

**Implication:** improving the detector (or using a better annotation strategy for
training) is a higher-leverage improvement than improving the counting algorithm.

---

## B2↔B3 Irreducible Ambiguity

The most important insight from this research is that B2 and B3 are **visually
indistinguishable from a single camera angle** at approximately 10–12% of trees (95–118
trees out of 953). This is not label noise — manual audit confirmed 0% labeling errors.

The confusion is inherent: during the transition from B2 to B3 maturity, bunches have
intermediate visual properties that cannot be resolved even by human annotators with
full context. The theoretical ceiling for algorithms without cross-view embeddings
is approximately **89.6% Acc±1** — which is why the 90% target was abandoned as
unreachable under the no-training constraint.

Per-class MAE breakdown for M01 (best method):
- B1 MAE = 0.181 (easy: visually distinct, positioned at bottom)
- B4 MAE = 0.310 (moderate: smallest bunches, sometimes missed)
- B2 MAE = 0.346 (hard: B2↔B3 confusion)
- B3 MAE = 0.757 (hardest: majority class, absorbs all B2↔B3 confusion)

---

## Stability vs Accuracy Trade-off

Some algorithms perform well on small development sets but degrade on the full 953-tree
dataset:

| Algorithm | Acc±1 on 228 trees | Acc±1 on 953 trees | Drop |
|-----------|:-----------------:|:-----------------:|:----:|
| M12_selector_overrides | 97.37% | 85.94% | −11.43 pp |
| M17_selector_regime | 96.05% | 85.62% | −10.43 pp |
| **M05_blend_vis_divide** | ~86% | **86.99%** | **+1 pp** |
| **M01_selector_b2b3** | n/a | **87.62%** | — |

The lesson: narrow override rules that fit a small dev set consistently overfit when
the dataset grows. Simple blend and selector methods generalize better.
