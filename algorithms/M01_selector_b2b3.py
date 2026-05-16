"""
Algorithm: selector_with_b2b3.

Canonical benchmark: 953 trees, Acc±1 = 87.62%, MAE = 0.3746,
n_fail = 118. See benchmarks/results/accuracy_953.csv.

The algorithm applies two corrections in sequence:

  1. A three-way selector chooses the base estimator from the tree detection
     profile:
       a. Dense, B3-dominated trees (b3frac >= 0.60 and n_total >= 25)
          use median3_floor.
       b. B1-rich trees with limited B3/B4 dominance
          (naive_B1 >= 3, b3frac < 0.45, naive_B4 < 10)
          use adaptive_corrected.
       c. All other trees use geometric_mean_blend.

  2. B2/B3 split correction preserves the predicted B2+B3 total, then
     reallocates that total using the naive B2:B3 ratio. This targets the main
     dataset ambiguity: B2 and B3 visual confusion changes class labels more
     often than total bunch count.

This is fully deterministic: no training, embeddings, or gradient optimization.
The BASE_FACTORS constants come from median naive/GT ratios on a 228-tree
development snapshot; no parameters are fitted on validation or test splits.

Input detections must contain:
  - "class": "B1", "B2", "B3", or "B4"
  - "x_norm": normalized YOLO box center x, 0..1
  - "y_norm": normalized YOLO box center y, retained for interface symmetry
  - "side_index": image side index, 0..7

Returns a unique per-class count dict: {"B1": int, "B2": int, "B3": int, "B4": int}.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

# Median naive/GT ratio per class, computed from a 228-tree JSON snapshot
# from 2026-04-22. Used as the base divisor for adaptive_corrected.
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


# ---------------------------------------------------------------------------
# Base estimators
# ---------------------------------------------------------------------------

def naive_count(dets: list) -> dict:
    """Raw per-class detection count before deduplication."""
    n = Counter(d["class"] for d in dets)
    return {c: int(n.get(c, 0)) for c in NAMES}


def adaptive_corrected(dets: list) -> dict:
    """Adaptive divisor: the duplicate rate decreases as trees get denser."""
    n_total = len(dets)
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = naive_count(dets)
    return {c: max(0, int(round(n[c] / (BASE_FACTORS[c] * scale)))) for c in NAMES}


def visibility_count(dets: list, alpha: float = 1.0, sigma: float = 0.3) -> dict:
    """
    Gaussian-like weighting by horizontal distance from the frame center.
    Center detections (x_norm ~= 0.5) get higher unique-count weight; edge
    detections get lower weight because they may also appear in adjacent views.
    """
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        total = sum(
            1.0 / (1.0 + alpha * np.exp(-((d["x_norm"] - 0.5) ** 2) / (2.0 * sigma ** 2)))
            for d in cd
        )
        out[c] = max(0, int(round(total)))
    return out


def side_coverage(dets: list) -> dict:
    """
    Physical floor from the maximum count on any single side: the unique count
    cannot be below max_per_side and cannot exceed the naive count.
    """
    vis = visibility_count(dets)
    n = naive_count(dets)
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        max_per_side = max(Counter(d["side_index"] for d in cd).values())
        out[c] = min(max(vis[c], max_per_side), n[c])
    return out


def _max_per_side(dets: list, c: str) -> int:
    cd = [d for d in dets if d["class"] == c]
    return max(Counter(d["side_index"] for d in cd).values()) if cd else 0


# ---------------------------------------------------------------------------
# Derived estimators used by the three-way selector
# ---------------------------------------------------------------------------

def geometric_mean_blend(dets: list) -> dict:
    """sqrt(visibility * adaptive_corrected) per class, with max_per_side floor."""
    v = visibility_count(dets)
    c = adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def median3_floor(dets: list) -> dict:
    """Per-class median of {visibility, adaptive, side_coverage}."""
    a = visibility_count(dets)
    b = adaptive_corrected(dets)
    s = side_coverage(dets)
    out = {cl: sorted([a[cl], b[cl], s[cl]])[1] for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


# ---------------------------------------------------------------------------
# Three-way selector
# ---------------------------------------------------------------------------

def selector_iter9_trifurc(dets: list) -> dict:
    """
    Choose a base estimator from the tree detection profile:
      - dense and B3-dominated: median3_floor
      - B1-rich with limited B3/B4 dominance: adaptive_corrected
      - default: geometric_mean_blend
    """
    n_total = len(dets)
    if n_total == 0:
        return geometric_mean_blend(dets)

    naive = naive_count(dets)
    b3frac = naive["B3"] / n_total

    if b3frac >= 0.60 and n_total >= 25:
        return median3_floor(dets)
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return adaptive_corrected(dets)
    return geometric_mean_blend(dets)


# ---------------------------------------------------------------------------
# Final algorithm: selector + B2/B3 split correction
# ---------------------------------------------------------------------------

def predict(detections: list) -> dict:
    """
    Estimate unique per-class counts with selector_with_b2b3.

    Steps:
      1. Run selector_iter9_trifurc for an initial B1/B2/B3/B4 prediction.
      2. Preserve joint total B2+B3, then reallocate B2:B3 using the naive
         detection ratio between those classes.
      3. Apply max_per_side floors for B2 and B3.

    Parameters
    ----------
    detections : list[dict]
        Bounding boxes from all tree sides. Required fields:
        "class", "x_norm", and "side_index".

    Returns
    -------
    dict[str, int]
        Unique per-class count for {"B1", "B2", "B3", "B4"}.
    """
    pred = selector_iter9_trifurc(detections)

    joint = pred["B2"] + pred["B3"]
    if joint == 0:
        return pred

    b23 = [d for d in detections if d["class"] in ("B2", "B3")]
    if not b23:
        return pred

    n_b3 = sum(1 for d in b23 if d["class"] == "B3")
    n_b2 = sum(1 for d in b23 if d["class"] == "B2")
    if n_b2 + n_b3 == 0:
        return pred

    frac_b3 = n_b3 / (n_b2 + n_b3)
    new_b3 = int(round(joint * frac_b3))
    new_b2 = joint - new_b3

    out = dict(pred)
    out["B2"] = max(new_b2, _max_per_side(detections, "B2"))
    out["B3"] = max(new_b3, _max_per_side(detections, "B3"))
    return out
