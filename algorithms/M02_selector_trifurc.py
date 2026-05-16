"""
Algorithm: selector_iter9_trifurc.

Three-way selector that chooses a base estimator from the tree detection
profile:
  1. Dense, B3-dominated trees (b3frac >= 0.60, n_total >= 25): median3_floor
  2. B1-rich trees with limited B3/B4 dominance
     (naive_B1 >= 3, b3frac < 0.45, naive_B4 < 10): adaptive_corrected
  3. All other trees: geometric_mean_blend

Canonical benchmark: 953 trees, Acc±1 = 87.62%, MAE = 0.3757,
n_fail = 118. See benchmarks/results/accuracy_953.csv.
"""

from collections import Counter
import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]


def _max_per_side(dets, c):
    cd = [d for d in dets if d["class"] == c]
    return max(Counter(d["side_index"] for d in cd).values()) if cd else 0


def _visibility_count(dets, alpha=1.0, sigma=0.3):
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


def _adaptive_corrected(dets):
    n_total = len(dets)
    base = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = Counter(d["class"] for d in dets)
    return {c: max(0, round(n.get(c, 0) / (base[c] * scale))) for c in NAMES}


def _side_coverage(dets):
    vis = _visibility_count(dets)
    from collections import Counter
    n = Counter(d["class"] for d in dets)
    out = {}
    for c in NAMES:
        cd = [d for d in dets if d["class"] == c]
        if not cd:
            out[c] = 0
            continue
        max_per_side = max(Counter(d["side_index"] for d in cd).values())
        out[c] = min(max(vis[c], max_per_side), n.get(c, 0))
    return out


def _geometric_mean_blend(dets):
    v = _visibility_count(dets)
    c = _adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def _median3_floor(dets):
    a = _visibility_count(dets)
    b = _adaptive_corrected(dets)
    s = _side_coverage(dets)
    out = {cl: sorted([a[cl], b[cl], s[cl]])[1] for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


def predict(detections: list) -> dict:
    """
    predictions = selector_iter9_trifurc(detections)
    
    Returns dict {"B1": int, "B2": int, "B3": int, "B4": int}
    """
    dets = detections
    n_total = len(dets)
    if n_total == 0:
        return _geometric_mean_blend(dets)
    naive = Counter(d["class"] for d in dets)
    b3frac = naive["B3"] / n_total
    if b3frac >= 0.60 and n_total >= 25:
        return _median3_floor(dets)
    if naive["B1"] >= 3 and b3frac < 0.45 and naive["B4"] < 10:
        return _adaptive_corrected(dets)
    return _geometric_mean_blend(dets)
