"""
M05_blend_vis_divide
Family: blend  | Old name: hybrid_vis_corr
Canonical benchmark: 953 trees, Acc±1 = 86.99%, MAE = 0.3875.

Weighted average of M06 (visibility) and M19 (adaptive divide):
    out = round(0.6 * vis + 0.4 * adaptive_corrected)

Simplest fallback method. No floor / ceiling clamp.
"""

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


def _visibility(dets, alpha=1.0, sigma=0.3):
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
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = Counter(d["class"] for d in dets)
    return {c: max(0, round(n.get(c, 0) / (BASE_FACTORS[c] * scale))) for c in NAMES}


def predict(detections: list) -> dict:
    vis = _visibility(detections)
    corr = _adaptive_corrected(detections)
    w = 0.6
    return {c: max(0, int(round(w * vis[c] + (1 - w) * corr[c]))) for c in NAMES}
