"""
Algoritma: geometric_mean_blend
Generasi: iter4–iter9 (eksperimen 10 Mei 2026)

Estimator dasar yang mengambil rata-rata geometri antara
visibility_count dan adaptive_corrected, dengan floor per kelas
berdasarkan max_per_side.

Benchmark 953 pohon: Acc ±1 = 86.15%, MAE = 0.3961, n_fail = 132.
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
    from collections import Counter
    n = Counter(d["class"] for d in dets)
    return {c: max(0, round(n.get(c, 0) / (base[c] * scale))) for c in NAMES}


def predict(detections: list) -> dict:
    """
    predictions = geometric_mean_blend(detections)
    
    Returns dict {"B1": int, "B2": int, "B3": int, "B4": int}
    """
    dets = detections
    v = _visibility_count(dets)
    c = _adaptive_corrected(dets)
    out = {}
    for cl in NAMES:
        if v[cl] == 0 or c[cl] == 0:
            out[cl] = (v[cl] + c[cl]) // 2
        else:
            out[cl] = int(round(np.sqrt(v[cl] * c[cl])))
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}
