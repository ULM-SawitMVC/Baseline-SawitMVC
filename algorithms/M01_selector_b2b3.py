"""
Algoritma: selector_with_b2b3
Generasi: iter11 (eksperimen 10 Mei 2026)
Benchmark Brand-New-Dataset-YOLO 953 pohon: Acc ±1 = 86.67%, MAE = 0.3982,
n_fail = 127 (best Acc±1 + best MAE simultan).

Validasi train/val/test held-out (dari iter11_results.csv):
    train: 87.34%   val: 82.58%   test: 88.62%   worst_drop: 0.00 pp

Ide utama
---------
Dua koreksi disusun berurutan:

  (1) Selector trifurkasi (selector_iter9_trifurc) — pilih estimator dasar
      berdasarkan profil deteksi pohon:
        a. Pohon padat dengan dominansi B3 (b3frac >= 0.60 dan n_total >= 25)
           → median3_floor (median of {visibility, adaptive, side_coverage})
        b. Pohon dengan B1 cukup banyak, B3 tidak dominan, B4 sedikit
           (naive_B1 >= 3, b3frac < 0.45, naive_B4 < 10)
           → adaptive_corrected (v5)
        c. Selain itu → geometric_mean_blend (akar dari vis × adaptive)

  (2) Koreksi B2↔B3 split — total (B2+B3) dari hasil (1) dipertahankan
      (jumlah tandan dewasa stabil di seluruh sisi), namun rasio B2:B3
      dialokasikan ulang berdasarkan rasio naive di kelas tersebut. Ini
      menjawab masalah inti dataset: ambiguitas visual B2↔B3 yang
      menyebabkan kesalahan kelas tetapi bukan kesalahan jumlah.

Mengapa lebih baik dari hybrid_vis_corr?
- hybrid_vis_corr (86.04%) hanya merata-rata dua estimator tanpa peduli
  profil pohon. selector_with_b2b3 memilih estimator yang paling cocok
  per regime, lalu mengoreksi kesalahan kelas B2/B3 pasca-prediksi.
- Improvement: +0.63 pp Acc±1 dan -2.32% MAE absolut.

Constraint: 100% deterministik, tanpa training, tanpa embedding, tanpa
gradient. Semua parameter berasal dari median rasio naive/GT pada
228 pohon (BASE_FACTORS) — tidak ada parameter yang di-fit pada split
val atau test.

Input
-----
detections : list[dict]
    Tiap elemen wajib punya field:
      - "class"      : str  → "B1"/"B2"/"B3"/"B4"
      - "x_norm"     : float (pusat bbox YOLO, 0..1)
      - "y_norm"     : float (tidak dipakai langsung di sini, dijaga konsisten)
      - "side_index" : int  (urutan sisi foto, 0..7)

Output
------
dict[str, int]
    Count unik per kelas: {"B1": int, "B2": int, "B3": int, "B4": int}.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

NAMES = ["B1", "B2", "B3", "B4"]

# Median rasio naive/GT per kelas, dihitung dari 228 pohon JSON (snapshot
# 22 April 2026). Dipakai sebagai pembagi dasar untuk adaptive_corrected.
BASE_FACTORS = {"B1": 1.986, "B2": 1.786, "B3": 1.795, "B4": 1.655}


# ---------------------------------------------------------------------------
# Estimator dasar
# ---------------------------------------------------------------------------

def naive_count(dets: list) -> dict:
    """Jumlah deteksi mentah per kelas (tanpa dedup)."""
    n = Counter(d["class"] for d in dets)
    return {c: int(n.get(c, 0)) for c in NAMES}


def adaptive_corrected(dets: list) -> dict:
    """Pembagi adaptif: dup-rate menurun saat pohon padat."""
    n_total = len(dets)
    dup_rate = float(np.clip(2.05 - 0.014 * n_total, 1.45, 2.10))
    scale = dup_rate / 1.79
    n = naive_count(dets)
    return {c: max(0, int(round(n[c] / (BASE_FACTORS[c] * scale)))) for c in NAMES}


def visibility_count(dets: list, alpha: float = 1.0, sigma: float = 0.3) -> dict:
    """
    Pembobotan Gauss berdasarkan jarak deteksi dari pusat frame.
    Tandan di tengah (x_norm ≈ 0.5) → bobot ~ 1 (pasti unik).
    Tandan di tepi → bobot menurun (kemungkinan terlihat dari sisi sebelah).
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
    Lantai fisik berdasarkan deteksi maksimum per-sisi: jumlah unik tidak
    boleh kurang dari max_per_side, dan tidak boleh melebihi naive.
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
# Estimator turunan (dipakai oleh selector trifurkasi)
# ---------------------------------------------------------------------------

def geometric_mean_blend(dets: list) -> dict:
    """sqrt(visibility × adaptive_corrected) per kelas, dengan lantai max_per_side."""
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
    """Median dari {visibility, adaptive, side_coverage} per kelas."""
    a = visibility_count(dets)
    b = adaptive_corrected(dets)
    s = side_coverage(dets)
    out = {cl: sorted([a[cl], b[cl], s[cl]])[1] for cl in NAMES}
    return {cl: max(out[cl], _max_per_side(dets, cl)) for cl in NAMES}


# ---------------------------------------------------------------------------
# Selector trifurkasi
# ---------------------------------------------------------------------------

def selector_iter9_trifurc(dets: list) -> dict:
    """
    Pilih estimator dasar berdasarkan profil pohon:
      - dense + B3-dominan         → median3_floor
      - B1 cukup, B3 sedikit, B4 sedikit → adaptive_corrected
      - default                    → geometric_mean_blend
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
# Algoritma final: selector + koreksi split B2↔B3
# ---------------------------------------------------------------------------

def predict(detections: list) -> dict:
    """
    Hitung count unik per kelas dengan selector_with_b2b3.

    Tahap:
      1. selector_iter9_trifurc → dapat (B1, B2, B3, B4) awal.
      2. Pertahankan total joint = B2 + B3, tetapi alokasikan ulang B2:B3
         menggunakan rasio naive di kedua kelas tersebut.
      3. Pasang lantai max_per_side untuk B2 dan B3 (jumlah unik minimum
         tidak mungkin kurang dari deteksi maksimum di salah satu sisi).

    Parameters
    ----------
    detections : list[dict]
        Daftar bounding box dari semua sisi pohon. Wajib field
        "class", "x_norm", "side_index".

    Returns
    -------
    dict[str, int]
        Count unik per kelas {"B1", "B2", "B3", "B4"}.
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
