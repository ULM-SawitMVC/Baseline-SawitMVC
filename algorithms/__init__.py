"""
SawitMVC Baseline — Deduplication Algorithms
=============================================

Top-5 heuristic algorithms for multi-view oil palm bunch deduplication.
All methods are deterministic, parameter-free, and require no training.

Benchmark: 953 trees, Brand-New-Dataset-YOLO (post-GT-fix 2026-05-16).

Usage
-----
    from algorithms import RANKING, get_algorithm
    from algorithms.M01_selector_b2b3 import predict

    result = predict(detections)  # {"B1": int, "B2": int, "B3": int, "B4": int}

Detection format
----------------
Each detection is a dict with keys:
    - "class"      : str  — one of "B1", "B2", "B3", "B4"
    - "x_norm"     : float — normalized x-center of bounding box (0.0–1.0)
    - "y_norm"     : float — normalized y-center of bounding box (0.0–1.0)
    - "side_index" : int   — camera side index (0-based)
"""

from .M01_selector_b2b3 import predict as predict_M01
from .M02_selector_trifurc import predict as predict_M02
from .M03_blend_geometric import predict as predict_M03
from .M04_blend_floor_clamped import predict as predict_M04
from .M05_blend_vis_divide import predict as predict_M05

RANKING = {
    "M01_selector_b2b3": {
        "rank": 1,
        "acc1": 0.8762,
        "macro_mae": 0.3746,
        "total_count_mae": 1.3305,
        "n_fail": 118,
        "predict": predict_M01,
        "description": (
            "Champion. Trifurcation selector routes each tree to the best "
            "estimator (median3_floor / adaptive / geometric_mean_blend), "
            "then reallocates the B2+B3 total using naive B2:B3 ratio to "
            "correct visual ambiguity between those two classes."
        ),
    },
    "M02_selector_trifurc": {
        "rank": 2,
        "acc1": 0.8762,
        "macro_mae": 0.3757,
        "total_count_mae": 1.3305,
        "n_fail": 118,
        "predict": predict_M02,
        "description": (
            "Runner-up. Same trifurcation logic as M01 but without the "
            "B2↔B3 split correction. Identical Acc±1 to M01 but slightly "
            "higher MAE."
        ),
    },
    "M03_blend_geometric": {
        "rank": 3,
        "acc1": 0.8699,
        "macro_mae": 0.3767,
        "total_count_mae": 1.3410,
        "n_fail": 124,
        "predict": predict_M03,
        "description": (
            "Geometric mean of visibility_count and adaptive_corrected, "
            "with a per-class floor at max_per_side. Simple, stable, and "
            "generalizes well across tree types."
        ),
    },
    "M04_blend_floor_clamped": {
        "rank": 4,
        "acc1": 0.8699,
        "macro_mae": 0.3848,
        "total_count_mae": 1.3421,
        "n_fail": 124,
        "predict": predict_M04,
        "description": (
            "Weighted blend (0.6 × visibility + 0.4 × adaptive_corrected) "
            "with a per-class floor enforced at max_per_side. One line of "
            "modification from the basic blend."
        ),
    },
    "M05_blend_vis_divide": {
        "rank": 5,
        "acc1": 0.8699,
        "macro_mae": 0.3875,
        "total_count_mae": 1.3463,
        "n_fail": 124,
        "predict": predict_M05,
        "description": (
            "Simplest fallback: weighted blend (0.6 × visibility + "
            "0.4 × adaptive_corrected) with no floor clamping. "
            "Recommended as a baseline for new dataset domains."
        ),
    },
}

ALGORITHMS = list(RANKING.keys())
BEST = "M01_selector_b2b3"


def get_algorithm(name: str):
    """Return the predict() function for the given algorithm name.

    Args:
        name: Algorithm name, e.g. "M01_selector_b2b3".

    Returns:
        Callable predict(detections) -> dict.

    Raises:
        KeyError: If the algorithm name is not found in RANKING.
    """
    if name not in RANKING:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {ALGORITHMS}"
        )
    return RANKING[name]["predict"]
