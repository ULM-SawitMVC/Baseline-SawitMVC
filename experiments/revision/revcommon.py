"""
Shared helpers for the ICIC 2026 revision analyses.

Keeps the feature bank, the model zoo, and the scoring rules identical to
experiments/exp_counting_controlled.py, and adds the extra estimators the
reviewers asked for: per-class RMSE, total-count error, bootstrap confidence
intervals, and paired significance tests.
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import CLASSES, _load_gt, _load_splits  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

GT_DIR = ROOT / "ground_truth" / "annotations"
PRED_DIR = ROOT / "predictions" / "y26mv2_per_tree"
ARCHIVE_PRED = ROOT / "archive" / "predictions"
OUT_DIR = ROOT / "results" / "revision"


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def extract_all_features(tree_json: dict, conf_min: float = 0.0) -> dict[str, float]:
    """67-dim feature bank; identical to exp_counting_controlled.extract_all_features.

    conf_min > 0 drops detections below that confidence before aggregation,
    which is how the detector operating-point sweep is produced.
    """
    sides = tree_json.get("images", {})
    n_sides = max(len(sides), 1)
    per_side_counts = {c: [] for c in CLASSES}
    confidences = {c: [] for c in CLASSES}
    areas = {c: [] for c in CLASSES}
    center_y = {c: [] for c in CLASSES}

    for side_data in sides.values():
        counts = {c: 0 for c in CLASSES}
        for ann in side_data.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls not in CLASSES:
                continue
            conf = float(ann.get("conf", 1.0))
            if conf < conf_min:
                continue
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            confidences[cls].append(conf)
            areas[cls].append(float(bbox[2]) * float(bbox[3]))
            center_y[cls].append(float(bbox[1]))
            counts[cls] += 1
        for c in CLASSES:
            per_side_counts[c].append(counts[c])

    features: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(per_side_counts[c], dtype=float)
        conf = np.array(confidences[c])
        area = np.array(areas[c])
        cy = np.array(center_y[c])
        n = len(conf)

        features[f"naive_sum_{c}"] = float(ps.sum())
        features[f"max_per_side_{c}"] = float(ps.max())
        features[f"mean_per_side_{c}"] = float(ps.mean())

        features[f"std_per_side_{c}"] = float(ps.std())
        features[f"min_per_side_{c}"] = float(ps.min())
        features[f"cv_per_side_{c}"] = float(ps.std() / (ps.mean() + 1e-6))
        features[f"n_sides_det_{c}"] = float((ps > 0).sum())
        features[f"consistency_{c}"] = float(1.0 / (1.0 + ps.std()))

        features[f"conf_sum_{c}"] = float(conf.sum())
        features[f"conf_mean_{c}"] = float(conf.mean()) if n > 0 else 0.0
        features[f"conf_max_{c}"] = float(conf.max()) if n > 0 else 0.0
        features[f"high_conf_{c}"] = float((conf >= 0.5).sum())
        features[f"vhigh_conf_{c}"] = float((conf >= 0.6).sum())

        features[f"mean_cy_{c}"] = float(cy.mean()) if n > 0 else 0.5
        features[f"mean_area_{c}"] = float(area.mean()) if n > 0 else 0.0

    total = sum(features[f"naive_sum_{c}"] for c in CLASSES)
    features["n_sides"] = float(n_sides)
    features["total_naive"] = float(total)
    for c in CLASSES:
        features[f"frac_{c}"] = features[f"naive_sum_{c}"] / (total + 1e-6)
    features["b3_b23_frac"] = (
        features["naive_sum_B3"]
        / (features["naive_sum_B2"] + features["naive_sum_B3"] + 1e-6)
    )
    return features


def load_dataset(inference_dir: Path, gt_dir: Path = GT_DIR, conf_min: float = 0.0):
    """-> (features DataFrame, y (n,4), tree_ids, splits, varieties)."""
    splits_map = _load_splits(gt_dir.parent)
    gt_map = _load_gt(gt_dir)
    manifest = pd.read_csv(gt_dir.parent / "split_manifest.csv", encoding="utf-8-sig")
    variety_map = dict(zip(manifest["tree_id"], manifest["variety"]))

    rows, labels, tree_ids, tree_splits, varieties = [], [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            data = json.load(f)
        tree_id = data.get("tree_name") or data.get("tree_id") or fp.stem
        if tree_id not in gt_map:
            continue
        rows.append(extract_all_features(data, conf_min=conf_min))
        labels.append([gt_map[tree_id].get(c, 0) for c in CLASSES])
        tree_ids.append(tree_id)
        tree_splits.append(splits_map.get(tree_id, data.get("split", "train")))
        varieties.append(variety_map.get(tree_id, "UNKNOWN"))
    return (
        pd.DataFrame(rows),
        np.array(labels, dtype=float),
        tree_ids,
        np.array(tree_splits),
        np.array(varieties),
    )


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    f0 = (
        [f"naive_sum_{c}" for c in CLASSES]
        + [f"max_per_side_{c}" for c in CLASSES]
        + [f"mean_per_side_{c}" for c in CLASSES]
        + ["n_sides"]
    )
    conf = (
        [f"conf_sum_{c}" for c in CLASSES]
        + [f"conf_mean_{c}" for c in CLASSES]
        + [f"conf_max_{c}" for c in CLASSES]
        + [f"high_conf_{c}" for c in CLASSES]
        + [f"vhigh_conf_{c}" for c in CLASSES]
    )
    spatial = [f"mean_cy_{c}" for c in CLASSES] + [f"mean_area_{c}" for c in CLASSES]
    distrib = (
        [f"std_per_side_{c}" for c in CLASSES]
        + [f"min_per_side_{c}" for c in CLASSES]
        + [f"cv_per_side_{c}" for c in CLASSES]
        + [f"n_sides_det_{c}" for c in CLASSES]
        + [f"consistency_{c}" for c in CLASSES]
    )
    return {
        "F0": f0,
        "F0+conf": f0 + conf,
        "F0+spatial": f0 + spatial,
        "F0+distrib": f0 + distrib,
        "F0+conf+spatial": f0 + conf + spatial,
        "F0+conf+distrib": f0 + conf + distrib,
        "F0+distrib+spatial": f0 + distrib + spatial,
        "F_all": df.columns.tolist(),
    }


def model_builders() -> dict[str, Callable[[], object]]:
    return {
        "LR": lambda: Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "SVM": lambda: GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("svr", MultiOutputRegressor(SVR(kernel="rbf")))]),
            {"svr__estimator__C": [0.1, 1, 10],
             "svr__estimator__gamma": ["scale", 0.01, 0.1]},
            cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
        ),
        "RF": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        "Ridge": lambda: Pipeline([("scaler", StandardScaler()),
                                   ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]),
        "ElasticNet": lambda: Pipeline([("scaler", StandardScaler()),
                                        ("elasticnet", MultiTaskElasticNetCV(
                                            cv=5, random_state=SEED, max_iter=3000))]),
    }


def round_counts(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y_pred), 0, None).astype(int)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def per_tree_stats(y_true: np.ndarray, y_pred_rounded: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tree quantities that every reported metric averages over."""
    truth = y_true.astype(int)
    err = y_pred_rounded - truth               # signed, (n,4)
    abserr = np.abs(err)
    ok = (abserr <= 1).astype(float)           # (n,4)
    tot_err = err.sum(axis=1)                  # (n,)
    return {
        "err": err.astype(float),
        "abserr": abserr.astype(float),
        "sqerr": (err.astype(float) ** 2),
        "ok": ok,
        "tree_ok": (ok.sum(axis=1) == 4).astype(float),
        "exact": (abserr.sum(axis=1) == 0).astype(float),
        "tot_abserr": np.abs(tot_err).astype(float),
        "tot_sqerr": (tot_err.astype(float) ** 2),
        "tot_err": tot_err.astype(float),
        "tot_ok": (np.abs(tot_err) <= 1).astype(float),
    }


def metrics_from_stats(st: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    m: dict[str, float] = {}
    for j, c in enumerate(CLASSES):
        m[f"acc_{c}"] = float(sel(st["ok"])[:, j].mean())
        m[f"mae_{c}"] = float(sel(st["abserr"])[:, j].mean())
        m[f"rmse_{c}"] = float(np.sqrt(sel(st["sqerr"])[:, j].mean()))
        m[f"bias_{c}"] = float(sel(st["err"])[:, j].mean())
    m["macro"] = float(np.mean([m[f"acc_{c}"] for c in CLASSES]))
    m["mae"] = float(np.mean([m[f"mae_{c}"] for c in CLASSES]))
    m["rmse"] = float(np.mean([m[f"rmse_{c}"] for c in CLASSES]))
    m["joint"] = float(sel(st["tree_ok"]).mean())
    m["exact"] = float(sel(st["exact"]).mean())
    m["total_mae"] = float(sel(st["tot_abserr"]).mean())
    m["total_rmse"] = float(np.sqrt(sel(st["tot_sqerr"]).mean()))
    m["total_bias"] = float(sel(st["tot_err"]).mean())
    m["total_acc"] = float(sel(st["tot_ok"]).mean())
    return m


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return metrics_from_stats(per_tree_stats(y_true, round_counts(y_pred)))


def bootstrap_indices(n: int, n_boot: int = 10000, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _boot_draws(st: dict[str, np.ndarray], keys: list[str], boot_idx: np.ndarray) -> dict[str, np.ndarray]:
    n_boot = boot_idx.shape[0]
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        mb = metrics_from_stats(st, boot_idx[b])
        for k in keys:
            draws[k][b] = mb[k]
    return draws


def bootstrap_ci(st: dict[str, np.ndarray], keys: list[str],
                 n_boot: int = 10000, seed: int = SEED) -> dict[str, dict[str, float]]:
    """Percentile 95% CI for each metric key."""
    n = st["ok"].shape[0]
    boot_idx = bootstrap_indices(n, n_boot, seed)
    point = metrics_from_stats(st)
    draws = _boot_draws(st, keys, boot_idx)
    return {
        k: {
            "point": point[k],
            "ci_lo": float(np.percentile(draws[k], 2.5)),
            "ci_hi": float(np.percentile(draws[k], 97.5)),
            "se": float(draws[k].std(ddof=1)),
        }
        for k in keys
    }


def paired_bootstrap(st_a: dict[str, np.ndarray], st_b: dict[str, np.ndarray],
                     keys: list[str], n_boot: int = 10000,
                     seed: int = SEED) -> dict[str, dict[str, float]]:
    """Paired bootstrap on metric differences (A - B); same trees resampled for both."""
    n = st_a["ok"].shape[0]
    assert st_b["ok"].shape[0] == n
    boot_idx = bootstrap_indices(n, n_boot, seed)
    pa, pb = metrics_from_stats(st_a), metrics_from_stats(st_b)
    da = _boot_draws(st_a, keys, boot_idx)
    db = _boot_draws(st_b, keys, boot_idx)
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        d = da[k] - db[k]
        p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
        out[k] = {
            "diff": pa[k] - pb[k],
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_boot": float(min(1.0, p)),
        }
    return out


def wilcoxon_macro_mae(st_a: dict[str, np.ndarray], st_b: dict[str, np.ndarray]) -> dict[str, float]:
    from scipy.stats import wilcoxon
    a = st_a["abserr"].mean(axis=1)
    b = st_b["abserr"].mean(axis=1)
    d = a - b
    if np.allclose(d, 0):
        return {"stat": float("nan"), "p": 1.0, "n_nonzero": 0}
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue),
            "n_nonzero": int((d != 0).sum())}


def mcnemar_tree(st_a: dict[str, np.ndarray], st_b: dict[str, np.ndarray]) -> dict[str, float]:
    """Exact McNemar on the Tree +/-1 indicator."""
    from scipy.stats import binomtest
    a = st_a["tree_ok"].astype(bool)
    b = st_b["tree_ok"].astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"n10": n10, "n01": n01, "p": 1.0}
    p = binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue
    return {"n10": n10, "n01": n01, "p": float(p)}


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")
