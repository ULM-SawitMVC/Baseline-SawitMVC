"""
Counting improvement experiments v3 — Extended feature engineering + model selection.

80 configurations tested: 8 feature sets × 5 models × 2 training strategies.

Feature sets (all supersets of 13-dim F0 baseline):
  F0                 13  naive_sum, max_per_side, mean_per_side × 4 classes + n_sides
  F0+conf            33  + per-class conf_sum, conf_mean, conf_max, high_conf≥0.5, vhigh_conf≥0.6
  F0+spatial         21  + per-class mean_cy (vertical centroid), mean_area (bbox size)
  F0+distrib         33  + per-class std_per_side, min_per_side, cv (std/mean),
                            n_sides_detected, consistency=1/(1+std)
  F0+conf+spatial    41  F0 + conf + spatial
  F0+conf+distrib    53  F0 + conf + distrib
  F0+distrib+spatial 37  F0 + distrib + spatial
  F_all              67  all above + total_naive, frac_Bc, b3/(b2+b3)

Models: LR, Ridge (RidgeCV), ElasticNet (MultiTaskElasticNetCV), XGBoost, LightGBM
Training strategies: train (716), train+val (812)

Usage:
  cd /workspace/Baseline-SawitMVC
  python experiments/exp_counting_v3.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import _load_gt, _load_splits, CLASSES

INFERENCE_DIR = ROOT / "predictions" / "y26mv2_per_tree"
GT_DIR        = ROOT / "ground_truth" / "annotations"
OUT_DIR       = ROOT / "results" / "experiments"


def extract_all_features(tree_json: dict) -> dict[str, float]:
    sides   = tree_json.get("images", {})
    n_sides = max(len(sides), 1)
    psc = {c: [] for c in CLASSES}
    cf_ = {c: [] for c in CLASSES}
    ar_ = {c: [] for c in CLASSES}
    cy_ = {c: [] for c in CLASSES}

    for sd in sides.values():
        cnt = {c: 0 for c in CLASSES}
        for ann in sd.get("annotations", []):
            cls  = ann.get("class_name", "")
            if cls not in CLASSES: continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            cf_[cls].append(conf)
            ar_[cls].append(float(bbox[2]) * float(bbox[3]))
            cy_[cls].append(float(bbox[1]))
            cnt[cls] += 1
        for c in CLASSES:
            psc[c].append(cnt[c])

    f: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(psc[c], dtype=float)
        cf = np.array(cf_[c])
        ar = np.array(ar_[c])
        cy = np.array(cy_[c])
        n  = len(cf)
        # F0 baseline
        f[f"naive_sum_{c}"]    = float(ps.sum())
        f[f"max_per_side_{c}"] = float(ps.max())
        f[f"mean_per_side_{c}"]= float(ps.mean())
        # distribution group
        f[f"std_per_side_{c}"] = float(ps.std())
        f[f"min_per_side_{c}"] = float(ps.min())
        f[f"cv_per_side_{c}"]  = float(ps.std() / (ps.mean() + 1e-6))
        f[f"n_sides_det_{c}"]  = float((ps > 0).sum())
        f[f"consistency_{c}"]  = float(1.0 / (1.0 + ps.std()))
        # confidence group
        f[f"conf_sum_{c}"]     = float(cf.sum())
        f[f"conf_mean_{c}"]    = float(cf.mean()) if n > 0 else 0.0
        f[f"conf_max_{c}"]     = float(cf.max())  if n > 0 else 0.0
        f[f"high_conf_{c}"]    = float((cf >= 0.5).sum())
        f[f"vhigh_conf_{c}"]   = float((cf >= 0.6).sum())
        # spatial group
        f[f"mean_cy_{c}"]      = float(cy.mean()) if n > 0 else 0.5
        f[f"mean_area_{c}"]    = float(ar.mean()) if n > 0 else 0.0

    total = sum(f[f"naive_sum_{c}"] for c in CLASSES)
    f["n_sides"]     = float(n_sides)
    f["total_naive"] = float(total)
    for c in CLASSES:
        f[f"frac_{c}"] = f[f"naive_sum_{c}"] / (total + 1e-6)
    f["b3_b23_frac"] = f["naive_sum_B3"] / (f["naive_sum_B2"] + f["naive_sum_B3"] + 1e-6)
    return f


def load_extended_dataset(inference_dir: Path, gt_dir: Path):
    splits_map = _load_splits(gt_dir.parent)
    gt_map     = _load_gt(gt_dir)
    rows, labels, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            d = json.load(f)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map: continue
        rows.append(extract_all_features(d))
        labels.append([gt_map[tid].get(c, 0) for c in CLASSES])
        tree_ids.append(tid)
        tree_splits.append(splits_map.get(tid, d.get("split", "train")))
    df = pd.DataFrame(rows)
    return df, np.array(labels, dtype=float), tree_ids, np.array(tree_splits)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yr = np.clip(np.round(y_pred), 0, None).astype(int)
    yt = y_true.astype(int)
    r: dict = {}
    for j, c in enumerate(CLASSES):
        err = np.abs(yr[:, j] - yt[:, j])
        r[f"acc_{c}"]  = float(np.mean(err <= 1))
        r[f"mae_{c}"]  = float(np.mean(err))
        r[f"bias_{c}"] = float(np.mean(yr[:, j] - yt[:, j]))
    r["macro_acc"] = float(np.mean([r[f"acc_{c}"]  for c in CLASSES]))
    r["macro_mae"] = float(np.mean([r[f"mae_{c}"]  for c in CLASSES]))
    r["joint_acc"] = float(np.mean(
        np.all(np.array([np.abs(yr[:, j] - yt[:, j]) <= 1 for j in range(len(CLASSES))]), axis=0)
    ))
    return r


def main() -> None:
    print("Loading extended features ...")
    df, y, tree_ids, splits = load_extended_dataset(INFERENCE_DIR, GT_DIR)
    all_feat_cols = df.columns.tolist()

    tr = splits == "train"; va = splits == "val"; te = splits == "test"
    y_te = y[te]
    print(f"Train={tr.sum()} | Val={va.sum()} | Test={te.sum()}")

    F0_cols      = ([f"naive_sum_{c}" for c in CLASSES]
                  + [f"max_per_side_{c}" for c in CLASSES]
                  + [f"mean_per_side_{c}" for c in CLASSES]
                  + ["n_sides"])
    CONF_cols    = ([f"conf_sum_{c}" for c in CLASSES] + [f"conf_mean_{c}" for c in CLASSES]
                  + [f"conf_max_{c}" for c in CLASSES] + [f"high_conf_{c}" for c in CLASSES]
                  + [f"vhigh_conf_{c}" for c in CLASSES])
    SPATIAL_cols = ([f"mean_cy_{c}" for c in CLASSES] + [f"mean_area_{c}" for c in CLASSES])
    DISTRIB_cols = ([f"std_per_side_{c}" for c in CLASSES] + [f"min_per_side_{c}" for c in CLASSES]
                  + [f"cv_per_side_{c}" for c in CLASSES] + [f"n_sides_det_{c}" for c in CLASSES]
                  + [f"consistency_{c}" for c in CLASSES])

    FEATURE_SETS = {
        "F0":               F0_cols,
        "F0+conf":          F0_cols + CONF_cols,
        "F0+spatial":       F0_cols + SPATIAL_cols,
        "F0+distrib":       F0_cols + DISTRIB_cols,
        "F0+conf+spatial":  F0_cols + CONF_cols + SPATIAL_cols,
        "F0+conf+distrib":  F0_cols + CONF_cols + DISTRIB_cols,
        "F0+distrib+spatial": F0_cols + DISTRIB_cols + SPATIAL_cols,
        "F_all":            all_feat_cols,
    }
    MODELS = {
        "LR":        lambda: Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]),
        "Ridge":     lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]),
        "ElasticNet":lambda: Pipeline([("sc", StandardScaler()), ("en", MultiTaskElasticNetCV(cv=5, random_state=42, max_iter=3000))]),
        "XGB":       lambda: MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, random_state=42, verbosity=0), n_jobs=-1),
        "LGB":       lambda: MultiOutputRegressor(LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                                num_leaves=31, subsample=0.8, random_state=42, verbose=-1), n_jobs=-1),
    }
    STRATEGIES = {"train": tr, "train+val": tr | va}

    results = []
    print(f"\n{'Feature Set':<25} {'Model':<12} {'Strategy':<10} {'Dims':>4} {'Macro':>7} {'Joint':>7} {'MAE':>7}")
    print("-" * 78)
    for fs_name, fcols in FEATURE_SETS.items():
        X_all = df[fcols].values.astype(float)
        X_te  = X_all[te]
        for strat_name, train_mask in STRATEGIES.items():
            X_tr = X_all[train_mask]; y_tr = y[train_mask]
            for mdl_name, mdl_fn in MODELS.items():
                try:
                    mdl = mdl_fn(); mdl.fit(X_tr, y_tr)
                    m = score(y_te, mdl.predict(X_te))
                    results.append({"features": fs_name, "model": mdl_name,
                                    "strategy": strat_name, "n_dim": len(fcols), **m})
                    print(f"{fs_name:<25} {mdl_name:<12} {strat_name:<10} {len(fcols):>4} "
                          f"{m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")
                except Exception as e:
                    print(f"FAIL {fs_name} {mdl_name} {strat_name}: {e}")

    rdf = pd.DataFrame(results).sort_values("macro_acc", ascending=False)
    print(f"\n{'='*78}\nTOP 15 OVERALL\n{'='*78}")
    print(rdf.head(15).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(OUT_DIR / "counting_v3_results.csv", index=False)
    print(f"\nSaved → {OUT_DIR / 'counting_v3_results.csv'}")


if __name__ == "__main__":
    main()
