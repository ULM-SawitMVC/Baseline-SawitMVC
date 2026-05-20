"""
Counting improvement experiments — Feature engineering + model selection.

Feature sets tested:
  F0  Baseline 13-dim (naive_sum, max_per_side, mean_per_side, n_sides)
  F1  F0 + confidence features (conf-weighted sum, mean conf, high-conf count,
      n_sides_detected per class)                          [+16 dims → 29]
  F2  F0 + structural features (per-side std, min_per_side, bbox area mean,
      detection density per class)                         [+16 dims → 29]
  F3  F0 + F1 + F2 full feature set                       [+32 dims → 45]

Models tested:
  LR   Linear Regression + StandardScaler          (current baseline)
  RID  Ridge Regression  + StandardScaler + alpha CV
  GBR  GradientBoostingRegressor (MultiOutput)
  XGB  XGBRegressor (MultiOutput)
  LGB  LGBMRegressor (MultiOutput)

Usage:
  cd /workspace/Baseline-SawitMVC
  python experiments/exp_counting_v2.py
  python experiments/exp_counting_v2.py --save-best
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import load_dataset, CLASSES

INFERENCE_DIR = ROOT / "predictions" / "y26mv2_per_tree"
GT_DIR        = ROOT / "ground_truth" / "annotations"
OUT_DIR       = ROOT / "results" / "experiments"

# ─── Extended feature extraction ─────────────────────────────────────────────

def extract_extended_features(tree_json: dict) -> dict[str, np.ndarray]:
    """Return a dict of feature groups from one prediction JSON.

    Groups:
      baseline   — 13 dims (matching current pipeline)
      confidence — 16 dims (conf-weighted sum, mean conf, high-conf count,
                              n_sides_detected)
      structural — 16 dims (per_side_std, min_per_side, bbox_area_mean,
                              detection_density)
    """
    sides  = tree_json.get("images", {})
    n_sides = max(len(sides), 1)

    # per-side per-class accumulation
    per_side_counts: dict[str, list[int]]   = {c: [] for c in CLASSES}
    per_side_confs:  dict[str, list[float]] = {c: [] for c in CLASSES}
    all_confs:       dict[str, list[float]] = {c: [] for c in CLASSES}
    all_areas:       dict[str, list[float]] = {c: [] for c in CLASSES}
    n_sides_detected: dict[str, int]        = {c: 0  for c in CLASSES}
    high_conf_count:  dict[str, int]        = {c: 0  for c in CLASSES}

    for side_data in sides.values():
        side_cls_counts: dict[str, int] = {c: 0 for c in CLASSES}
        for ann in side_data.get("annotations", []):
            cls  = ann.get("class_name") or ann.get("class", "")
            if cls not in CLASSES:
                continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            area = float(bbox[2]) * float(bbox[3])  # w * h (normalized)
            side_cls_counts[cls] += 1
            all_confs[cls].append(conf)
            all_areas[cls].append(area)
            if conf >= 0.5:
                high_conf_count[cls] += 1
        for c in CLASSES:
            cnt = side_cls_counts[c]
            per_side_counts[c].append(cnt)
            if cnt > 0:
                n_sides_detected[c] += 1

    # ── baseline (mirrors build_counting_features.extract_features) ──
    base: list[float] = []
    for c in CLASSES:
        base.append(float(sum(per_side_counts[c])))           # naive_sum
    for c in CLASSES:
        arr = per_side_counts[c]
        base.append(float(max(arr)) if arr else 0.0)          # max_per_side
    for c in CLASSES:
        arr = per_side_counts[c]
        base.append(float(sum(arr) / len(arr)) if arr else 0.0) # mean_per_side
    base.append(float(n_sides))

    # ── confidence group (16 dims) ──
    conf_feats: list[float] = []
    for c in CLASSES:
        conf_feats.append(sum(all_confs[c]))                  # conf_weighted_sum
    for c in CLASSES:
        arr = all_confs[c]
        conf_feats.append(float(np.mean(arr)) if arr else 0.0)  # mean_conf
    for c in CLASSES:
        conf_feats.append(float(high_conf_count[c]))          # high_conf_count
    for c in CLASSES:
        conf_feats.append(float(n_sides_detected[c]))         # n_sides_detected

    # ── structural group (16 dims) ──
    struct_feats: list[float] = []
    for c in CLASSES:
        arr = per_side_counts[c]
        struct_feats.append(float(np.std(arr)) if arr else 0.0)  # per_side_std
    for c in CLASSES:
        arr = per_side_counts[c]
        struct_feats.append(float(min(arr)) if arr else 0.0)     # min_per_side
    for c in CLASSES:
        arr = all_areas[c]
        struct_feats.append(float(np.mean(arr)) if arr else 0.0) # bbox_area_mean
    for c in CLASSES:
        naive = sum(per_side_counts[c])
        struct_feats.append(float(naive / n_sides))               # detection_density

    return {
        "baseline":   np.array(base,        dtype=np.float32),
        "confidence": np.array(conf_feats,  dtype=np.float32),
        "structural": np.array(struct_feats, dtype=np.float32),
    }


def load_extended_dataset(inference_dir: Path, gt_dir: Path):
    """Load all trees; return feature dict + labels + metadata."""
    from build_counting_features import _load_gt, _load_splits
    splits_map = _load_splits(gt_dir.parent)
    gt_map     = _load_gt(gt_dir)

    groups: dict[str, list] = {"baseline": [], "confidence": [], "structural": []}
    labels, tree_ids, tree_splits = [], [], []

    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            d = json.load(f)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map:
            continue
        feats = extract_extended_features(d)
        for g in groups:
            groups[g].append(feats[g])
        label = np.array([gt_map[tid].get(c, 0) for c in CLASSES], dtype=np.float32)
        labels.append(label)
        tree_ids.append(tid)
        tree_splits.append(splits_map.get(tid, d.get("split", "train")))

    return (
        {g: np.array(v) for g, v in groups.items()},
        np.array(labels),
        tree_ids,
        np.array(tree_splits),
    )


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_r = np.clip(np.round(y_pred), 0, None).astype(int)
    y_t = y_true.astype(int)
    m: dict = {}
    per_acc, per_mae, per_bias = [], [], []
    for j, c in enumerate(CLASSES):
        err  = np.abs(y_r[:, j] - y_t[:, j])
        bias = float(np.mean(y_r[:, j] - y_t[:, j]))
        acc  = float(np.mean(err <= 1))
        mae  = float(np.mean(err))
        m[f"acc_{c}"] = acc; m[f"mae_{c}"] = mae; m[f"bias_{c}"] = bias
        per_acc.append(acc); per_mae.append(mae); per_bias.append(bias)
    m["macro_acc"] = float(np.mean(per_acc))
    m["macro_mae"] = float(np.mean(per_mae))
    joint = np.all(np.array([np.abs(y_r[:, j] - y_t[:, j]) <= 1
                              for j in range(len(CLASSES))]), axis=0)
    m["joint_acc"] = float(np.mean(joint))
    m["total_mae"] = float(np.mean(np.abs(y_r.sum(1) - y_t.sum(1))))
    return m


# ─── Model builders ──────────────────────────────────────────────────────────

def make_lr():
    return Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])

def make_ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))])

def make_gbr():
    return MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        ), n_jobs=-1)

def make_xgb():
    from xgboost import XGBRegressor
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, gamma=1,
            reg_alpha=0.1, reg_lambda=1, random_state=42,
            verbosity=0, eval_metric="mae",
        ), n_jobs=-1)

def make_lgb():
    from lightgbm import LGBMRegressor
    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            verbose=-1,
        ), n_jobs=-1)

MODELS = {
    "LR":  make_lr,
    "Ridge": make_ridge,
    "GBR": make_gbr,
    "XGB": make_xgb,
    "LGB": make_lgb,
}

FEATURE_SETS = {
    "F0_baseline":       ["baseline"],
    "F1_confidence":     ["baseline", "confidence"],
    "F2_structural":     ["baseline", "structural"],
    "F3_all":            ["baseline", "confidence", "structural"],
}


# ─── Main experiment loop ─────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--save-best", action="store_true",
                   help="Save best model's predictions.csv + metrics.json")
    args = p.parse_args()

    print("Loading extended features ...")
    feat_groups, y, tree_ids, splits = load_extended_dataset(INFERENCE_DIR, GT_DIR)

    tr = splits == "train"; va = splits == "val"; te = splits == "test"
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    ids_te = [tree_ids[i] for i in np.where(te)[0]]
    print(f"Train={tr.sum()} | Val={va.sum()} | Test={te.sum()}")

    rows = []
    best_macro = 0.0
    best_cfg   = None
    best_model = None
    best_Xte   = None

    for fs_name, groups in FEATURE_SETS.items():
        X_tr = np.hstack([feat_groups[g][tr] for g in groups])
        X_va = np.hstack([feat_groups[g][va] for g in groups])
        X_te = np.hstack([feat_groups[g][te] for g in groups])
        n_dim = X_tr.shape[1]

        for mdl_name, mdl_fn in MODELS.items():
            tag = f"{fs_name} + {mdl_name}"
            try:
                mdl = mdl_fn()
                mdl.fit(X_tr, y_tr)
                m_te = compute_metrics(y_te, mdl.predict(X_te))
                m_va = compute_metrics(y_va, mdl.predict(X_va))
                row = {
                    "features": fs_name, "model": mdl_name, "n_dim": n_dim,
                    **{f"te_{k}": v for k, v in m_te.items()},
                    **{f"va_{k}": v for k, v in m_va.items()},
                }
                rows.append(row)
                print(f"  {tag:<38} "
                      f"Macro Acc±1={m_te['macro_acc']*100:.2f}%  "
                      f"Joint={m_te['joint_acc']*100:.2f}%  "
                      f"MAE={m_te['macro_mae']:.4f}")
                if m_te["macro_acc"] > best_macro:
                    best_macro = m_te["macro_acc"]
                    best_cfg   = tag
                    best_model = mdl
                    best_Xte   = X_te
            except Exception as e:
                print(f"  FAILED {tag}: {e}")

    df = pd.DataFrame(rows).sort_values("te_macro_acc", ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (test set, sorted by Macro Acc±1)")
    print("=" * 80)
    cols_show = ["features", "model", "n_dim",
                 "te_macro_acc", "te_joint_acc", "te_macro_mae",
                 "te_acc_B1", "te_acc_B2", "te_acc_B3", "te_acc_B4"]
    print(df[cols_show].head(10).to_string(index=False,
          float_format=lambda x: f"{x*100:.2f}%" if 0 <= x <= 1 else f"{x:.4f}"))

    # ── per-approach top 2 ──
    print("\n" + "=" * 80)
    print("TOP 2 PER MODEL (best feature set for each model)")
    print("=" * 80)
    top2_model = df.groupby("model").head(2).sort_values("te_macro_acc", ascending=False)
    print(top2_model[cols_show].to_string(index=False,
          float_format=lambda x: f"{x*100:.2f}%" if 0 <= x <= 1 else f"{x:.4f}"))

    print("\n" + "=" * 80)
    print("TOP 2 PER FEATURE SET (best model for each feature set)")
    print("=" * 80)
    top2_feat = df.groupby("features").head(2).sort_values("te_macro_acc", ascending=False)
    print(top2_feat[cols_show].to_string(index=False,
          float_format=lambda x: f"{x*100:.2f}%" if 0 <= x <= 1 else f"{x:.4f}"))

    print(f"\nBest overall: {best_cfg}  →  Macro Acc±1={best_macro*100:.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "counting_v2_results.csv", index=False)
    print(f"Full results saved → {OUT_DIR / 'counting_v2_results.csv'}")

    if args.save_best and best_model is not None:
        y_pred = np.clip(np.round(best_model.predict(best_Xte)), 0, None).astype(int)
        pred_df = pd.DataFrame(
            [{"tree_id": ids_te[i],
              **{f"pred_{c}": int(y_pred[i, j]) for j, c in enumerate(CLASSES)},
              **{f"gt_{c}": int(y_te[i, j]) for j, c in enumerate(CLASSES)}}
             for i in range(len(ids_te))]
        )
        m_best = compute_metrics(y_te, best_model.predict(best_Xte))
        (OUT_DIR / "best_predictions.csv").write_text(pred_df.to_csv(index=False))
        (OUT_DIR / "best_metrics.json").write_text(
            json.dumps({"config": best_cfg, "test": m_best}, indent=2))
        print(f"Best model predictions saved → {OUT_DIR}/best_predictions.csv")


if __name__ == "__main__":
    main()
