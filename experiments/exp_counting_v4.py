"""
Counting experiments v4 — Maximum feature extraction + stacking ensemble.

Strategies:
  1. Extended feature set: 200+ dims
     - Multi-threshold counts (conf ≥ 0.30 / 0.35 / 0.40 / 0.45 / 0.50)
       per class: filtered_sum, filtered_max_per_side, filtered_n_sides_det
     - Derived features: duplication_factor, entropy, harmonic_mean_nonzero
     - Extended spatial: std_cy, std_area, mean_cx
     - Extended confidence: conf_std, p25, p50, p75
     - Cross-class: B1/B4 ratio, B2/B3 ratio, total_high_conf
  2. Bayesian hyperparameter search (Optuna) for XGB and LGB
  3. Stacking ensemble: Ridge(F_all) + ElasticNet(F0+spatial) + XGB_opt + LGB_opt
     → 5-fold OOF on train, meta-Ridge on OOF predictions

Usage:
  cd /workspace/Baseline-SawitMVC
  python experiments/exp_counting_v4.py
"""
from __future__ import annotations
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import _load_gt, _load_splits, CLASSES

INFERENCE_DIR = ROOT / "predictions" / "y26mv2_per_tree"
GT_DIR        = ROOT / "ground_truth" / "annotations"
OUT_DIR       = ROOT / "results" / "experiments"
THRESHOLDS    = [0.30, 0.35, 0.40, 0.45, 0.50]   # added on top of baseline 0.25


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_max_features(tree_json: dict) -> dict[str, float]:
    sides   = tree_json.get("images", {})
    n_sides = max(len(sides), 1)

    # collect per-side, per-detection data
    psc      = {c: [] for c in CLASSES}   # per-side counts (conf≥0.25)
    psc_t    = {c: {t: [] for t in THRESHOLDS} for c in CLASSES}
    cf_      = {c: [] for c in CLASSES}
    ar_      = {c: [] for c in CLASSES}
    cy_      = {c: [] for c in CLASSES}
    cx_      = {c: [] for c in CLASSES}

    for sd in sides.values():
        cnt   = {c: 0 for c in CLASSES}
        cnt_t = {c: {t: 0 for t in THRESHOLDS} for c in CLASSES}
        for ann in sd.get("annotations", []):
            cls  = ann.get("class_name", "")
            if cls not in CLASSES: continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            cx_[cls].append(float(bbox[0]))
            cy_[cls].append(float(bbox[1]))
            ar_[cls].append(float(bbox[2]) * float(bbox[3]))
            cf_[cls].append(conf)
            cnt[cls] += 1
            for t in THRESHOLDS:
                if conf >= t:
                    cnt_t[cls][t] += 1
        for c in CLASSES:
            psc[c].append(cnt[c])
            for t in THRESHOLDS:
                psc_t[c][t].append(cnt_t[c][t])

    f: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(psc[c], dtype=float)
        cf = np.array(cf_[c])
        ar = np.array(ar_[c])
        cy = np.array(cy_[c])
        cx = np.array(cx_[c])
        n  = len(cf)
        ns = len(ps)

        # ── F0 baseline ──
        f[f"naive_sum_{c}"]    = float(ps.sum())
        f[f"max_per_side_{c}"] = float(ps.max())
        f[f"mean_per_side_{c}"]= float(ps.mean())

        # ── Distribution stats ──
        f[f"std_per_side_{c}"] = float(ps.std())
        f[f"min_per_side_{c}"] = float(ps.min())
        f[f"cv_per_side_{c}"]  = float(ps.std() / (ps.mean() + 1e-6))
        f[f"n_sides_det_{c}"]  = float((ps > 0).sum())
        f[f"consistency_{c}"]  = float(1.0 / (1.0 + ps.std()))

        # entropy of per-side distribution
        p_norm = ps / (ps.sum() + 1e-9)
        entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-9)))
        f[f"entropy_{c}"]      = entropy

        # harmonic mean of nonzero per-side counts
        nz = ps[ps > 0]
        f[f"hmean_nonzero_{c}"]= float(len(nz) / np.sum(1.0 / (nz + 1e-9))) if len(nz) > 0 else 0.0

        # duplication factor estimate: naive_sum / max_per_side
        f[f"dup_factor_{c}"]   = float(ps.sum() / (ps.max() + 1e-6))

        # ── Confidence stats ──
        f[f"conf_sum_{c}"]     = float(cf.sum())
        f[f"conf_mean_{c}"]    = float(cf.mean()) if n > 0 else 0.0
        f[f"conf_std_{c}"]     = float(cf.std())  if n > 0 else 0.0
        f[f"conf_max_{c}"]     = float(cf.max())  if n > 0 else 0.0
        f[f"conf_p25_{c}"]     = float(np.percentile(cf, 25)) if n > 0 else 0.0
        f[f"conf_p50_{c}"]     = float(np.percentile(cf, 50)) if n > 0 else 0.0
        f[f"conf_p75_{c}"]     = float(np.percentile(cf, 75)) if n > 0 else 0.0
        f[f"high_conf_{c}"]    = float((cf >= 0.5).sum())
        f[f"vhigh_conf_{c}"]   = float((cf >= 0.6).sum())

        # ── Spatial stats ──
        f[f"mean_cy_{c}"]      = float(cy.mean()) if n > 0 else 0.5
        f[f"std_cy_{c}"]       = float(cy.std())  if n > 0 else 0.0
        f[f"mean_cx_{c}"]      = float(cx.mean()) if n > 0 else 0.5
        f[f"mean_area_{c}"]    = float(ar.mean()) if n > 0 else 0.0
        f[f"std_area_{c}"]     = float(ar.std())  if n > 0 else 0.0

        # ── Multi-threshold counts ──
        for t in THRESHOLDS:
            ps_t = np.array(psc_t[c][t], dtype=float)
            f[f"sum_t{int(t*100)}_{c}"]   = float(ps_t.sum())
            f[f"max_t{int(t*100)}_{c}"]   = float(ps_t.max())
            f[f"ndet_t{int(t*100)}_{c}"]  = float((ps_t > 0).sum())

    # ── Global features ──
    total = sum(f[f"naive_sum_{c}"] for c in CLASSES)
    f["n_sides"]     = float(n_sides)
    f["total_naive"] = float(total)
    for c in CLASSES:
        f[f"frac_{c}"] = f[f"naive_sum_{c}"] / (total + 1e-6)
    f["b3_b23_frac"] = f["naive_sum_B3"] / (f["naive_sum_B2"] + f["naive_sum_B3"] + 1e-6)
    # cross-class count ratios
    f["b1_b4_ratio"] = f["naive_sum_B1"] / (f["naive_sum_B4"] + 1e-6)
    f["b2_b3_ratio"] = f["naive_sum_B2"] / (f["naive_sum_B3"] + 1e-6)
    f["total_high_conf"] = sum(f[f"high_conf_{c}"] for c in CLASSES)
    return f


def load_dataset(inference_dir: Path, gt_dir: Path):
    splits_map = _load_splits(gt_dir.parent)
    gt_map     = _load_gt(gt_dir)
    rows, labels, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as fh:
            d = json.load(fh)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map: continue
        rows.append(extract_max_features(d))
        labels.append([gt_map[tid].get(c, 0) for c in CLASSES])
        tree_ids.append(tid)
        tree_splits.append(splits_map.get(tid, d.get("split", "train")))
    df = pd.DataFrame(rows)
    return df, np.array(labels, dtype=float), tree_ids, np.array(tree_splits)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yr = np.clip(np.round(y_pred), 0, None).astype(int)
    yt = y_true.astype(int)
    r: dict = {}
    for j, c in enumerate(CLASSES):
        err = np.abs(yr[:, j] - yt[:, j])
        r[f"acc_{c}"]  = float(np.mean(err <= 1))
        r[f"mae_{c}"]  = float(np.mean(err))
        r[f"bias_{c}"] = float(np.mean(yr[:, j] - yt[:, j]))
    r["macro_acc"] = float(np.mean([r[f"acc_{c}"] for c in CLASSES]))
    r["macro_mae"] = float(np.mean([r[f"mae_{c}"] for c in CLASSES]))
    r["joint_acc"] = float(np.mean(
        np.all(np.array([np.abs(yr[:, j] - yt[:, j]) <= 1
                          for j in range(len(CLASSES))]), axis=0)
    ))
    return r


def cv_mae(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []
    for tr_idx, va_idx in kf.split(X):
        m = model(); m.fit(X[tr_idx], y[tr_idx])
        p = np.clip(np.round(m.predict(X[va_idx])), 0, None)
        maes.append(np.mean(np.abs(p - y[va_idx].astype(int))))
    return float(np.mean(maes))


# ─── Optuna XGB ──────────────────────────────────────────────────────────────

def tune_xgb(X_tr, y_tr, n_trials=60):
    def objective(trial):
        params = dict(
            n_estimators   = trial.suggest_int("n_estimators", 100, 600),
            max_depth      = trial.suggest_int("max_depth", 3, 7),
            learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample      = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha      = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda     = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        )
        model_fn = lambda: MultiOutputRegressor(
            XGBRegressor(**params, random_state=42, verbosity=0), n_jobs=-1)
        return cv_mae(model_fn, X_tr, y_tr)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return MultiOutputRegressor(XGBRegressor(**best, random_state=42, verbosity=0), n_jobs=-1)


def tune_lgb(X_tr, y_tr, n_trials=60):
    def objective(trial):
        params = dict(
            n_estimators   = trial.suggest_int("n_estimators", 100, 600),
            max_depth      = trial.suggest_int("max_depth", 3, 7),
            learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves     = trial.suggest_int("num_leaves", 15, 63),
            subsample      = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha      = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda     = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        )
        model_fn = lambda: MultiOutputRegressor(
            LGBMRegressor(**params, random_state=42, verbose=-1), n_jobs=-1)
        return cv_mae(model_fn, X_tr, y_tr)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return MultiOutputRegressor(LGBMRegressor(**best, random_state=42, verbose=-1), n_jobs=-1)


# ─── Stacking ─────────────────────────────────────────────────────────────────

def stacking_predict(base_models, meta_model, X_tr, y_tr, X_te, n_folds=5):
    """5-fold OOF stacking. Returns meta-model test predictions."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_tr), len(base_models) * len(CLASSES)))
    te_preds  = np.zeros((len(X_te), len(base_models) * len(CLASSES)))

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        for m_idx, (mdl_fn, feat_idx) in enumerate(base_models):
            X_f_tr = X_tr[tr_idx][:, feat_idx]
            X_f_va = X_tr[va_idx][:, feat_idx]
            mdl = mdl_fn(); mdl.fit(X_f_tr, y_tr[tr_idx])
            pred_va = np.clip(mdl.predict(X_f_va), 0, None)
            oof_preds[va_idx, m_idx*4:(m_idx+1)*4] = pred_va
            te_preds[:, m_idx*4:(m_idx+1)*4] += np.clip(
                mdl.predict(X_te[:, feat_idx]), 0, None) / n_folds

    # fit meta model on OOF
    meta = meta_model(); meta.fit(oof_preds, y_tr)
    return meta.predict(te_preds)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading 200+ dim features ...")
    df, y, tree_ids, splits = load_dataset(INFERENCE_DIR, GT_DIR)
    all_cols = df.columns.tolist()
    print(f"Total features: {len(all_cols)}")

    tr = splits == "train"; va = splits == "val"; te = splits == "test"
    y_tr = y[tr]; y_tv = y[tr | va]; y_te = y[te]
    X_all = df.values.astype(float)
    X_tr  = X_all[tr]; X_tv = X_all[tr | va]; X_te = X_all[te]
    print(f"Train={tr.sum()} | Val={va.sum()} | Test={te.sum()}")

    # F0 cols for stacking sub-models
    F0_cols    = ([f"naive_sum_{c}" for c in CLASSES]
                + [f"max_per_side_{c}" for c in CLASSES]
                + [f"mean_per_side_{c}" for c in CLASSES]
                + ["n_sides"])
    SP_cols    = F0_cols + [f"mean_cy_{c}" for c in CLASSES] + [f"mean_area_{c}" for c in CLASSES]
    F0_idx     = [all_cols.index(c) for c in F0_cols]
    SP_idx     = [all_cols.index(c) for c in SP_cols]
    ALL_idx    = list(range(len(all_cols)))

    results = []
    print(f"\n{'Config':<50} {'Macro':>7} {'Joint':>7} {'MAE':>7}")
    print("-" * 75)

    def run(label, X_train, y_train, X_test, model_fn):
        mdl = model_fn(); mdl.fit(X_train, y_train)
        m = score(y_te, mdl.predict(X_test))
        results.append({"config": label, **m})
        print(f"{label:<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")
        return mdl

    # ── 1. Ridge on full feature set ──
    run("Ridge + F_all-67 (train)",
        X_tr[:, ALL_idx], y_tr, X_te[:, ALL_idx],
        lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]))
    run("Ridge + F_all-67 (train+val)",
        X_tv[:, ALL_idx], y_tv, X_te[:, ALL_idx],
        lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]))

    # ── 2. ElasticNet variants ──
    run("ElasticNet + F_all (train)",
        X_tr[:, ALL_idx], y_tr, X_te[:, ALL_idx],
        lambda: Pipeline([("sc", StandardScaler()), ("en", MultiTaskElasticNetCV(cv=5, random_state=42, max_iter=5000))]))
    run("ElasticNet + F0+spatial (train)",
        X_tr[:, SP_idx], y_tr, X_te[:, SP_idx],
        lambda: Pipeline([("sc", StandardScaler()), ("en", MultiTaskElasticNetCV(cv=5, random_state=42, max_iter=5000))]))

    # ── 3. Optuna XGB ──
    print(f"\n{'Tuning XGB (Optuna, 60 trials)...':<50}", flush=True)
    xgb_opt = tune_xgb(X_tr[:, ALL_idx], y_tr)
    xgb_opt.fit(X_tr[:, ALL_idx], y_tr)
    m = score(y_te, xgb_opt.predict(X_te[:, ALL_idx]))
    results.append({"config": "XGB-Optuna + F_all (train)", **m})
    print(f"{'XGB-Optuna + F_all (train)':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    xgb_tv = tune_xgb(X_tv[:, ALL_idx], y_tv, n_trials=40)
    xgb_tv.fit(X_tv[:, ALL_idx], y_tv)
    m = score(y_te, xgb_tv.predict(X_te[:, ALL_idx]))
    results.append({"config": "XGB-Optuna + F_all (train+val)", **m})
    print(f"{'XGB-Optuna + F_all (train+val)':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    # ── 4. Optuna LGB ──
    print(f"\n{'Tuning LGB (Optuna, 60 trials)...':<50}", flush=True)
    lgb_opt = tune_lgb(X_tr[:, ALL_idx], y_tr)
    lgb_opt.fit(X_tr[:, ALL_idx], y_tr)
    m = score(y_te, lgb_opt.predict(X_te[:, ALL_idx]))
    results.append({"config": "LGB-Optuna + F_all (train)", **m})
    print(f"{'LGB-Optuna + F_all (train)':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    lgb_tv = tune_lgb(X_tv[:, ALL_idx], y_tv, n_trials=40)
    lgb_tv.fit(X_tv[:, ALL_idx], y_tv)
    m = score(y_te, lgb_tv.predict(X_te[:, ALL_idx]))
    results.append({"config": "LGB-Optuna + F_all (train+val)", **m})
    print(f"{'LGB-Optuna + F_all (train+val)':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    # ── 5. Stacking ──
    print(f"\n{'Stacking ensemble (5-fold OOF)...':<50}", flush=True)
    base_models = [
        (lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]), ALL_idx),
        (lambda: Pipeline([("sc", StandardScaler()), ("en", MultiTaskElasticNetCV(cv=5, random_state=42, max_iter=3000))]), SP_idx),
        (lambda: MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, random_state=42, verbosity=0), n_jobs=-1), ALL_idx),
        (lambda: MultiOutputRegressor(LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
               num_leaves=31, subsample=0.8, random_state=42, verbose=-1), n_jobs=-1), ALL_idx),
    ]
    meta_fn = lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))])

    for label, X_train, y_train in [("train", X_tr, y_tr), ("train+val", X_tv, y_tv)]:
        stk_pred = stacking_predict(base_models, meta_fn, X_train, y_train, X_te)
        m = score(y_te, stk_pred)
        results.append({"config": f"Stacking (4 base + Ridge meta, {label})", **m})
        print(f"{'Stacking (4 base + Ridge meta, ' + label + ')':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    # ── 6. Optuna stacking ──
    print(f"\n{'Stacking with Optuna base models...':<50}", flush=True)
    base_opt = [
        (lambda: Pipeline([("sc", StandardScaler()), ("rid", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500]))]), ALL_idx),
        (lambda: Pipeline([("sc", StandardScaler()), ("en", MultiTaskElasticNetCV(cv=5, random_state=42, max_iter=3000))]), SP_idx),
        (lambda xgb_opt=xgb_opt: xgb_opt, ALL_idx),
        (lambda lgb_opt=lgb_opt: lgb_opt, ALL_idx),
    ]
    stk_opt = stacking_predict(base_opt, meta_fn, X_tr, y_tr, X_te)
    m = score(y_te, stk_opt)
    results.append({"config": "Stacking-Optuna (4 base, train)", **m})
    print(f"{'Stacking-Optuna (4 base, train)':<50} {m['macro_acc']*100:6.2f}% {m['joint_acc']*100:6.2f}% {m['macro_mae']:6.4f}")

    # ── Summary ──
    rdf = pd.DataFrame(results).sort_values("macro_acc", ascending=False)
    print(f"\n{'='*75}")
    print("FINAL RANKING")
    print(f"{'='*75}")
    cols = ["config", "macro_acc", "joint_acc", "macro_mae",
            "acc_B1", "acc_B2", "acc_B3", "acc_B4",
            "bias_B1", "bias_B2", "bias_B3", "bias_B4"]
    print(rdf[cols].to_string(index=False,
          float_format=lambda x: f"{x*100:.2f}%" if 0 <= x <= 1 else f"{x:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(OUT_DIR / "counting_v4_results.csv", index=False)
    print(f"\nSaved → {OUT_DIR}/counting_v4_results.csv")
    print(f"Total features used: {len(all_cols)}")


if __name__ == "__main__":
    main()
