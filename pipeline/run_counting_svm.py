"""
SVM counting — train on 13-dim features from inference JSONs, evaluate on test split.

Usage:
    python pipeline/run_counting_svm.py
    python pipeline/run_counting_svm.py --inference-dir predictions/y26mv2_per_tree/
    python pipeline/run_counting_svm.py --inference-dir predictions/y26mv2_per_tree/ \
        --out results/e2e_per_tree/y26mv2_svm/
    # Save the fitted model so later evaluations skip training:
    python pipeline/run_counting_svm.py --save-model models/counters/svm.pkl
    # Reuse a previously saved model (no GridSearchCV):
    python pipeline/run_counting_svm.py --load-model models/counters/svm.pkl
"""
from __future__ import annotations
import argparse, json, os, pickle, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import load_dataset, CLASSES, FEATURE_NAMES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tree_ids: list[str]) -> tuple[dict, pd.DataFrame]:
    y_pred_r = np.clip(np.round(y_pred), 0, None).astype(int)
    y_true_i = y_true.astype(int)
    rows = [dict(tree_id=tid,
                 **{f"pred_{c}": y_pred_r[i, j] for j, c in enumerate(CLASSES)},
                 **{f"gt_{c}": y_true_i[i, j] for j, c in enumerate(CLASSES)})
            for i, tid in enumerate(tree_ids)]
    df = pd.DataFrame(rows)
    m: dict = {"n_trees": int(len(tree_ids))}
    for j, c in enumerate(CLASSES):
        err = np.abs(y_pred_r[:, j] - y_true_i[:, j])
        m[f"MAE_{c}"] = float(np.mean(err))
        m[f"bias_{c}"] = float(np.mean(y_pred_r[:, j] - y_true_i[:, j]))
        m[f"acc_pm1_{c}"] = float(np.mean(err <= 1))
    m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
    m["macro_acc_pm1"] = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
    total_err = np.abs(y_pred_r.sum(1) - y_true_i.sum(1))
    m["total_count_mae"] = float(np.mean(total_err))
    m["total_pm1_acc"] = float(np.mean(total_err <= 1))
    m["exact_profile_acc"] = float(np.mean(np.all(y_pred_r == y_true_i, axis=1)))
    return m, df


def main() -> None:
    p = argparse.ArgumentParser(description="SVM counting on inference features")
    p.add_argument("--inference-dir", type=Path,
                   default=ROOT / "predictions" / "y26mv2_per_tree",
                   help="Folder of prediction JSONs (one per tree)")
    p.add_argument("--gt-dir", type=Path,
                   default=ROOT / "ground_truth" / "annotations",
                   help="Ground-truth JSON folder")
    p.add_argument("--out", type=Path, default=None,
                   help="Output folder. Default: results/e2e_per_tree/{name}_svm/")
    p.add_argument("--save-model", type=Path, default=None,
                   help="Optional path to pickle the fitted estimator (e.g. models/counters/svm.pkl).")
    p.add_argument("--load-model", type=Path, default=None,
                   help="Optional path to reuse a previously fitted estimator; skips GridSearchCV.")
    args = p.parse_args()

    name = args.inference_dir.name.replace("_per_tree", "").replace("_inference", "")
    out_dir = args.out or ROOT / "results" / "e2e_per_tree" / f"{name}_svm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.inference_dir} ...")
    X, y, tree_ids, tree_splits = load_dataset(args.inference_dir, args.gt_dir)
    splits = np.array(tree_splits)

    train_m, val_m, test_m = splits == "train", splits == "val", splits == "test"
    X_tr, y_tr = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_te, y_te = X[test_m], y[test_m]
    ids_val = [tree_ids[i] for i in np.where(val_m)[0]]
    ids_te  = [tree_ids[i] for i in np.where(test_m)[0]]
    print(f"Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

    if args.load_model and args.load_model.exists():
        print(f"Loading fitted estimator from {args.load_model}")
        with open(args.load_model, "rb") as f:
            estimator = pickle.load(f)
        best_params = getattr(estimator, "best_params_", {})
        cv_mae = float("nan")
    else:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("svr", MultiOutputRegressor(SVR(kernel="rbf")))])
        grid = {"svr__estimator__C": [0.1, 1, 10],
                "svr__estimator__gamma": ["scale", 0.01, 0.1]}
        print("GridSearchCV (3-fold) ...")
        gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        estimator = gs
        best_params = gs.best_params_
        cv_mae = float(-gs.best_score_)
        print(f"Best params: {best_params}  CV-MAE: {cv_mae:.4f}")

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_model, "wb") as f:
            pickle.dump(estimator, f)
        print(f"Saved fitted estimator to {args.save_model}")

    m_te, df_te = compute_metrics(y_te, estimator.predict(X_te), ids_te)
    m_te["best_params"] = str(best_params)
    m_te["cv_mae"] = cv_mae
    m_te["split"] = "test"

    m_val, df_val = compute_metrics(y_val, estimator.predict(X_val), ids_val)
    m_val["split"] = "val"

    (out_dir / "metrics.json").write_text(json.dumps({"test": m_te, "val": m_val}, indent=2))
    df_te.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n=== {name} - SVM (test, n={X_te.shape[0]}) ===")
    print(f"  Class ±1 Acc : {m_te['macro_acc_pm1']*100:.2f}%")
    print(f"  Macro MAE     : {m_te['macro_class_mae']:.4f}")
    print(f"  Total MAE     : {m_te['total_count_mae']:.4f}")
    for c in CLASSES:
        print(f"  {c}: Class ±1 Acc={m_te[f'acc_pm1_{c}']*100:.1f}%  MAE={m_te[f'MAE_{c}']:.3f}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
