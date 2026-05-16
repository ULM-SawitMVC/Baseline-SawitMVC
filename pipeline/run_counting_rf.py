"""
Random Forest counting — train on 13-dim features from inference JSONs, evaluate on test split.

Usage:
    python pipeline/run_counting_rf.py
    python pipeline/run_counting_rf.py --inference-dir predictions/y26m_vanilla_local_inference/
    python pipeline/run_counting_rf.py --inference-dir predictions/y26m_vanilla_local_inference/ --out benchmarks/e2e/my_result/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_counting_features import load_dataset, CLASSES, FEATURE_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, tree_ids: list[str]) -> tuple[dict, pd.DataFrame]:
    y_pred_r = np.clip(np.round(y_pred), 0, None).astype(int)
    y_true_i = y_true.astype(int)
    rows = [dict(tree_id=tid,
                 **{f"pred_{c}": y_pred_r[i, j] for j, c in enumerate(CLASSES)},
                 **{f"gt_{c}": y_true_i[i, j] for j, c in enumerate(CLASSES)})
            for i, tid in enumerate(tree_ids)]
    df = pd.DataFrame(rows)
    m: dict = {}
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
    p = argparse.ArgumentParser(description="RF counting on inference features")
    p.add_argument("--inference-dir", type=Path,
                   default=ROOT / "predictions" / "y26n_vanilla_local_inference",
                   help="Folder of prediction JSONs (one per tree)")
    p.add_argument("--gt-dir", type=Path,
                   default=ROOT / "SawitMVC-YOLO" / "json",
                   help="GT JSON folder (SawitMVC-YOLO/json/)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output folder. Default: benchmarks/e2e/e2e_{name}_rf/")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=10)
    args = p.parse_args()

    name = args.inference_dir.name.replace("_inference", "")
    out_dir = args.out or ROOT / "benchmarks" / "e2e" / f"e2e_{name}_rf"
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

    rf = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                               random_state=42, n_jobs=-1)
    print(f"Training RF (n={args.n_estimators}, max_depth={args.max_depth}) ...")
    rf.fit(X_tr, y_tr)

    m_te, df_te = compute_metrics(y_te, rf.predict(X_te), ids_te)
    m_te["n_estimators"] = args.n_estimators
    m_te["max_depth"] = args.max_depth
    m_te["split"] = "test"

    m_val, df_val = compute_metrics(y_val, rf.predict(X_val), ids_val)
    m_val["split"] = "val"

    (out_dir / "metrics.json").write_text(json.dumps({"test": m_te, "val": m_val}, indent=2))
    df_te.to_csv(out_dir / "predictions.csv", index=False)

    # Feature importance
    fi = pd.DataFrame({"feature": FEATURE_NAMES,
                        "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\n=== {name} → RF (test, n={X_te.shape[0]}) ===")
    print(f"  Macro Acc±1 : {m_te['macro_acc_pm1']*100:.2f}%")
    print(f"  Macro MAE   : {m_te['macro_class_mae']:.4f}")
    print(f"  Total MAE   : {m_te['total_count_mae']:.4f}")
    for c in CLASSES:
        print(f"  {c}: Acc±1={m_te[f'acc_pm1_{c}']*100:.1f}%  MAE={m_te[f'MAE_{c}']:.3f}")
    print(f"\nTop-5 features:\n{fi.head(5).to_string(index=False)}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
