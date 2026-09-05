"""
A6 - Why Random Forest trails the linear and kernel counters.

Answers the reviewer request to discuss the poor Random Forest performance more
thoroughly.

For every counter the script reports the dynamic range of its predictions, the
shrinkage of that range relative to the targets (standard-deviation ratio and
the slope of an ordinary least-squares fit of prediction on target), and
accuracy split by how many bunches a tree actually carries. A forest predicts
the mean of the training targets that fall in each leaf, so it cannot produce a
value outside the training range and it pulls sparsely populated high counts
towards the bulk of the distribution; the table makes that visible.

Usage:  python experiments/revision/a6_rf_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

MODELS = ["LR", "Ridge", "ElasticNet", "SVM", "RF"]


def main() -> None:
    rows, strat_rows = [], []
    for cond, source in (("fixed", rc.PRED_DIR), ("gt", rc.GT_DIR)):
        df, y, tree_ids, splits, _ = rc.load_dataset(source)
        tr, te = splits == "train", splits == "test"
        fsets = rc.feature_sets(df)
        x = df[fsets["F0"]].values.astype(float)
        y_te = y[te]
        totals = y_te.sum(axis=1)
        edges = np.quantile(totals, [1 / 3, 2 / 3])
        bucket = np.digitize(totals, edges)

        for model_name in MODELS:
            model = rc.model_builders()[model_name]()
            model.fit(x[tr], y[tr])
            pred = rc.round_counts(model.predict(x[te]))
            st = rc.per_tree_stats(y_te, pred)
            for j, c in enumerate(rc.CLASSES):
                gt_j, pr_j = y_te[:, j], pred[:, j].astype(float)
                slope = float(np.polyfit(gt_j, pr_j, 1)[0]) if gt_j.std() > 0 else float("nan")
                rows.append({
                    "condition": cond, "model": model_name, "class": c,
                    "gt_min": float(gt_j.min()), "gt_max": float(gt_j.max()),
                    "gt_train_max": float(y[tr][:, j].max()),
                    "pred_min": float(pr_j.min()), "pred_max": float(pr_j.max()),
                    "std_ratio": float(pr_j.std() / gt_j.std()) if gt_j.std() > 0 else float("nan"),
                    "ols_slope": slope,
                    "pearson_r": float(np.corrcoef(gt_j, pr_j)[0, 1]) if pr_j.std() > 0 else float("nan"),
                    "mae": float(np.abs(pr_j - gt_j).mean()),
                    "bias": float((pr_j - gt_j).mean()),
                })
            for b, name in enumerate(["low", "mid", "high"]):
                sel = bucket == b
                m = rc.metrics_from_stats({k: v[sel] for k, v in st.items()})
                strat_rows.append({
                    "condition": cond, "model": model_name, "tree_load": name,
                    "n_trees": int(sel.sum()),
                    "mean_gt_total": float(totals[sel].mean()),
                    "macro": m["macro"], "joint": m["joint"], "mae": m["mae"],
                })

    range_df = pd.DataFrame(rows)
    strat_df = pd.DataFrame(strat_rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    range_df.to_csv(rc.OUT_DIR / "a6_prediction_range.csv", index=False)
    strat_df.to_csv(rc.OUT_DIR / "a6_accuracy_by_load.csv", index=False)

    print("Prediction range and shrinkage, fixed detector, F0 features:")
    sub = range_df[range_df.condition == "fixed"]
    print(sub.pivot(index="class", columns="model",
                    values="pred_max").to_string(), "\n(pred_max)")
    print(sub.pivot(index="class", columns="model",
                    values="std_ratio").round(3).to_string(), "\n(std_ratio)")
    print(sub.pivot(index="class", columns="model",
                    values="ols_slope").round(3).to_string(), "\n(ols_slope)")
    print("\nGT max per class in the test split:")
    print(sub[sub.model == "RF"][["class", "gt_max", "gt_train_max"]].to_string(index=False))

    print("\nClass +-1 Acc by tree load, fixed detector:")
    print(strat_df[strat_df.condition == "fixed"]
          .pivot(index="tree_load", columns="model", values="macro")
          .mul(100).round(2).to_string())
    print("\nClass +-1 Acc by tree load, GT detections:")
    print(strat_df[strat_df.condition == "gt"]
          .pivot(index="tree_load", columns="model", values="macro")
          .mul(100).round(2).to_string())
    print(f"\nwrote {rc.OUT_DIR / 'a6_prediction_range.csv'}")


if __name__ == "__main__":
    main()
