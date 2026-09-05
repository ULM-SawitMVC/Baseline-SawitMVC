"""
A3 - Repeated cross-validation and cross-plantation transfer.

Answers reviewer requests: strengthen validation with repeated splits or
cross-validation, and use an independent plantation where one is available.

Two protocols:

* Repeated stratified 5-fold cross-validation, 5 repeats (seeds 42..46).
  Under GT detections every tree is usable. Under the fixed detector the folds
  are drawn from 237 trees excluded from detector gradient training (val and
  test). Validation trees still participated in detector development. The
  all-953 variant is computed for contrast and includes detector training trees.
  The manuscript now uses a10 for matched GT/fixed pools and a test-only check.

* Cross-plantation transfer between the two estates in SawitMVC (DAMIMAS and
  LONSUM): the counter is fitted on one estate and evaluated on the other.
  Under the fixed detector the evaluation estate contributes only its
  trees outside detector gradient training. The detector was trained on both
  estates, so this is counter transfer, not independent full-pipeline transfer.

Usage:  python experiments/revision/a3_resampling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

N_REPEATS = 5
N_FOLDS = 5
CONFIGS = [("F0", "LR"), ("F0", "Ridge"), ("F0", "ElasticNet"), ("F0", "SVM"),
           ("F0", "RF"), ("F_all", "Ridge")]


def cv_scores(df, y, strat, pool_mask, fsets, features, model_name):
    x = df[fsets[features]].values.astype(float)
    xp, yp, sp = x[pool_mask], y[pool_mask], strat[pool_mask]
    fold_metrics = []
    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rc.SEED + repeat)
        for tr_idx, te_idx in skf.split(xp, sp):
            model = rc.model_builders()[model_name]()
            model.fit(xp[tr_idx], yp[tr_idx])
            pred = rc.round_counts(model.predict(xp[te_idx]))
            fold_metrics.append(rc.metrics_from_stats(rc.per_tree_stats(yp[te_idx], pred)))
    keys = ["macro", "joint", "mae", "rmse", "total_mae"]
    return {k: (float(np.mean([m[k] for m in fold_metrics])),
                float(np.std([m[k] for m in fold_metrics], ddof=1))) for k in keys}


def main() -> None:
    manifest = pd.read_csv(rc.ROOT / "ground_truth" / "split_manifest.csv",
                           encoding="utf-8-sig")
    strat_map = dict(zip(manifest.tree_id, manifest.strat_key))

    sources = {"gt": rc.GT_DIR, "fixed": rc.PRED_DIR}
    cv_rows, transfer_rows = [], []

    for cond, source in sources.items():
        df, y, tree_ids, splits, varieties = rc.load_dataset(source)
        fsets = rc.feature_sets(df)
        strat = np.array([strat_map.get(t, "NA") for t in tree_ids])
        held_out = np.isin(splits, ["val", "test"])
        pools = {"all953": np.ones(len(tree_ids), dtype=bool)}
        if cond == "fixed":
            pools["detector_heldout237"] = held_out

        for pool_name, pool_mask in pools.items():
            for features, model_name in CONFIGS:
                s = cv_scores(df, y, strat, pool_mask, fsets, features, model_name)
                cv_rows.append({
                    "condition": cond, "pool": pool_name, "n_pool": int(pool_mask.sum()),
                    "features": features, "model": model_name,
                    **{f"{k}_mean": v[0] for k, v in s.items()},
                    **{f"{k}_std": v[1] for k, v in s.items()},
                })
                print(f"CV {cond:5s} {pool_name:20s} {features:6s} {model_name:11s} "
                      f"macro={s['macro'][0]*100:6.2f}+-{s['macro'][1]*100:4.2f}  "
                      f"tree={s['joint'][0]*100:6.2f}+-{s['joint'][1]*100:4.2f}  "
                      f"mae={s['mae'][0]:.3f}+-{s['mae'][1]:.3f}")

        # ---- cross-plantation transfer ------------------------------------
        estates = sorted(set(varieties))
        for features, model_name in CONFIGS:
            x = df[fsets[features]].values.astype(float)
            for train_estate in estates:
                for test_estate in estates:
                    tr_mask = (varieties == train_estate) & (splits == "train")
                    te_mask = (varieties == test_estate) & (
                        held_out if cond == "fixed" else np.isin(splits, ["val", "test"]))
                    if tr_mask.sum() < 30 or te_mask.sum() < 20:
                        continue
                    model = rc.model_builders()[model_name]()
                    model.fit(x[tr_mask], y[tr_mask])
                    pred = rc.round_counts(model.predict(x[te_mask]))
                    m = rc.metrics_from_stats(rc.per_tree_stats(y[te_mask], pred))
                    transfer_rows.append({
                        "condition": cond, "features": features, "model": model_name,
                        "train_estate": train_estate, "test_estate": test_estate,
                        "same_estate": train_estate == test_estate,
                        "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
                        "macro": m["macro"], "joint": m["joint"], "mae": m["mae"],
                        "total_mae": m["total_mae"],
                    })

    cv_df = pd.DataFrame(cv_rows)
    tr_df = pd.DataFrame(transfer_rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(rc.OUT_DIR / "a3_repeated_cv.csv", index=False)
    tr_df.to_csv(rc.OUT_DIR / "a3_cross_estate.csv", index=False)

    print("\nCross-plantation transfer (Ridge + F_all):")
    sub = tr_df[(tr_df.model == "Ridge") & (tr_df.features == "F_all")]
    print(sub.to_string(index=False))
    print("\nCross-plantation transfer (ElasticNet + F0):")
    sub = tr_df[(tr_df.model == "ElasticNet") & (tr_df.features == "F0")]
    print(sub.to_string(index=False))
    print(f"\nwrote {rc.OUT_DIR / 'a3_repeated_cv.csv'} and {rc.OUT_DIR / 'a3_cross_estate.csv'}")


if __name__ == "__main__":
    main()
