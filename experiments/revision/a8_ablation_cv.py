"""
A8 - The feature ablation re-run under repeated cross-validation.

The ablation in the paper is measured on one fixed test split. This script
repeats it with Ridge under repeated stratified 5-fold cross-validation (5
repeats) over the 237 trees outside detector gradient training. This includes
detector validation trees. Fold SD is descriptive; repeated folds overlap.

Usage:  python experiments/revision/a8_ablation_cv.py
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


def main() -> None:
    manifest = pd.read_csv(rc.ROOT / "ground_truth" / "split_manifest.csv",
                           encoding="utf-8-sig")
    strat_map = dict(zip(manifest.tree_id, manifest.strat_key))

    df, y, tree_ids, splits, _ = rc.load_dataset(rc.PRED_DIR)
    fsets = rc.feature_sets(df)
    strat = np.array([strat_map.get(t, "NA") for t in tree_ids])
    pool = np.isin(splits, ["val", "test"])
    tr, te = splits == "train", splits == "test"

    rows = []
    for name, cols in fsets.items():
        x = df[cols].values.astype(float)

        model = rc.model_builders()["Ridge"]()
        model.fit(x[tr], y[tr])
        fixed = rc.metrics_from_stats(
            rc.per_tree_stats(y[te], rc.round_counts(model.predict(x[te]))))

        xp, yp, sp = x[pool], y[pool], strat[pool]
        folds = []
        for repeat in range(N_REPEATS):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=rc.SEED + repeat)
            for a, b in skf.split(xp, sp):
                m = rc.model_builders()["Ridge"]()
                m.fit(xp[a], yp[a])
                folds.append(rc.metrics_from_stats(
                    rc.per_tree_stats(yp[b], rc.round_counts(m.predict(xp[b])))))
        cv = {k: (float(np.mean([f[k] for f in folds])),
                  float(np.std([f[k] for f in folds], ddof=1)))
              for k in ("macro", "joint", "mae")}

        rows.append({"features": name, "n_dim": len(cols),
                     "split_macro": fixed["macro"], "split_joint": fixed["joint"],
                     "split_mae": fixed["mae"],
                     "cv_macro_mean": cv["macro"][0], "cv_macro_std": cv["macro"][1],
                     "cv_joint_mean": cv["joint"][0], "cv_joint_std": cv["joint"][1],
                     "cv_mae_mean": cv["mae"][0], "cv_mae_std": cv["mae"][1]})
        print(f"{name:20s} dim={len(cols):3d}  split={fixed['macro']*100:6.2f}%  "
              f"cv={cv['macro'][0]*100:6.2f}+-{cv['macro'][1]*100:4.2f}%  "
              f"cv tree={cv['joint'][0]*100:6.2f}+-{cv['joint'][1]*100:4.2f}%")

    out = pd.DataFrame(rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(rc.OUT_DIR / "a8_ablation_cv.csv", index=False)
    print(f"\nwrote {rc.OUT_DIR / 'a8_ablation_cv.csv'}")


if __name__ == "__main__":
    main()
