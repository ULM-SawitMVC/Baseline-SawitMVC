"""
A9 - Operating-point sweep extended below the released threshold.

The cached predictions of the paper were produced at a confidence threshold of
0.25, so they can only be filtered upwards. This script uses a second inference
pass of the same checkpoint at 0.05 (predictions/y26mv2_conf005_per_tree) to
extend the sweep downwards, where the detector emits more evidence and more
false positives.

The consistency of the two passes is checked at 0.25 before the sweep is used.

Usage:
  python pipeline/run_e2e_inference.py --name y26mv2_conf005 \
      --weights models/yolo/y26mv2.pt --out predictions/y26mv2_conf005_per_tree \
      --conf 0.05
  python experiments/revision/a9_low_threshold_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

LOW_DIR = rc.ROOT / "predictions" / "y26mv2_conf005_per_tree"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
CONFIGS = [("F0", "ElasticNet"), ("F0", "Ridge"), ("F_all", "Ridge")]


def evaluate(source: Path, tau: float) -> list[dict]:
    df, y, tree_ids, splits, _ = rc.load_dataset(source, conf_min=tau)
    tr, te = splits == "train", splits == "test"
    fsets = rc.feature_sets(df)
    n_det = float(df.loc[te, "total_naive"].sum())
    rows = []
    for features, model_name in CONFIGS:
        x = df[fsets[features]].values.astype(float)
        model = rc.model_builders()[model_name]()
        model.fit(x[tr], y[tr])
        m = rc.metrics_from_stats(
            rc.per_tree_stats(y[te], rc.round_counts(model.predict(x[te]))))
        rows.append({"threshold": tau, "features": features, "model": model_name,
                     "test_detections": n_det, **m})
    return rows


def main() -> None:
    n_low = len(list(LOW_DIR.glob("*.json")))
    if n_low != 953:
        raise SystemExit(f"expected 953 low-threshold prediction files, found {n_low}")

    ref = evaluate(rc.PRED_DIR, 0.25)
    chk = evaluate(LOW_DIR, 0.25)
    print("consistency at tau = 0.25 (released pass vs 0.05 pass filtered up):")
    for a, b in zip(ref, chk):
        print(f"  {a['features']:6s} {a['model']:11s} "
              f"det {a['test_detections']:.0f} vs {b['test_detections']:.0f}   "
              f"macro {a['macro']*100:.2f}% vs {b['macro']*100:.2f}%")

    rows = []
    for tau in THRESHOLDS:
        for row in evaluate(LOW_DIR, tau):
            rows.append(row)
            if row["features"] == "F_all":
                print(f"tau={tau:.2f} det={row['test_detections']:6.0f}  "
                      f"macro={row['macro']*100:6.2f}%  tree={row['joint']*100:6.2f}%  "
                      f"mae={row['mae']:.4f}")
    out = pd.DataFrame(rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(rc.OUT_DIR / "a9_threshold_sweep_full.csv", index=False)
    print(f"\nwrote {rc.OUT_DIR / 'a9_threshold_sweep_full.csv'}")


if __name__ == "__main__":
    main()
