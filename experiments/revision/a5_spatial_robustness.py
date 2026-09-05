"""
A5 - Robustness of the vertical spatial features to camera-pitch variation.

Answers the reviewer request to clarify how robust the vertical spatial
features are to real-world camera pitch variation.

The spatial group holds, per class, the mean normalised vertical centroid
(mean_cy) and the mean box area. A pitch change moves the horizon in the frame,
approximated here by a synthetic affine feature perturbation cy -> a*cy + b.
This is not a calibrated camera rotation: visibility, area and detection
quality remain fixed. Missing-class sentinel values remain unchanged. The counter is
fitted on unperturbed training trees and evaluated on perturbed test trees, so
the sweep measures a train/test pitch mismatch rather than a global convention
change.

Reported alongside: the per-class separation of mean_cy that the feature relies
on, and the accuracy of the same counter with the spatial group removed.

Usage:  python experiments/revision/a5_spatial_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

SHIFTS = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
NOISE = [0.0, 0.01, 0.03, 0.05]


def main() -> None:
    df, y, tree_ids, splits, _ = rc.load_dataset(rc.PRED_DIR)
    tr, te = splits == "train", splits == "test"
    fsets = rc.feature_sets(df)
    cy_cols = [f"mean_cy_{c}" for c in rc.CLASSES]

    # ---- how separated are the classes on the vertical axis? --------------
    sep_rows = []
    for source, label in ((rc.PRED_DIR, "fixed detector"), (rc.GT_DIR, "ground truth")):
        d, _, _, sp, _ = rc.load_dataset(source)
        sub = d.loc[sp == "test"]
        for c in rc.CLASSES:
            vals = sub.loc[sub[f"naive_sum_{c}"] > 0, f"mean_cy_{c}"]
            sep_rows.append({"source": label, "class": c,
                             "mean": float(vals.mean()), "std": float(vals.std())})
    sep = pd.DataFrame(sep_rows)
    print("Per-class mean_cy on the test split (0 = image top, 1 = image bottom):")
    print(sep.pivot(index="class", columns="source", values=["mean", "std"]).to_string())

    rows = []
    for features, model_name in (("F_all", "Ridge"), ("F0+spatial", "Ridge")):
        cols = fsets[features]
        x = df[cols].values.astype(float)
        model = rc.model_builders()[model_name]()
        model.fit(x[tr], y[tr])
        pos = [cols.index(c) for c in cy_cols]
        present = df.loc[te, [f"naive_sum_{c}" for c in rc.CLASSES]].to_numpy() > 0
        base = rc.metrics_from_stats(
            rc.per_tree_stats(y[te], rc.round_counts(model.predict(x[te]))))

        rng = np.random.default_rng(rc.SEED)
        for scale in SCALES:
            for shift in SHIFTS:
                for sigma in NOISE:
                    if sigma > 0 and (scale != 1.0 or shift != 0.0):
                        continue  # sweep noise only on the unperturbed geometry
                    xt = x[te].copy()
                    xt[:, pos] = xt[:, pos] * scale + shift
                    if sigma > 0:
                        xt[:, pos] += rng.normal(0.0, sigma, size=xt[:, pos].shape)
                    xt[:, pos] = np.clip(xt[:, pos], 0.0, 1.0)
                    xt[:, pos] = np.where(present, xt[:, pos], x[te][:, pos])
                    m = rc.metrics_from_stats(
                        rc.per_tree_stats(y[te], rc.round_counts(model.predict(xt))))
                    rows.append({"features": features, "model": model_name,
                                 "scale": scale, "shift": shift, "noise": sigma,
                                 "macro": m["macro"], "joint": m["joint"], "mae": m["mae"],
                                 "d_macro_pp": (m["macro"] - base["macro"]) * 100,
                                 "d_joint_pp": (m["joint"] - base["joint"]) * 100})

    out = pd.DataFrame(rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(rc.OUT_DIR / "a5_pitch_robustness.csv", index=False)
    sep.to_csv(rc.OUT_DIR / "a5_cy_separation.csv", index=False)

    fall = out[(out.features == "F_all") & (out.noise == 0)]
    print("\nRidge + F_all, Class +-1 Acc under an affine pitch perturbation of mean_cy:")
    print(fall.pivot(index="shift", columns="scale", values="macro").mul(100).round(2).to_string())
    print("\nWorst case over the whole sweep: "
          f"{fall.macro.min()*100:.2f}% (delta {fall.d_macro_pp.min():+.2f} pp), "
          f"largest gain {fall.d_macro_pp.max():+.2f} pp")
    noise_rows = out[(out.features == "F_all") & (out.noise > 0)]
    print("\nAdditive noise on mean_cy:")
    print(noise_rows[["noise", "macro", "joint", "d_macro_pp"]].to_string(index=False))
    print(f"\nwrote {rc.OUT_DIR / 'a5_pitch_robustness.csv'}")


if __name__ == "__main__":
    main()
