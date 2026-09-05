"""
A1 - Extended counting metrics with bootstrap uncertainty and paired tests.

Answers reviewer requests: statistical uncertainty/significance for the reported
model differences, and additional counting metrics (per-class MAE/RMSE and
total bunch-count error).

Protocol is unchanged from the submitted paper: counters are fitted on the 716
training trees and evaluated on the 141 test trees. Uncertainty comes from a
tree-level percentile bootstrap (B = 10000, seed 42); model-vs-model tests are
paired on the same trees.

Usage:  python experiments/revision/a1_metrics_significance.py

The bootstrap-tail p_boot values below are retained for historical comparison.
The audited manuscript uses a10_revision_audit.py for exact paired p-values,
complete comparison families, and Holm correction; a1 remains the CI source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

N_BOOT = 10000
CI_KEYS = ["macro", "joint", "mae", "rmse", "total_mae", "total_rmse", "total_acc",
           "exact"] + [f"acc_{c}" for c in rc.CLASSES] + [f"mae_{c}" for c in rc.CLASSES] \
    + [f"rmse_{c}" for c in rc.CLASSES] + [f"bias_{c}" for c in rc.CLASSES]
DIFF_KEYS = ["macro", "joint", "mae", "total_mae"]


def fit_predict(df, y, mask_train, mask_test, columns, model_name):
    x = df[columns].values.astype(float)
    model = rc.model_builders()[model_name]()
    model.fit(x[mask_train], y[mask_train])
    return rc.round_counts(model.predict(x[mask_test]))


def main() -> None:
    conditions = {
        "fixed": rc.PRED_DIR,
        "gt": rc.GT_DIR,
    }
    preds: dict[tuple[str, str, str], np.ndarray] = {}
    truth: np.ndarray | None = None
    rows: list[dict] = []

    for cond, source in conditions.items():
        df, y, tree_ids, splits, _ = rc.load_dataset(source)
        tr = splits == "train"
        te = splits == "test"
        if truth is None:
            truth = y[te]
        else:
            assert np.array_equal(truth, y[te]), "test targets differ between conditions"
        fsets = rc.feature_sets(df)
        for fname in ("F0", "F_all"):
            for mname in ("LR", "Ridge", "ElasticNet", "SVM", "RF"):
                yp = fit_predict(df, y, tr, te, fsets[fname], mname)
                preds[(cond, fname, mname)] = yp
                st = rc.per_tree_stats(truth, yp)
                m = rc.metrics_from_stats(st)
                rows.append({"condition": cond, "features": fname, "model": mname,
                             "n_dim": len(fsets[fname]), **m})
                print(f"{cond:6s} {fname:6s} {mname:11s} "
                      f"macro={m['macro']*100:6.2f}%  joint={m['joint']*100:6.2f}%  "
                      f"mae={m['mae']:.4f}  totMAE={m['total_mae']:.3f}")

    table = pd.DataFrame(rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(rc.OUT_DIR / "a1_extended_metrics.csv", index=False)
    print(f"wrote {rc.OUT_DIR / 'a1_extended_metrics.csv'}")

    # ---- bootstrap CIs for the configurations quoted in the paper ----------
    quoted = [
        ("gt", "F0", "ElasticNet"), ("gt", "F0", "SVM"), ("gt", "F0", "Ridge"),
        ("gt", "F0", "LR"), ("gt", "F0", "RF"),
        ("fixed", "F0", "ElasticNet"), ("fixed", "F0", "Ridge"),
        ("fixed", "F0", "LR"), ("fixed", "F0", "SVM"), ("fixed", "F0", "RF"),
        ("fixed", "F_all", "Ridge"),
    ]
    cis = {}
    for key in quoted:
        st = rc.per_tree_stats(truth, preds[key])
        cis["|".join(key)] = rc.bootstrap_ci(st, CI_KEYS, n_boot=N_BOOT)
        c = cis["|".join(key)]
        print(f"CI {'|'.join(key):26s} macro={c['macro']['point']*100:6.2f}% "
              f"[{c['macro']['ci_lo']*100:.2f}, {c['macro']['ci_hi']*100:.2f}]  "
              f"joint={c['joint']['point']*100:6.2f}% "
              f"[{c['joint']['ci_lo']*100:.2f}, {c['joint']['ci_hi']*100:.2f}]")
    rc.dump(cis, rc.OUT_DIR / "a1_bootstrap_ci.json")

    # ---- paired comparisons ----------------------------------------------
    comparisons = [
        # headline gap
        (("gt", "F0", "ElasticNet"), ("fixed", "F_all", "Ridge"), "GT best vs fixed best"),
        (("gt", "F0", "ElasticNet"), ("fixed", "F0", "ElasticNet"), "ElasticNet: GT vs fixed"),
        # counter-vs-counter under the fixed detector, F0
        (("fixed", "F0", "ElasticNet"), ("fixed", "F0", "Ridge"), "fixed F0: EN vs Ridge"),
        (("fixed", "F0", "ElasticNet"), ("fixed", "F0", "LR"), "fixed F0: EN vs LR"),
        (("fixed", "F0", "ElasticNet"), ("fixed", "F0", "SVM"), "fixed F0: EN vs SVM"),
        (("fixed", "F0", "ElasticNet"), ("fixed", "F0", "RF"), "fixed F0: EN vs RF"),
        (("fixed", "F0", "LR"), ("fixed", "F0", "RF"), "fixed F0: LR vs RF"),
        # feature ablation gain
        (("fixed", "F_all", "Ridge"), ("fixed", "F0", "Ridge"), "fixed Ridge: F_all vs F0"),
        # counter families under GT
        (("gt", "F0", "ElasticNet"), ("gt", "F0", "RF"), "GT F0: EN vs RF"),
        (("gt", "F0", "ElasticNet"), ("gt", "F0", "SVM"), "GT F0: EN vs SVM"),
    ]
    out = {}
    for a, b, label in comparisons:
        sa = rc.per_tree_stats(truth, preds[a])
        sb = rc.per_tree_stats(truth, preds[b])
        entry = {
            "a": "|".join(a), "b": "|".join(b),
            "paired_bootstrap": rc.paired_bootstrap(sa, sb, DIFF_KEYS, n_boot=N_BOOT),
            "wilcoxon_macro_mae": rc.wilcoxon_macro_mae(sa, sb),
            "mcnemar_tree_pm1": rc.mcnemar_tree(sa, sb),
        }
        out[label] = entry
        pb = entry["paired_bootstrap"]["macro"]
        pj = entry["paired_bootstrap"]["joint"]
        print(f"{label:32s} dMacro={pb['diff']*100:+6.2f}pp "
              f"[{pb['ci_lo']*100:+.2f},{pb['ci_hi']*100:+.2f}] p={pb['p_boot']:.4f} | "
              f"dTree={pj['diff']*100:+6.2f}pp p={pj['p_boot']:.4f} | "
              f"McNemar p={entry['mcnemar_tree_pm1']['p']:.4f} | "
              f"Wilcoxon p={entry['wilcoxon_macro_mae']['p']:.4f}")
    rc.dump(out, rc.OUT_DIR / "a1_paired_tests.json")

    # ---- spread across the five counters (fixed, F0) ----------------------
    fixed_f0 = table[(table.condition == "fixed") & (table.features == "F0")]
    print("\nfixed/F0 spread: macro range = "
          f"{(fixed_f0.macro.max() - fixed_f0.macro.min())*100:.2f} pp, "
          f"joint range = {(fixed_f0.joint.max() - fixed_f0.joint.min())*100:.2f} pp")


if __name__ == "__main__":
    main()
