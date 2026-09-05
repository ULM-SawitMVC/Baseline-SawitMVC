"""Audit manuscript claims with exact paired tests and matched resampling.

Run: python experiments/revision/a10_revision_audit.py

Existing a1--a9 outputs remain historical records. This script adds all ten
counter pairs per condition, all seven Ridge ablations against F0, and matched
GT/fixed Ridge+F0 CV. Models use the original builders, preserving the baseline.
Exact sign-flip tests exchange the two predictions within each tree, keeping
its four class outcomes together. Dynamic programming enumerates the null
distribution of the integer number of correct classes without simulation.
See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
for the paired sample-exchange null hypothesis. Bootstrap CIs are conditional
on fitted configurations; they do not correct post-hoc model selection.
"""
from __future__ import annotations

import json
import platform
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold

import revcommon as rc


def exact_class_test(a, b):
    delta = (a["ok"].sum(axis=1) - b["ok"].sum(axis=1)).astype(int)
    distribution = Counter({0: 1})
    for value in np.abs(delta[delta != 0]):
        next_distribution = Counter()
        for total, ways in distribution.items():
            next_distribution[total + int(value)] += ways
            next_distribution[total - int(value)] += ways
        distribution = next_distribution
    observed = int(delta.sum())
    extreme = sum(ways for total, ways in distribution.items()
                  if abs(total) >= abs(observed))
    return extreme / sum(distribution.values())


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    adjusted = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    result[order] = adjusted
    return result


def comparison(a, b):
    delta = a["ok"].mean(axis=1) - b["ok"].mean(axis=1)
    boot = delta[rc.bootstrap_indices(len(delta))].mean(axis=1)
    return {"difference_pp": float(delta.mean() * 100),
            "ci_lo_pp": float(np.quantile(boot, .025) * 100),
            "ci_hi_pp": float(np.quantile(boot, .975) * 100),
            "class_p_exact": exact_class_test(a, b),
            "tree_p_mcnemar": rc.mcnemar_tree(a, b)["p"],
            "mae_p_wilcoxon": rc.wilcoxon_macro_mae(a, b)["p"]}


def main():
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, stats, prediction_rows = {}, {}, []
    metric_rows = []
    original = pd.read_csv(rc.OUT_DIR / "a1_extended_metrics.csv")
    models = ["LR", "Ridge", "ElasticNet", "SVM", "RF"]
    for condition, source in [("gt", rc.GT_DIR), ("fixed", rc.PRED_DIR)]:
        df, y, ids, splits, _ = rc.load_dataset(source)
        data[condition] = (df, y, ids, splits)
        tr, te = splits == "train", splits == "test"
        fsets = rc.feature_sets(df)
        configs = [("F0", model) for model in models]
        if condition == "fixed":
            configs += [(features, "Ridge") for features in fsets if features != "F0"]
        for features, model_name in configs:
            x = df[fsets[features]].to_numpy(float)
            model = rc.model_builders()[model_name]()
            model.fit(x[tr], y[tr])
            prediction = rc.round_counts(model.predict(x[te]))
            stat = rc.per_tree_stats(y[te], prediction)
            stats[(condition, features, model_name)] = stat
            metrics = rc.metrics_from_stats(stat)
            ref = original[(original.condition == condition)
                           & (original.features == features) & (original.model == model_name)]
            if not ref.empty:
                for key in ("macro", "joint", "mae", "rmse", "total_mae"):
                    assert np.isclose(metrics[key], ref.iloc[0][key], atol=1e-12), (condition, features, model_name, key)
            metric_rows.append(dict(condition=condition, features=features, model=model_name, **metrics))
            for tid, truth, pred in zip(np.array(ids)[te], y[te], prediction):
                prediction_rows.append(dict(condition=condition, features=features, model=model_name,
                    tree_id=tid, **{f"true_{c}": int(v) for c, v in zip(rc.CLASSES, truth)},
                    **{f"pred_{c}": int(v) for c, v in zip(rc.CLASSES, pred)}))
            print(condition, features, model_name, round(metrics["macro"] * 100, 4), flush=True)
    assert data["gt"][2] == data["fixed"][2]
    assert np.array_equal(data["gt"][1], data["fixed"][1])
    assert np.array_equal(data["gt"][3], data["fixed"][3])
    pd.DataFrame(prediction_rows).to_csv(rc.OUT_DIR / "a10_test_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(rc.OUT_DIR / "a10_verified_metrics.csv", index=False)

    tests = []
    for condition in ("gt", "fixed"):
        family = []
        for a, b in combinations(models, 2):
            family.append(dict(family=f"{condition}_counters", a=a, b=b,
                **comparison(stats[(condition, "F0", a)], stats[(condition, "F0", b)])))
        for metric in ("class_p_exact", "tree_p_mcnemar", "mae_p_wilcoxon"):
            for row, p in zip(family, holm([row[metric] for row in family])):
                row[metric + "_holm"] = p
        tests.extend(family)
    family = []
    for features in rc.feature_sets(data["fixed"][0]):
        if features != "F0":
            family.append(dict(family="fixed_features", a=features, b="F0",
                **comparison(stats[("fixed", features, "Ridge")], stats[("fixed", "F0", "Ridge")])) )
    for metric in ("class_p_exact", "tree_p_mcnemar", "mae_p_wilcoxon"):
        for row, p in zip(family, holm([row[metric] for row in family])):
            row[metric + "_holm"] = p
    tests.extend(family)
    for a, b, label in [(("gt", "F0", "ElasticNet"), ("fixed", "F_all", "Ridge"), "reported_best"),
                         (("gt", "F0", "Ridge"), ("fixed", "F0", "Ridge"), "same_Ridge_F0"),
                         (("gt", "F0", "ElasticNet"), ("fixed", "F0", "ElasticNet"), "same_EN_F0")]:
        tests.append(dict(family="condition_gap", a=label, b="fixed", **comparison(stats[a], stats[b])))
    test_df = pd.DataFrame(tests)
    test_df.to_csv(rc.OUT_DIR / "a10_exact_paired_tests.csv", index=False)
    print(test_df[["family", "a", "b", "difference_pp", "class_p_exact", "class_p_exact_holm"]].to_string(index=False), flush=True)

    # Identical tree IDs, folds, targets, model, feature family and train sizes.
    manifest = pd.read_csv(rc.GT_DIR.parent / "split_manifest.csv", encoding="utf-8-sig").set_index("tree_id")
    ids, splits = data["gt"][2:]
    strat = manifest.loc[ids, "strat_key"].to_numpy()
    folds, predictions = [], []
    for pool_name, pool in [("nontraining237", np.isin(splits, ["val", "test"])),
                            ("testonly141", splits == "test")]:
        pool_idx = np.flatnonzero(pool)
        for repeat in range(5):
            cv = StratifiedKFold(5, shuffle=True, random_state=42 + repeat)
            for fold, (train, test) in enumerate(cv.split(pool_idx, strat[pool])):
                tr, te = pool_idx[train], pool_idx[test]
                scores = {}
                for condition in ("gt", "fixed"):
                    df, y, tree_ids, _ = data[condition]
                    x = df[rc.feature_sets(df)["F0"]].to_numpy(float)
                    model = rc.model_builders()["Ridge"]()
                    model.fit(x[tr], y[tr])
                    pred = rc.round_counts(model.predict(x[te]))
                    scores[condition] = rc.score(y[te], pred)["macro"]
                    for tid, true, predicted in zip(np.array(tree_ids)[te], y[te], pred):
                        predictions.append(dict(pool=pool_name, repeat=repeat, fold=fold,
                            condition=condition, tree_id=tid,
                            **{f"true_{c}": int(v) for c, v in zip(rc.CLASSES, true)},
                            **{f"pred_{c}": int(v) for c, v in zip(rc.CLASSES, predicted)}))
                folds.append(dict(pool=pool_name, repeat=repeat, fold=fold, n_train=len(tr), n_test=len(te),
                    gt=scores["gt"], fixed=scores["fixed"], gap=scores["gt"]-scores["fixed"]))
        print("Matched CV completed:", pool_name, flush=True)
    fold_df = pd.DataFrame(folds)
    fold_df.to_csv(rc.OUT_DIR / "a10_matched_cv_folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(rc.OUT_DIR / "a10_matched_cv_predictions.csv", index=False)
    summary = {pool: {metric: {"mean_pct": float(part[metric].mean()*100),
                              "std_pct": float(part[metric].std(ddof=1)*100)}
                      for metric in ("gt", "fixed", "gap")}
               for pool, part in fold_df.groupby("pool")}
    summary["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    summary["caveats"] = ["Fold SD is descriptive; repeated folds overlap.",
        "237 pool includes detector validation trees; 141 pool excludes them.",
        "Detector is frozen, not retrained in outer folds.",
        "Historical model builders standardize before internal Ridge/ElasticNet CV; outer fold test trees are excluded.",
        "Reported-best configurations were highlighted after comparing test results; inference is conditional and exploratory."]
    rc.dump(summary, rc.OUT_DIR / "a10_audit_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
