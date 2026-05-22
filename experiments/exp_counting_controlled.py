"""
Controlled counting model comparison on cached YOLO tree predictions.

This experiment keeps the train/test split and feature set fixed while changing
the counter model. It writes 80 rows:
8 feature sets x 5 models x 2 training strategies.

Main strategy:
  train_only: train on 716 training trees, test on 141 test trees.

Additional strategy:
  train_val: train on 716 training + 96 validation trees, test on the same
  141 test trees. This is useful for detailed reporting, but the headline
  comparison should use train_only.

Usage:
  python experiments/exp_counting_controlled.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from build_counting_features import CLASSES, _load_gt, _load_splits  # noqa: E402


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

INFERENCE_DIR = ROOT / "predictions" / "y26mv2_per_tree"
GT_DIR = ROOT / "ground_truth" / "annotations"
OUT_DIR = ROOT / "results" / "experiments"
OUT_PATH = OUT_DIR / "counting_controlled_results.csv"


def extract_all_features(tree_json: dict) -> dict[str, float]:
    """Extract the same 67-dim feature bank used by exp_counting_v3."""
    sides = tree_json.get("images", {})
    n_sides = max(len(sides), 1)
    per_side_counts = {c: [] for c in CLASSES}
    confidences = {c: [] for c in CLASSES}
    areas = {c: [] for c in CLASSES}
    center_y = {c: [] for c in CLASSES}

    for side_data in sides.values():
        counts = {c: 0 for c in CLASSES}
        for ann in side_data.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls not in CLASSES:
                continue
            conf = float(ann.get("conf", 1.0))
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            confidences[cls].append(conf)
            areas[cls].append(float(bbox[2]) * float(bbox[3]))
            center_y[cls].append(float(bbox[1]))
            counts[cls] += 1
        for c in CLASSES:
            per_side_counts[c].append(counts[c])

    features: dict[str, float] = {}
    for c in CLASSES:
        ps = np.array(per_side_counts[c], dtype=float)
        conf = np.array(confidences[c])
        area = np.array(areas[c])
        cy = np.array(center_y[c])
        n = len(conf)

        features[f"naive_sum_{c}"] = float(ps.sum())
        features[f"max_per_side_{c}"] = float(ps.max())
        features[f"mean_per_side_{c}"] = float(ps.mean())

        features[f"std_per_side_{c}"] = float(ps.std())
        features[f"min_per_side_{c}"] = float(ps.min())
        features[f"cv_per_side_{c}"] = float(ps.std() / (ps.mean() + 1e-6))
        features[f"n_sides_det_{c}"] = float((ps > 0).sum())
        features[f"consistency_{c}"] = float(1.0 / (1.0 + ps.std()))

        features[f"conf_sum_{c}"] = float(conf.sum())
        features[f"conf_mean_{c}"] = float(conf.mean()) if n > 0 else 0.0
        features[f"conf_max_{c}"] = float(conf.max()) if n > 0 else 0.0
        features[f"high_conf_{c}"] = float((conf >= 0.5).sum())
        features[f"vhigh_conf_{c}"] = float((conf >= 0.6).sum())

        features[f"mean_cy_{c}"] = float(cy.mean()) if n > 0 else 0.5
        features[f"mean_area_{c}"] = float(area.mean()) if n > 0 else 0.0

    total = sum(features[f"naive_sum_{c}"] for c in CLASSES)
    features["n_sides"] = float(n_sides)
    features["total_naive"] = float(total)
    for c in CLASSES:
        features[f"frac_{c}"] = features[f"naive_sum_{c}"] / (total + 1e-6)
    features["b3_b23_frac"] = (
        features["naive_sum_B3"]
        / (features["naive_sum_B2"] + features["naive_sum_B3"] + 1e-6)
    )
    return features


def load_dataset(inference_dir: Path, gt_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    splits_map = _load_splits(gt_dir.parent)
    gt_map = _load_gt(gt_dir)
    rows: list[dict[str, float]] = []
    labels: list[list[int]] = []
    tree_ids: list[str] = []
    tree_splits: list[str] = []

    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            data = json.load(f)
        tree_id = data.get("tree_name") or data.get("tree_id") or fp.stem
        if tree_id not in gt_map:
            continue
        rows.append(extract_all_features(data))
        labels.append([gt_map[tree_id].get(c, 0) for c in CLASSES])
        tree_ids.append(tree_id)
        tree_splits.append(splits_map.get(tree_id, data.get("split", "train")))

    return pd.DataFrame(rows), np.array(labels, dtype=float), tree_ids, np.array(tree_splits)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    f0_cols = (
        [f"naive_sum_{c}" for c in CLASSES]
        + [f"max_per_side_{c}" for c in CLASSES]
        + [f"mean_per_side_{c}" for c in CLASSES]
        + ["n_sides"]
    )
    conf_cols = (
        [f"conf_sum_{c}" for c in CLASSES]
        + [f"conf_mean_{c}" for c in CLASSES]
        + [f"conf_max_{c}" for c in CLASSES]
        + [f"high_conf_{c}" for c in CLASSES]
        + [f"vhigh_conf_{c}" for c in CLASSES]
    )
    spatial_cols = [f"mean_cy_{c}" for c in CLASSES] + [f"mean_area_{c}" for c in CLASSES]
    distrib_cols = (
        [f"std_per_side_{c}" for c in CLASSES]
        + [f"min_per_side_{c}" for c in CLASSES]
        + [f"cv_per_side_{c}" for c in CLASSES]
        + [f"n_sides_det_{c}" for c in CLASSES]
        + [f"consistency_{c}" for c in CLASSES]
    )

    return {
        "F0": f0_cols,
        "F0+conf": f0_cols + conf_cols,
        "F0+spatial": f0_cols + spatial_cols,
        "F0+distrib": f0_cols + distrib_cols,
        "F0+conf+spatial": f0_cols + conf_cols + spatial_cols,
        "F0+conf+distrib": f0_cols + conf_cols + distrib_cols,
        "F0+distrib+spatial": f0_cols + distrib_cols + spatial_cols,
        "F_all": df.columns.tolist(),
    }


def model_builders() -> dict[str, Callable[[], object]]:
    return {
        "LR": lambda: Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "SVM": lambda: GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", MultiOutputRegressor(SVR(kernel="rbf"))),
                ]
            ),
            {
                "svr__estimator__C": [0.1, 1, 10],
                "svr__estimator__gamma": ["scale", 0.01, 0.1],
            },
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        ),
        "RF": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Ridge": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500])),
            ]
        ),
        "ElasticNet": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    MultiTaskElasticNetCV(cv=5, random_state=SEED, max_iter=3000),
                ),
            ]
        ),
    }


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.round(y_pred), 0, None).astype(int)
    truth = y_true.astype(int)
    metrics: dict[str, float] = {}

    for j, c in enumerate(CLASSES):
        err = np.abs(rounded[:, j] - truth[:, j])
        metrics[f"acc_{c}"] = float(np.mean(err <= 1))
        metrics[f"mae_{c}"] = float(np.mean(err))
        metrics[f"bias_{c}"] = float(np.mean(rounded[:, j] - truth[:, j]))

    metrics["macro"] = float(np.mean([metrics[f"acc_{c}"] for c in CLASSES]))
    metrics["joint"] = float(np.mean(np.all(np.abs(rounded - truth) <= 1, axis=1)))
    metrics["mae"] = float(np.mean([metrics[f"mae_{c}"] for c in CLASSES]))
    return metrics


def main() -> None:
    print("Loading cached YOLO predictions and ground truth ...")
    df, y, _, splits = load_dataset(INFERENCE_DIR, GT_DIR)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    print(
        f"Trees: train={train_mask.sum()} | val={val_mask.sum()} | "
        f"test={test_mask.sum()} | total={len(splits)}"
    )

    if int(train_mask.sum()) != 716 or int(val_mask.sum()) != 96 or int(test_mask.sum()) != 141:
        raise RuntimeError("Unexpected split counts; expected 716 train, 96 val, 141 test.")

    feature_map = feature_sets(df)
    builders = model_builders()
    strategies = {
        "train_only": train_mask,
        "train_val": train_mask | val_mask,
    }

    rows: list[dict[str, float | int | str]] = []
    y_test = y[test_mask]
    print(
        f"\n{'Features':<22} {'Model':<10} {'Strategy':<11} "
        f"{'Dims':>4} {'Macro':>8} {'Joint':>8} {'MAE':>7}"
    )
    print("-" * 78)

    for feature_name, columns in feature_map.items():
        x_all = df[columns].values.astype(float)
        x_test = x_all[test_mask]

        for strategy_name, mask in strategies.items():
            x_train = x_all[mask]
            y_train = y[mask]

            for model_name, make_model in builders.items():
                model = make_model()
                model.fit(x_train, y_train)
                metrics = score(y_test, model.predict(x_test))
                row: dict[str, float | int | str] = {
                    "features": feature_name,
                    "model": model_name,
                    "strategy": strategy_name,
                    "n_dim": len(columns),
                    "train_trees": int(mask.sum()),
                    "test_trees": int(test_mask.sum()),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{feature_name:<22} {model_name:<10} {strategy_name:<11} "
                    f"{len(columns):>4} {metrics['macro']*100:7.2f}% "
                    f"{metrics['joint']*100:7.2f}% {metrics['mae']:7.4f}"
                )

    result_df = pd.DataFrame(rows)
    ordered_cols = (
        ["features", "model", "strategy", "n_dim", "train_trees", "test_trees", "macro", "joint", "mae"]
        + [f"acc_{c}" for c in CLASSES]
        + [f"mae_{c}" for c in CLASSES]
        + [f"bias_{c}" for c in CLASSES]
    )
    result_df = result_df[ordered_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(result_df)} rows to {OUT_PATH}")
    print("\nTop 10 train_only configurations:")
    top = (
        result_df[result_df["strategy"] == "train_only"]
        .sort_values(["macro", "joint", "mae"], ascending=[False, False, True])
        .head(10)
    )
    print(top[["features", "model", "n_dim", "macro", "joint", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
