"""Generate publication-oriented figures for the paper draft."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt


def pct(x: float) -> float:
    return 100.0 * float(x)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def figure_split_distribution() -> None:
    rows = []
    with (ROOT / "ground_truth" / "split_manifest.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    df = pd.DataFrame(rows)
    class_cols = ["B1", "B2", "B3", "B4"]
    for c in class_cols:
        df[c] = df[c].astype(int)

    grouped = df.groupby("new_split")[class_cols].sum().loc[["train", "val", "test"]]
    labels = ["Train", "Validation", "Test"]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bottom = np.zeros(len(grouped))
    colors = ["#2F6B4F", "#5B8EBC", "#D59F3A", "#9B4F5F"]
    for cls, color in zip(class_cols, colors):
        vals = grouped[cls].to_numpy()
        ax.bar(labels, vals, bottom=bottom, label=cls, color=color, width=0.58)
        bottom += vals

    ax.set_ylabel("Unique bunches")
    ax.set_title("Unique FFB class distribution by tree-level split")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig01_split_class_distribution")


def figure_end_to_end_gap() -> None:
    controlled = pd.read_csv(ROOT / "results" / "experiments" / "counting_controlled_results.csv")
    ridge_fall = controlled[
        (controlled["features"] == "F_all")
        & (controlled["model"] == "Ridge")
        & (controlled["strategy"] == "train_only")
    ].iloc[0]
    gt_elastic = load_json(ROOT / "results" / "e2e_upper_bound" / "gt_elasticnet" / "metrics.json")["test"]
    gt_elastic_pred = pd.read_csv(ROOT / "results" / "e2e_upper_bound" / "gt_elasticnet" / "predictions.csv")
    gt_tree_pm1 = (
        (gt_elastic_pred[["pred_B1", "pred_B2", "pred_B3", "pred_B4"]].to_numpy()
        - gt_elastic_pred[["gt_B1", "gt_B2", "gt_B3", "gt_B4"]].to_numpy())
        .astype(float)
    )
    gt_tree_pm1 = float((np.abs(gt_tree_pm1) <= 1).all(axis=1).mean() * 100.0)

    labels = [
        "Naive GT\nappearance sum",
        "Visibility-adaptive\nGT divisor",
        "ElasticNet\nperfect detection",
        "YOLO + Ridge\nend-to-end",
    ]
    class_acc = [50.00, 95.92, pct(gt_elastic["macro_acc_pm1"]), pct(ridge_fall["macro"])]
    tree_acc = [6.38, 87.23, gt_tree_pm1, pct(ridge_fall["joint"])]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - width / 2, class_acc, width, label="Class ±1 Acc", color="#2F6B4F")
    ax.bar(x + width / 2, tree_acc, width, label="Tree ±1 Acc", color="#5B8EBC")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Counting performance gap between perfect and YOLO detections")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig02_detection_gap")


def figure_controlled_matrix() -> None:
    df = pd.read_csv(ROOT / "results" / "experiments" / "counting_controlled_results.csv")
    df = df[df["strategy"] == "train_only"].copy()
    feature_order = [
        "F0",
        "F0+conf",
        "F0+spatial",
        "F0+distrib",
        "F0+conf+spatial",
        "F0+conf+distrib",
        "F0+distrib+spatial",
        "F_all",
    ]
    model_order = ["LR", "SVM", "RF", "Ridge", "ElasticNet"]
    mat = (
        df.pivot(index="features", columns="model", values="macro")
        .loc[feature_order, model_order]
        .to_numpy()
        * 100.0
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=72.0, vmax=78.0, aspect="auto")
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order)
    ax.set_yticks(np.arange(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_title("Controlled train-only matrix: Class ±1 Acc")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="#111111", fontsize=7)

    ax.scatter([3], [7], s=220, facecolors="none", edgecolors="#B21E35", linewidths=2.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class ±1 Acc (%)")
    save(fig, "fig03_controlled_matrix")


def main() -> None:
    set_style()
    figure_split_distribution()
    figure_end_to_end_gap()
    figure_controlled_matrix()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
