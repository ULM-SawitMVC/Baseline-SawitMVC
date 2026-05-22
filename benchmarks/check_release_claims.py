"""
Release consistency checks for stored benchmark artifacts.

This script verifies the small set of public headline claims that are easy to
make stale during release edits:
  - heuristic top-5 metrics from results/heuristics_953/accuracy_full.csv
  - canonical baseline E2E best result from results/e2e_per_tree/*/metrics.json
  - v3 experiment winner from results/experiments/counting_v3_results.csv
  - v4 deep-probe ceiling from results/experiments/counting_v4_results.csv

It does not parse prose tables; use it as a guardrail before updating README/docs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_HEURISTICS = {
    "M01_selector_b2b3": (87.62, 0.3746),
    "M02_selector_trifurc": (87.62, 0.3757),
    "M03_blend_geometric": (86.99, 0.3767),
    "M04_blend_floor_clamped": (86.99, 0.3848),
    "M05_blend_vis_divide": (86.99, 0.3875),
}
EXPECTED_E2E_BEST = ("y26mv2_lr", 0.7571, 1.0479, 141)
EXPECTED_V3_BEST = ("F_all", "Ridge", "train_only", 67, 0.7748, 0.3262, 1.0355)
EXPECTED_V4_TOP = ("ElasticNet + F0+spatial (train)", 0.7677, 0.3121, 1.0390)
EXPECTED_V4_CEILING = 0.7748
EXPECTED_TRACK_C_BEST = ("gt_elasticnet", 0.9805, 0.9220, 0.2766, 141)


def _close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def check_heuristics() -> list[str]:
    path = ROOT / "results" / "heuristics_953" / "accuracy_full.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}

    errors: list[str] = []
    for method, (exp_acc, exp_mae) in EXPECTED_HEURISTICS.items():
        row = rows.get(method)
        if row is None:
            errors.append(f"Missing heuristic row: {method}")
            continue
        acc = float(row["acc_within1_pct"])
        mae = float(row["MAE"])
        if not _close(acc, exp_acc, 0.005) or not _close(mae, exp_mae):
            errors.append(
                f"{method}: expected Class ±1 Acc={exp_acc}, MAE={exp_mae}; "
                f"found Class ±1 Acc={acc}, MAE={mae}"
            )
    return errors


def check_e2e() -> list[str]:
    e2e_dir = ROOT / "results" / "e2e_per_tree"
    best: tuple[float, str, dict] | None = None
    errors: list[str] = []

    for metrics_path in sorted(e2e_dir.glob("y26mv2_*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = data.get("test", {})
        n_trees = int(test.get("n_trees", 0))
        if n_trees != 141:
            errors.append(f"{metrics_path.parent.name}: expected n_trees=141, found {n_trees}")
        acc = float(test.get("macro_acc_pm1", 0.0))
        if best is None or acc > best[0]:
            best = (acc, metrics_path.parent.name, test)

    if best is None:
        errors.append("No E2E metrics found")
        return errors

    exp_name, exp_acc, exp_mae, exp_n = EXPECTED_E2E_BEST
    acc, name, test = best
    mae = float(test["macro_class_mae"])
    n_trees = int(test["n_trees"])
    if name != exp_name or not _close(acc, exp_acc) or not _close(mae, exp_mae) or n_trees != exp_n:
        errors.append(
            "Unexpected E2E best: "
            f"{name} Class ±1 Acc={acc:.4f}, MAE={mae:.4f}, n={n_trees}; "
            f"expected {exp_name} Class ±1 Acc={exp_acc:.4f}, MAE={exp_mae:.4f}, n={exp_n}"
        )
    return errors


def check_v3_experiments() -> list[str]:
    path = ROOT / "results" / "experiments" / "counting_v3_results.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return [f"No rows found in {path}"]

    best = max(rows, key=lambda r: float(r["macro"]))
    exp_features, exp_model, exp_strategy, exp_dims, exp_acc, exp_joint, exp_mae = EXPECTED_V3_BEST
    errors: list[str] = []
    if (
        best["features"] != exp_features
        or best["model"] != exp_model
        or best["strategy"] != exp_strategy
        or int(best["n_dim"]) != exp_dims
        or not _close(float(best["macro"]), exp_acc, 5e-4)
        or not _close(float(best["joint"]), exp_joint, 5e-4)
        or not _close(float(best["mae"]), exp_mae, 5e-4)
    ):
        errors.append(
            "Unexpected v3 best: "
            f"{best['features']} + {best['model']} ({best['strategy']}, {best['n_dim']} dim) "
            f"Class ±1 Acc={float(best['macro']):.4f}, Tree ±1 Acc={float(best['joint']):.4f}, "
            f"MAE={float(best['mae']):.4f}"
        )
    return errors


def check_v4_experiments() -> list[str]:
    path = ROOT / "results" / "experiments" / "counting_v4_results.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return [f"No rows found in {path}"]

    best = max(rows, key=lambda r: float(r["macro_acc"]))
    exp_name, exp_acc, exp_joint, exp_mae = EXPECTED_V4_TOP
    errors: list[str] = []
    if (
        best["config"] != exp_name
        or not _close(float(best["macro_acc"]), exp_acc, 5e-4)
        or not _close(float(best["joint_acc"]), exp_joint, 5e-4)
        or not _close(float(best["macro_mae"]), exp_mae, 5e-4)
    ):
        errors.append(
            "Unexpected v4 top row: "
            f"{best['config']} Class ±1 Acc={float(best['macro_acc']):.4f}, "
            f"Tree ±1 Acc={float(best['joint_acc']):.4f}, MAE={float(best['macro_mae']):.4f}"
        )

    if float(best["macro_acc"]) >= EXPECTED_V4_CEILING:
        errors.append(
            "Unexpected v4 ceiling break: "
            f"best v4 Class ±1 Acc={float(best['macro_acc']):.4f} should stay below {EXPECTED_V4_CEILING:.4f}"
        )
    return errors


def check_track_c() -> list[str]:
    root = ROOT / "results" / "e2e_upper_bound"
    best: tuple[float, str, float, float, int] | None = None
    errors: list[str] = []

    for metrics_path in sorted(root.glob("gt_*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = data.get("test", {})
        pred_path = metrics_path.parent / "predictions.csv"
        with open(pred_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        n = len(rows)
        joint = 0.0
        for row in rows:
            ok = True
            for c in ("B1", "B2", "B3", "B4"):
                if abs(int(row[f"pred_{c}"]) - int(row[f"gt_{c}"])) > 1:
                    ok = False
                    break
            joint += 1.0 if ok else 0.0
        joint /= n if n else 1.0

        macro = float(test.get("macro_acc_pm1", 0.0))
        mae = float(test.get("macro_class_mae", 0.0))
        n_trees = int(test.get("n_trees", 0))
        item = (macro, metrics_path.parent.name, joint, mae, n_trees)
        if best is None or item[0] > best[0]:
            best = item

    if best is None:
        return ["No Track C metrics found"]

    exp_name, exp_acc, exp_joint, exp_mae, exp_n = EXPECTED_TRACK_C_BEST
    acc, name, joint, mae, n_trees = best
    if (
        name != exp_name
        or not _close(acc, exp_acc, 5e-4)
        or not _close(joint, exp_joint, 5e-4)
        or not _close(mae, exp_mae, 5e-4)
        or n_trees != exp_n
    ):
        errors.append(
            "Unexpected Track C best: "
            f"{name} Class ±1 Acc={acc:.4f}, Tree ±1 Acc={joint:.4f}, MAE={mae:.4f}, n={n_trees}; "
            f"expected {exp_name} Class ±1 Acc={exp_acc:.4f}, Tree ±1 Acc={exp_joint:.4f}, "
            f"MAE={exp_mae:.4f}, n={exp_n}"
        )
    return errors


def main() -> None:
    errors = (
        check_heuristics()
        + check_e2e()
        + check_v3_experiments()
        + check_v4_experiments()
        + check_track_c()
    )
    if errors:
        print("Release claim check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("Release claim check passed.")


if __name__ == "__main__":
    main()
