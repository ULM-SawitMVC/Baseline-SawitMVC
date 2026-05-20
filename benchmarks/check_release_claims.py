"""
Release consistency checks for stored benchmark artifacts.

This script verifies the small set of public headline claims that are easy to
make stale during release edits:
  - heuristic top-5 metrics from results/heuristics_953/accuracy_full.csv
  - canonical E2E best result from results/e2e_per_tree/*/metrics.json
  - canonical E2E test split size

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
                f"{method}: expected Acc+/-1={exp_acc}, MAE={exp_mae}; "
                f"found Acc+/-1={acc}, MAE={mae}"
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
            errors.append(f"{metrics_path.parent.name}: expected n_trees=95, found {n_trees}")
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
            f"{name} Acc+/-1={acc:.4f}, MAE={mae:.4f}, n={n_trees}; "
            f"expected {exp_name} Acc+/-1={exp_acc:.4f}, MAE={exp_mae:.4f}, n={exp_n}"
        )
    return errors


def main() -> None:
    errors = check_heuristics() + check_e2e()
    if errors:
        print("Release claim check failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("Release claim check passed.")


if __name__ == "__main__":
    main()
