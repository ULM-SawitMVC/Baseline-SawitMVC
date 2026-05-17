#!/usr/bin/env python3
"""
SawitMVC Baseline — Benchmark Runner
======================================
Evaluate the top-5 deduplication heuristics on the bundled SawitMVC-YOLO ground
truth and print a ranked results table.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --data /path/to/annotations/folder
    python benchmarks/run_benchmark.py --data ./ground_truth/annotations/ --save

The script expects JSON files produced by the SawitMVC annotation pipeline.
Each file corresponds to one tree and follows the schema in docs/dataset.md
and ground_truth/README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Allow running from repo root or benchmarks/ subdirectory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms import RANKING


CLASSES = ["B1", "B2", "B3", "B4"]
DEFAULT_JSON_DIR = ROOT / "ground_truth" / "annotations"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tree(json_path: Path) -> dict | None:
    """Load one tree JSON and return {"dets": list, "gt": dict}."""
    with open(json_path, encoding="utf-8-sig") as f:
        tree = json.load(f)

    # Ground truth: summary.by_class
    summary = tree.get("summary", {})
    by_class = summary.get("by_class", {})
    gt = {c: int(by_class.get(c, 0)) for c in CLASSES}

    # Detections: flatten all sides
    dets = []
    for side_key, side in tree.get("images", {}).items():
        side_index = side.get("side_index", 0)
        for ann in side.get("annotations", []):
            bbox = ann.get("bbox_yolo", [0.5, 0.5, 0.1, 0.1])
            dets.append({
                "class": ann.get("class_name", "B3"),
                "x_norm": float(bbox[0]),
                "y_norm": float(bbox[1]),
                "side_index": int(side_index),
            })

    return {"dets": dets, "gt": gt, "tree_id": tree.get("tree_id", json_path.stem)}


def load_dataset(json_dir: Path) -> list[dict]:
    """Load all JSON files from a directory."""
    paths = sorted(json_dir.glob("*.json"))
    if not paths:
        print(f"ERROR: No JSON files found in {json_dir}", file=sys.stderr)
        sys.exit(1)

    trees = []
    for p in paths:
        try:
            t = load_tree(p)
            if t is not None:
                trees.append(t)
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}", file=sys.stderr)

    return trees


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def within1(pred: dict, gt: dict) -> bool:
    """True if every class prediction is within ±1 of ground truth."""
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in CLASSES)


def class_mae(pred: dict, gt: dict) -> float:
    """Macro-averaged absolute error across the 4 classes."""
    return sum(abs(pred.get(c, 0) - gt.get(c, 0)) for c in CLASSES) / len(CLASSES)


def total_mae(pred: dict, gt: dict) -> float:
    """Absolute error of the total bunch count."""
    return abs(sum(pred.get(c, 0) for c in CLASSES) - sum(gt.get(c, 0) for c in CLASSES))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(trees: list[dict]) -> list[dict]:
    """Evaluate all top-5 algorithms and return sorted results."""
    n = len(trees)
    results = []

    for algo_name, meta in RANKING.items():
        predict_fn = meta["predict"]
        acc_count = 0
        mae_sum = 0.0
        total_mae_sum = 0.0

        for tree in trees:
            pred = predict_fn(tree["dets"])
            gt = tree["gt"]
            if within1(pred, gt):
                acc_count += 1
            mae_sum += class_mae(pred, gt)
            total_mae_sum += total_mae(pred, gt)

        results.append({
            "rank": meta["rank"],
            "algorithm": algo_name,
            "acc1_pct": acc_count / n * 100,
            "macro_mae": mae_sum / n,
            "total_count_mae": total_mae_sum / n,
            "n_fail": n - acc_count,
        })

    results.sort(key=lambda r: (-r["acc1_pct"], r["macro_mae"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def print_table(results: list[dict], n_trees: int) -> None:
    """Print a formatted results table."""
    header = f"\nSawitMVC Baseline - Benchmark Results ({n_trees} trees)"
    print(header)
    print("=" * len(header))
    print(f"{'Rank':<5} {'Algorithm':<30} {'Acc+/-1':>8}  {'MAE':>6}  {'Total MAE':>10}  {'Fail':>5}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['rank']:>4}  "
            f"{r['algorithm']:<30} "
            f"{r['acc1_pct']:>7.2f}%  "
            f"{r['macro_mae']:>6.4f}  "
            f"{r['total_count_mae']:>10.4f}  "
            f"{r['n_fail']:>5}"
        )
    print("-" * 70)
    print(f"      {'Naive sum (reference)':<30} {'3.78%':>7}   {'2.2867':>6}  {'9.1469':>10}")
    print()


def save_csv(results: list[dict], out_dir: Path) -> None:
    """Save results to CSV at results/heuristics_953/benchmark_top5.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark_top5.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "algorithm", "acc1_pct", "macro_mae", "total_count_mae", "n_fail"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SawitMVC Baseline Benchmark")
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help="Folder of ground-truth annotation JSONs (default: ./ground_truth/annotations/)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save results CSV to results/heuristics_953/benchmark_top5.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(
            f"ERROR: Annotation directory not found: {args.data}\n"
            "The repository bundles annotations at ./ground_truth/annotations/.\n"
            "If you have copied or symlinked the Hugging Face release, pass --data with its path.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset from {args.data} ...")
    trees = load_dataset(args.data)
    print(f"Loaded {len(trees)} trees.")

    print("Running top-5 algorithms ...")
    results = run_benchmark(trees)

    print_table(results, len(trees))

    if args.save:
        out_dir = ROOT / "results" / "heuristics_953"
        save_csv(results, out_dir)


if __name__ == "__main__":
    main()
