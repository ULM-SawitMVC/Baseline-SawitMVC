"""
Feature extraction library — 13-dim per tree.

Library: from pipeline.build_counting_features import load_dataset, extract_features
CLI:     python pipeline/build_counting_features.py --inference-dir predictions/y26mv2_per_tree/

Feature layout (13 dims): naive_sum_{B1..B4}, max_per_side_{B1..B4},
mean_per_side_{B1..B4}, n_sides.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Optional
import numpy as np

CLASSES = ["B1", "B2", "B3", "B4"]
FEATURE_NAMES = (
    [f"naive_sum_{c}"    for c in CLASSES]   # 4: total detections per class
    + [f"max_per_side_{c}" for c in CLASSES] # 4: max on any single side
    + [f"mean_per_side_{c}" for c in CLASSES]# 4: mean per side
    + ["n_sides"]                             # 1: 4 or 8
)  # Total: 13


def extract_features(tree_json: dict) -> np.ndarray:
    """13-dim feature vector from one tree JSON (GT or inference format)."""
    sides = tree_json.get("images", {})
    n_sides = len(sides)
    per_side: dict[str, list[int]] = {c: [] for c in CLASSES}
    for side_data in sides.values():
        counts = {c: 0 for c in CLASSES}
        for ann in side_data.get("annotations", []):
            cls = ann.get("class_name") or ann.get("class", "")
            if cls in counts:
                counts[cls] += 1
        for c in CLASSES:
            per_side[c].append(counts[c])
    feats: list[float] = []
    for c in CLASSES:
        feats.append(float(sum(per_side[c])))
    for c in CLASSES:
        feats.append(float(max(per_side[c])) if per_side[c] else 0.0)
    for c in CLASSES:
        arr = per_side[c]
        feats.append(float(sum(arr) / len(arr)) if arr else 0.0)
    feats.append(float(n_sides))
    return np.array(feats, dtype=np.float32)


def _load_splits(data_dir: Path) -> dict[str, str]:
    manifest = data_dir / "split_manifest.csv"
    splits: dict[str, str] = {}
    if not manifest.exists():
        return splits
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("tree_id", "")
            sp = row.get("new_split") or row.get("split", "train")
            if tid:
                splits[tid] = sp
    return splits


def _load_gt(gt_dir: Path) -> dict[str, dict[str, int]]:
    gt: dict[str, dict[str, int]] = {}
    for fp in sorted(gt_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            d = json.load(f)
        tid = d.get("tree_id") or d.get("tree_name") or fp.stem
        by_class = d.get("summary", {}).get("by_class", {})
        gt[tid] = {c: int(by_class.get(c, 0)) for c in CLASSES}
    return gt


def load_dataset(
    inference_dir: Path,
    gt_dir: Path,
    data_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load trees from inference JSONs + GT → (X, y, tree_ids, splits).

    Args:
        inference_dir: Folder of prediction JSONs (one per tree).
        gt_dir:        Ground-truth JSON folder (e.g. ground_truth/annotations/).
        data_dir:      Folder containing split_manifest.csv (defaults to gt_dir parent).
    Returns:
        X (n,13), y (n,4), tree_ids, splits
    """
    if data_dir is None:
        data_dir = gt_dir.parent
    splits_map = _load_splits(data_dir)
    gt_map = _load_gt(gt_dir)
    X, y, tree_ids, tree_splits = [], [], [], []
    for fp in sorted(inference_dir.glob("*.json")):
        with open(fp, encoding="utf-8-sig") as f:
            d = json.load(f)
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map:
            continue
        feats = extract_features(d)
        label = np.array([gt_map[tid].get(c, 0) for c in CLASSES], dtype=np.float32)
        split = splits_map.get(tid, d.get("split", "train"))
        X.append(feats); y.append(label); tree_ids.append(tid); tree_splits.append(split)
    return np.array(X), np.array(y), tree_ids, tree_splits


if __name__ == "__main__":
    import argparse
    ROOT = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Extract 13-dim counting features")
    p.add_argument("--inference-dir", type=Path,
                   default=ROOT / "predictions" / "y26mv2_per_tree")
    p.add_argument("--gt-dir", type=Path, default=ROOT / "ground_truth" / "annotations")
    args = p.parse_args()
    X, y, ids, splits = load_dataset(args.inference_dir, args.gt_dir)
    from collections import Counter
    print(f"Loaded {len(ids)} trees | X: {X.shape} | y: {y.shape}")
    print("Feature names:", FEATURE_NAMES)
    print("Split distribution:", Counter(splits))
