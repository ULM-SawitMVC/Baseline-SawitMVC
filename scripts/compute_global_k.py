#!/usr/bin/env python3
"""Compute global vs class-wise correction factor metrics."""
import json
import csv
from pathlib import Path

CLASSES = ["B1", "B2", "B3", "B4"]
ROOT = Path(__file__).resolve().parent.parent
ann_dir = ROOT / "ground_truth" / "annotations"
manifest = ROOT / "ground_truth" / "split_manifest.csv"

splits = {}
with open(manifest) as f:
    for row in csv.DictReader(f):
        splits[row["tree_id"]] = row["new_split"]

# Calibrate on training split
naive_total = {c: 0.0 for c in CLASSES}
gt_total = {c: 0.0 for c in CLASSES}
for jp in sorted(ann_dir.glob("*.json")):
    if splits.get(jp.stem) != "train":
        continue
    with open(jp, encoding="utf-8-sig") as f:
        tree = json.load(f)
    gt = {c: int(tree.get("summary", {}).get("by_class", {}).get(c, 0)) for c in CLASSES}
    naive = {c: 0 for c in CLASSES}
    for side in tree.get("images", {}).values():
        for ann in side.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls in naive:
                naive[cls] += 1
    for c in CLASSES:
        naive_total[c] += naive[c]
        gt_total[c] += gt[c]

k_class = {c: naive_total[c] / gt_total[c] for c in CLASSES}
k_global = sum(naive_total.values()) / sum(gt_total.values())

print(f"k_global = {k_global:.4f}")
print("k_class:", {c: round(k_class[c], 4) for c in CLASSES})
print()


def pred_global(naive):
    return {c: max(0, round(naive[c] / k_global)) for c in CLASSES}


def pred_classwise(naive):
    return {c: max(0, round(naive[c] / k_class[c])) for c in CLASSES}


def pred_naive_fn(naive):
    return dict(naive)


def within1(pred, gt):
    return all(abs(pred.get(c, 0) - gt.get(c, 0)) <= 1 for c in CLASSES)


print(f"{'Method':<20} {'Set':<8} {'n':<5} {'Class ±1%':>10} {'Tree ±1%':>10} {'MacroMAE':>9} {'MeanBias':>9}")
print("-" * 75)

for split_name in ["all", "test"]:
    trees_data = []
    for jp in sorted(ann_dir.glob("*.json")):
        s = splits.get(jp.stem)
        if split_name == "test" and s != "test":
            continue
        with open(jp, encoding="utf-8-sig") as f:
            tree = json.load(f)
        gt = {c: int(tree.get("summary", {}).get("by_class", {}).get(c, 0)) for c in CLASSES}
        naive = {c: 0 for c in CLASSES}
        for side in tree.get("images", {}).values():
            for ann in side.get("annotations", []):
                cls = ann.get("class_name", "")
                if cls in naive:
                    naive[cls] += 1
        trees_data.append((naive, gt))

    n = len(trees_data)
    for mname, pfn in [("naive", pred_naive_fn), ("global_k", pred_global), ("classwise_k", pred_classwise)]:
        preds_all = [pfn(naive) for naive, gt in trees_data]
        gts_all = [gt for naive, gt in trees_data]
        joint_acc = sum(within1(p, g) for p, g in zip(preds_all, gts_all)) / n * 100
        macro_mae = sum(
            sum(abs(p.get(c, 0) - g.get(c, 0)) for c in CLASSES) / 4
            for p, g in zip(preds_all, gts_all)
        ) / n
        per_c_acc = [
            sum(abs(p.get(c, 0) - g.get(c, 0)) <= 1 for p, g in zip(preds_all, gts_all)) / n * 100
            for c in CLASSES
        ]
        macro_acc1 = sum(per_c_acc) / 4
        mean_bias = sum(
            sum(p.get(c, 0) - g.get(c, 0) for c in CLASSES) / 4
            for p, g in zip(preds_all, gts_all)
        ) / n
        print(f"{mname:<20} {split_name:<8} {n:<5} {macro_acc1:>9.2f}% {joint_acc:>9.2f}% {macro_mae:>9.3f} {mean_bias:>9.3f}")
