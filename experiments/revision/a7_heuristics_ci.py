"""
A7 - Heuristic baselines under GT detections, with bootstrap intervals.

Recomputes the three heuristic rows of the GT-condition table (naive
appearance sum, global divisor, and the M01 profile selector) on the 141 test
trees so that the same tree-level bootstrap used for the learned counters can
be attached to them.

Usage:  python experiments/revision/a7_heuristics_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

sys.path.insert(0, str(rc.ROOT))
from algorithms.M01_selector_b2b3 import predict as predict_m01  # noqa: E402

CI_KEYS = ["macro", "joint", "mae", "rmse", "total_mae", "total_acc"]


def tree_payload(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        tree = json.load(f)
    dets, naive = [], {c: 0 for c in rc.CLASSES}
    for side in tree.get("images", {}).values():
        side_index = int(side.get("side_index", 0))
        for ann in side.get("annotations", []):
            cls = ann.get("class_name", "")
            if cls not in rc.CLASSES:
                continue
            bbox = ann.get("bbox_yolo", [0, 0, 0, 0])
            dets.append({"class": cls, "x_norm": float(bbox[0]),
                         "y_norm": float(bbox[1]), "side_index": side_index})
            naive[cls] += 1
    gt = tree.get("summary", {}).get("by_class", {})
    truth = {c: int(gt.get(c, 0)) for c in rc.CLASSES}
    return dets, naive, truth


def main() -> None:
    manifest = pd.read_csv(rc.ROOT / "ground_truth" / "split_manifest.csv",
                           encoding="utf-8-sig")
    split_of = dict(zip(manifest.tree_id, manifest.new_split))

    naive_sum = {c: 0.0 for c in rc.CLASSES}
    gt_sum = {c: 0.0 for c in rc.CLASSES}
    for tid, sp in split_of.items():
        if sp != "train":
            continue
        _, naive, truth = tree_payload(rc.GT_DIR / f"{tid}.json")
        for c in rc.CLASSES:
            naive_sum[c] += naive[c]
            gt_sum[c] += truth[c]
    k_global = sum(naive_sum.values()) / sum(gt_sum.values())
    print(f"global divisor k = {k_global:.4f} (fitted on the 716 training trees)")

    test_ids = [t for t, s in split_of.items() if s == "test"]
    y, p_naive, p_global, p_m01 = [], [], [], []
    for tid in sorted(test_ids):
        dets, naive, truth = tree_payload(rc.GT_DIR / f"{tid}.json")
        y.append([truth[c] for c in rc.CLASSES])
        p_naive.append([naive[c] for c in rc.CLASSES])
        p_global.append([max(0, round(naive[c] / k_global)) for c in rc.CLASSES])
        m01 = predict_m01(dets)
        p_m01.append([int(m01.get(c, 0)) for c in rc.CLASSES])

    y = np.array(y, dtype=float)
    out = {}
    for name, pred in (("naive sum", p_naive), ("global divisor", p_global),
                       ("M01 profile selector", p_m01)):
        st = rc.per_tree_stats(y, np.array(pred, dtype=int))
        m = rc.metrics_from_stats(st)
        out[name] = {"point": m, "ci": rc.bootstrap_ci(st, CI_KEYS)}
        ci = out[name]["ci"]
        print(f"{name:22s} macro={m['macro']*100:6.2f}% "
              f"[{ci['macro']['ci_lo']*100:.2f}, {ci['macro']['ci_hi']*100:.2f}]  "
              f"tree={m['joint']*100:6.2f}% "
              f"[{ci['joint']['ci_lo']*100:.2f}, {ci['joint']['ci_hi']*100:.2f}]  "
              f"mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  totMAE={m['total_mae']:.3f}")

    out["k_global"] = k_global
    rc.dump(out, rc.OUT_DIR / "a7_heuristics_ci.json")


if __name__ == "__main__":
    main()
