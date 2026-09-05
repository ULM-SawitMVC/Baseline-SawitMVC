"""
A4 - How far the conclusion depends on the particular detector checkpoint.

Answers reviewer requests: clarify the scope of the detector-related conclusion
when only one checkpoint is evaluated, and evaluate an additional detector to
test the generality of the finding.

Two experiments:

1. Operating-point sweep. The confidence threshold of the released y26mv2
   checkpoint is raised from 0.25 to 0.70 in steps, which moves the detector
   along its own precision/recall curve. Counters are refitted at every
   threshold on the official 716 training trees and evaluated on the 141 test
   trees.

2. Detector-capacity comparison. Three earlier YOLO26 checkpoints (n, s, m)
   are compared against the released y26mv2 checkpoint. Those three were
   trained under the previous 763/95/95 protocol, so a checkpoint-fair subset
   is used: counters are fitted on the 590 trees that are training trees under
   both protocols and evaluated on the 64 trees that are held out under both.
   Evaluation trees are outside gradient training, but include validation
   trees used in detector development. This is within-family sensitivity;
   historical splits and capacity are confounded, not an architecture ranking.

Usage:  python experiments/revision/a4_detector_sensitivity.py
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc
import a2_error_decomposition as a2

OLD_SPLIT_COMMIT = "06acd73a^:ground_truth/split_manifest.csv"
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
CONFIGS = [("F0", "ElasticNet"), ("F0", "Ridge"), ("F_all", "Ridge")]


def old_split_map() -> dict[str, str]:
    cached = rc.OUT_DIR / "a4_old_split_manifest.csv"
    if cached.exists():
        old = pd.read_csv(cached)
    else:
        raw = subprocess.run(["git", "show", OLD_SPLIT_COMMIT], cwd=rc.ROOT,
                             capture_output=True, text=True, check=True).stdout
        old = pd.read_csv(io.StringIO(raw))
        old.columns = [c.lstrip("﻿") for c in old.columns]
        rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
        old.to_csv(cached, index=False)
    return dict(zip(old.tree_id, old.new_split))


def sweep() -> pd.DataFrame:
    rows = []
    for tau in THRESHOLDS:
        df, y, tree_ids, splits, _ = rc.load_dataset(rc.PRED_DIR, conf_min=tau)
        tr, te = splits == "train", splits == "test"
        fsets = rc.feature_sets(df)
        n_det = float(df.loc[te, "total_naive"].sum())
        for features, model_name in CONFIGS:
            x = df[fsets[features]].values.astype(float)
            model = rc.model_builders()[model_name]()
            model.fit(x[tr], y[tr])
            m = rc.metrics_from_stats(
                rc.per_tree_stats(y[te], rc.round_counts(model.predict(x[te]))))
            rows.append({"threshold": tau, "features": features, "model": model_name,
                         "test_detections": n_det, **m})
            print(f"tau={tau:.2f} {features:6s} {model_name:11s} "
                  f"det={n_det:6.0f}  macro={m['macro']*100:6.2f}%  "
                  f"tree={m['joint']*100:6.2f}%  mae={m['mae']:.4f}")
    return pd.DataFrame(rows)


def detector_recall(pred_dir: Path, tree_ids: list[str]) -> dict[str, float]:
    """Appearance-level recall per class on the given trees."""
    n_gt = {c: 0 for c in rc.CLASSES}
    n_tp = {c: 0 for c in rc.CLASSES}
    n_loc = {c: 0 for c in rc.CLASSES}
    n_fp = 0
    for tid in tree_ids:
        gt = a2.load_json(rc.GT_DIR / f"{tid}.json")
        pred = a2.load_json(pred_dir / f"{tid}.json")
        pairs, fps, fns = a2.match_tree(gt, pred)
        n_fp += len(fps)
        for _, _, gcls, pcls, _ in pairs:
            n_gt[gcls] += 1
            n_loc[gcls] += 1
            if gcls == pcls:
                n_tp[gcls] += 1
        for _, _, gcls in fns:
            n_gt[gcls] += 1
    out = {f"recall_{c}": (n_tp[c] / n_gt[c] if n_gt[c] else float("nan")) for c in rc.CLASSES}
    out["recall_macro"] = float(np.mean(list(out.values())))
    out["loc_recall_macro"] = float(np.mean(
        [n_loc[c] / n_gt[c] if n_gt[c] else np.nan for c in rc.CLASSES]))
    out["false_positives"] = n_fp
    return out


def capacity_comparison() -> pd.DataFrame:
    old_map = old_split_map()
    manifest = pd.read_csv(rc.ROOT / "ground_truth" / "split_manifest.csv",
                           encoding="utf-8-sig")
    new_map = dict(zip(manifest.tree_id, manifest.new_split))
    common_train = {t for t in new_map
                    if new_map[t] == "train" and old_map.get(t) == "train"}
    common_eval = {t for t in new_map
                   if new_map[t] in ("val", "test") and old_map.get(t) in ("val", "test")}
    print(f"checkpoint-fair subset: {len(common_train)} fit trees, "
          f"{len(common_eval)} evaluation trees")

    detectors = {
        "YOLO26n (old protocol)": rc.ARCHIVE_PRED / "y26n_per_tree",
        "YOLO26s (old protocol)": rc.ARCHIVE_PRED / "y26s_per_tree",
        "YOLO26m (old protocol)": rc.ARCHIVE_PRED / "y26m_per_tree",
        "YOLO26m (y26mv2, released)": rc.PRED_DIR,
        "ground-truth detections": rc.GT_DIR,
    }
    rows = []
    for label, pred_dir in detectors.items():
        df, y, tree_ids, _, _ = rc.load_dataset(pred_dir)
        tr = np.array([t in common_train for t in tree_ids])
        te = np.array([t in common_eval for t in tree_ids])
        eval_ids = [t for t in tree_ids if t in common_eval]
        rec = ({} if pred_dir == rc.GT_DIR else detector_recall(pred_dir, eval_ids))
        fsets = rc.feature_sets(df)
        for features, model_name in CONFIGS:
            x = df[fsets[features]].values.astype(float)
            model = rc.model_builders()[model_name]()
            model.fit(x[tr], y[tr])
            m = rc.metrics_from_stats(
                rc.per_tree_stats(y[te], rc.round_counts(model.predict(x[te]))))
            rows.append({"detector": label, "features": features, "model": model_name,
                         "n_fit": int(tr.sum()), "n_eval": int(te.sum()), **rec, **m})
            print(f"{label:28s} {features:6s} {model_name:11s} "
                  f"macro={m['macro']*100:6.2f}%  tree={m['joint']*100:6.2f}%  "
                  f"mae={m['mae']:.4f}"
                  + (f"  recall={rec['recall_macro']*100:5.1f}%" if rec else ""))
    return pd.DataFrame(rows)


def main() -> None:
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== operating-point sweep ===")
    sweep_df = sweep()
    sweep_df.to_csv(rc.OUT_DIR / "a4_confidence_sweep.csv", index=False)
    print("\n=== detector-capacity comparison ===")
    cap_df = capacity_comparison()
    cap_df.to_csv(rc.OUT_DIR / "a4_detector_capacity.csv", index=False)
    print(f"\nwrote {rc.OUT_DIR / 'a4_confidence_sweep.csv'} and "
          f"{rc.OUT_DIR / 'a4_detector_capacity.csv'}")


if __name__ == "__main__":
    main()
