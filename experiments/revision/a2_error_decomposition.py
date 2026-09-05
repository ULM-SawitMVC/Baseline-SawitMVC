"""
A2 - Detection-level error decomposition and a detector/association error ladder.

Answers reviewer requests: a rigorous decomposition of detector and counting
error, an explicit split between missed detections, false positives, and
maturity-class confusion, and a clear separation of detector limitations from
multi-view association/aggregation limitations.

Two products:

1. A per-image matching of YOLO26m detections against the ground-truth boxes on
   the 141 test trees (greedy, class-agnostic, IoU >= 0.5, pixel coordinates),
   giving a 5x5 confusion matrix over {B1..B4, background} plus per-class
   precision and recall.

2. An oracle ladder that walks from perfect information to the deployed
   pipeline, each rung adding exactly one detector error source while
   multi-view association stays perfect (detections inherit the identity of the
   ground-truth bunch they match):

     L0  ground-truth counts                       (no error)
     L1  + missed detections                       (recall only)
     L2  + maturity-class confusion                (recall + classification)
     L3  + false positives                         (full detector error)
     L4  learned counter on detector output        (deployed pipeline)

   These are diagnostic estimators, not an additive attribution of deployed
   error. L2 uses confidence-weighted voting and L3 counts each unmatched
   detection separately; neither is an optimum over possible counters.

Usage:  python experiments/revision/a2_error_decomposition.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revcommon as rc

IOU_THRESHOLD = 0.5
LABELS = rc.CLASSES + ["background"]


def to_xyxy(bbox_yolo, width: float, height: float) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in bbox_yolo[:4])
    return (
        (cx - w / 2) * width, (cy - h / 2) * height,
        (cx + w / 2) * width, (cy + h / 2) * height,
    )


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def bunch_index(gt: dict) -> dict[tuple[str, int], int]:
    """(side, box_index) -> bunch_id."""
    out: dict[tuple[str, int], int] = {}
    for bunch in gt.get("bunches", []):
        bid = int(bunch["bunch_id"])
        for app in bunch.get("appearances", []):
            out[(app["side"], int(app["box_index"]))] = bid
    return out


def bunch_class(gt: dict) -> dict[int, str]:
    return {int(b["bunch_id"]): b["class"] for b in gt.get("bunches", [])}


def match_tree(gt: dict, pred: dict):
    """Greedy class-agnostic matching per side.

    Returns (pairs, fp, fn) where
      pairs = list of (side, gt_box_index, gt_class, pred_class, conf)
      fp    = list of (side, pred_class, conf)
      fn    = list of (side, gt_box_index, gt_class)
    """
    pairs, fps, fns = [], [], []
    for side, gt_side in gt.get("images", {}).items():
        width = float(gt_side.get("width", 1.0))
        height = float(gt_side.get("height", 1.0))
        gt_boxes = [
            (int(a.get("box_index", i)), a["class_name"],
             to_xyxy(a["bbox_yolo"], width, height))
            for i, a in enumerate(gt_side.get("annotations", []))
        ]
        pred_side = pred.get("images", {}).get(side, {})
        pred_boxes = [
            (a["class_name"], float(a.get("conf", 1.0)),
             to_xyxy(a["bbox_yolo"], width, height))
            for a in pred_side.get("annotations", [])
            if a.get("class_name") in rc.CLASSES
        ]
        pred_boxes.sort(key=lambda t: -t[1])

        used = set()
        for pcls, conf, pbox in pred_boxes:
            best_j, best_iou = -1, IOU_THRESHOLD
            for j, (bidx, gcls, gbox) in enumerate(gt_boxes):
                if j in used:
                    continue
                v = iou(pbox, gbox)
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0:
                used.add(best_j)
                bidx, gcls, _ = gt_boxes[best_j]
                pairs.append((side, bidx, gcls, pcls, conf))
            else:
                fps.append((side, pcls, conf))
        for j, (bidx, gcls, _) in enumerate(gt_boxes):
            if j not in used:
                fns.append((side, bidx, gcls))
    return pairs, fps, fns


def counts_vector(counter: Counter) -> np.ndarray:
    return np.array([counter.get(c, 0) for c in rc.CLASSES], dtype=float)


def main() -> None:
    manifest = pd.read_csv(rc.ROOT / "ground_truth" / "split_manifest.csv",
                           encoding="utf-8-sig")
    test_ids = manifest.loc[manifest.new_split == "test", "tree_id"].tolist()

    confusion = np.zeros((5, 5), dtype=int)   # rows = GT label, cols = predicted
    y_true, l1, l2, l3 = [], [], [], []
    n_gt_appearance = Counter()
    n_pred = Counter()
    per_class_missed_bunch = Counter()
    per_class_unique_bunch = Counter()
    fp_conf_all: list[float] = []
    tp_conf_all: list[float] = []

    for tid in test_ids:
        gt = load_json(rc.GT_DIR / f"{tid}.json")
        pred = load_json(rc.PRED_DIR / f"{tid}.json")
        bidx_map = bunch_index(gt)
        bcls_map = bunch_class(gt)

        pairs, fps, fns = match_tree(gt, pred)

        for _, _, gcls, pcls, conf in pairs:
            confusion[LABELS.index(gcls), LABELS.index(pcls)] += 1
            n_gt_appearance[gcls] += 1
            n_pred[pcls] += 1
            tp_conf_all.append(conf)
        for _, pcls, conf in fps:
            confusion[LABELS.index("background"), LABELS.index(pcls)] += 1
            n_pred[pcls] += 1
            fp_conf_all.append(conf)
        for _, _, gcls in fns:
            confusion[LABELS.index(gcls), LABELS.index("background")] += 1
            n_gt_appearance[gcls] += 1

        # ---- oracle ladder -------------------------------------------------
        detected_votes: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for side, bidx, _gcls, pcls, conf in pairs:
            bid = bidx_map.get((side, bidx))
            if bid is not None:
                detected_votes[bid].append((pcls, conf))

        gt_counts = Counter(bcls_map.values())
        per_class_unique_bunch.update(gt_counts)
        y_true.append(counts_vector(gt_counts))

        # L1: bunches with at least one matched detection, true class kept
        c1 = Counter(bcls_map[b] for b in detected_votes)
        l1.append(counts_vector(c1))

        # L2: same bunches, class = confidence-weighted majority of detections
        c2 = Counter()
        for bid, votes in detected_votes.items():
            weight: dict[str, float] = defaultdict(float)
            for cls, conf in votes:
                weight[cls] += conf
            c2[max(weight.items(), key=lambda kv: kv[1])[0]] += 1
        l2.append(counts_vector(c2))

        # L3: L2 plus every unmatched detection as an extra bunch
        c3 = Counter(c2)
        for _, pcls, _ in fps:
            c3[pcls] += 1
        l3.append(counts_vector(c3))

        for bid, cls in bcls_map.items():
            if bid not in detected_votes:
                per_class_missed_bunch[cls] += 1

    y_true = np.array(y_true)
    ladders = {"L1_recall": np.array(l1), "L2_recall_class": np.array(l2),
               "L3_full_detector": np.array(l3)}

    # ---- L4: the deployed counter -----------------------------------------
    df, y, tree_ids, splits, _ = rc.load_dataset(rc.PRED_DIR)
    order = {t: i for i, t in enumerate(tree_ids)}
    idx = np.array([order[t] for t in test_ids])
    tr = splits == "train"
    fsets = rc.feature_sets(df)
    x = df[fsets["F_all"]].values.astype(float)
    model = rc.model_builders()["Ridge"]()
    model.fit(x[tr], y[tr])
    l4 = rc.round_counts(model.predict(x[idx]))
    assert np.array_equal(y[idx], y_true), "target mismatch between ladder and counter"
    ladders["L4_deployed_counter"] = l4.astype(float)

    # naive summation of raw detections, for reference
    naive = df.loc[idx, [f"naive_sum_{c}" for c in rc.CLASSES]].values
    ladders["naive_sum_detector"] = naive

    rows = []
    for name, pred in ladders.items():
        st = rc.per_tree_stats(y_true, rc.round_counts(pred))
        m = rc.metrics_from_stats(st)
        rows.append({"stage": name, **m})
        print(f"{name:22s} macro={m['macro']*100:6.2f}%  tree={m['joint']*100:6.2f}%  "
              f"mae={m['mae']:.4f}  totMAE={m['total_mae']:.3f}  "
              + "  ".join(f"{c}:{m[f'acc_{c}']*100:5.1f}%" for c in rc.CLASSES))
    ladder_df = pd.DataFrame(rows)
    rc.OUT_DIR.mkdir(parents=True, exist_ok=True)
    ladder_df.to_csv(rc.OUT_DIR / "a2_error_ladder.csv", index=False)

    # ---- confusion matrix and detection rates ------------------------------
    conf_df = pd.DataFrame(confusion, index=[f"gt_{l}" for l in LABELS],
                           columns=[f"pred_{l}" for l in LABELS])
    conf_df.to_csv(rc.OUT_DIR / "a2_confusion_matrix.csv")
    print("\nAppearance-level confusion (rows = GT, cols = prediction):")
    print(conf_df.to_string())

    det_rows = []
    for i, c in enumerate(rc.CLASSES):
        n_gt = confusion[i, :].sum()
        n_pr = confusion[:, i].sum()
        tp = confusion[i, i]
        missed = confusion[i, LABELS.index("background")]
        confused = n_gt - tp - missed
        det_rows.append({
            "class": c,
            "gt_appearances": int(n_gt),
            "detections": int(n_pr),
            "true_positive": int(tp),
            "missed": int(missed),
            "class_confused": int(confused),
            "false_positive_bg": int(confusion[LABELS.index("background"), i]),
            "recall": float(tp / n_gt) if n_gt else float("nan"),
            "precision": float(tp / n_pr) if n_pr else float("nan"),
            "localised_recall": float((n_gt - missed) / n_gt) if n_gt else float("nan"),
            "unique_bunches": int(per_class_unique_bunch[c]),
        })
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(rc.OUT_DIR / "a2_detection_rates.csv", index=False)
    print("\nTest-split appearance-level detection rates:")
    print(det_df.drop(columns=["unique_bunches"]).to_string(index=False))

    summary = {
        "iou_threshold": IOU_THRESHOLD,
        "n_test_trees": len(test_ids),
        "gt_appearances_total": int(confusion[:4, :].sum()),
        "detections_total": int(confusion[:, :4].sum()),
        "false_positives_total": int(confusion[LABELS.index("background"), :4].sum()),
        "missed_total": int(confusion[:4, LABELS.index("background")].sum()),
        "class_confused_total": int(
            confusion[:4, :4].sum() - np.trace(confusion[:4, :4])),
        "missed_unique_bunches_per_class": dict(per_class_missed_bunch),
        "mean_conf_tp": float(np.mean(tp_conf_all)) if tp_conf_all else float("nan"),
        "mean_conf_fp": float(np.mean(fp_conf_all)) if fp_conf_all else float("nan"),
        "ladder": ladder_df.to_dict(orient="records"),
    }
    rc.dump(summary, rc.OUT_DIR / "a2_decomposition_summary.json")


if __name__ == "__main__":
    main()
