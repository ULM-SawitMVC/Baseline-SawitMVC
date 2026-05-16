"""
Complete E2E pipeline — per-image inference, aggregate by tree, ML or heuristic counter.

Unlike run_e2e_pipeline.py (which groups all sides before inference), this script:
  1. Runs YOLO on each image independently (or derives from existing per-tree JSONs)
  2. Groups per-image JSONs by tree
  3. Applies a counter: simple aggregation (max/mean/sum) OR ML/heuristic counter

Simple aggregation (no training):
  max  : count[c] = max detections in any single image  (best simple dedup)
  mean : count[c] = round(mean detections per image)
  sum  : count[c] = total detections across all images (naive, overcounts)

Full ML/heuristic counters (identical features to per-tree pipeline):
  m01  : M01 heuristic (multi-view dedup)
  svm  : SVM with RBF kernel (GridSearchCV)
  rf   : Random Forest (n=200, max_depth=10)
  lr   : Linear Regression + StandardScaler (interpretable)

Usage:
    # Full run (requires GPU + dataset)
    python pipeline/run_e2e_per_image.py --name y26n --weights models/y26n.pt

    # Skip inference: derive per-image from existing per-tree predictions (no GPU needed)
    python pipeline/run_e2e_per_image.py --name y26n --weights models/y26n.pt \\
        --data /workspace/SawitMVC-YOLO --skip-inference

    # Run only specific counters
    python pipeline/run_e2e_per_image.py --name y26n --weights models/y26n.pt \\
        --skip-inference --counters max svm lr
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gt(gt_dir: Path) -> dict[str, dict]:
    gt: dict = {}
    for fp in sorted(gt_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        tid = d.get("tree_id") or d.get("tree_name") or fp.stem
        by_class = d.get("summary", {}).get("by_class", {})
        gt_split = d.get("split", "train")
        gt[tid] = {"gt": {c: int(by_class.get(c, 0)) for c in CLASSES}, "split": gt_split}
    return gt


def _load_splits(data_dir: Path) -> dict[str, str]:
    manifest = data_dir / "split_manifest.csv"
    import csv
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


def _compute_metrics(rows: list[dict], split: str) -> dict:
    sub = [r for r in rows if r["split"] == split]
    if not sub:
        return {}
    m: dict = {}
    for c in CLASSES:
        errs = [abs(r[f"pred_{c}"] - r[f"gt_{c}"]) for r in sub]
        m[f"MAE_{c}"] = float(np.mean(errs))
        m[f"bias_{c}"] = float(np.mean([r[f"pred_{c}"] - r[f"gt_{c}"] for r in sub]))
        m[f"acc_pm1_{c}"] = float(np.mean([e <= 1 for e in errs]))
    m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
    m["macro_acc_pm1"] = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
    total_errs = [abs(sum(r[f"pred_{c}"] for c in CLASSES) - sum(r[f"gt_{c}"] for c in CLASSES)) for r in sub]
    m["total_count_mae"] = float(np.mean(total_errs))
    m["total_pm1_acc"] = float(np.mean([e <= 1 for e in total_errs]))
    m["exact_profile_acc"] = float(np.mean([all(r[f"pred_{c}"] == r[f"gt_{c}"] for c in CLASSES) for r in sub]))
    m["split"] = split
    return m


def _save_results(rows: list[dict], out_dir: Path, name: str, counter: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m_test = _compute_metrics(rows, "test")
    m_val  = _compute_metrics(rows, "val")
    (out_dir / "metrics.json").write_text(json.dumps({"test": m_test, "val": m_val}, indent=2))
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    n_test = len([r for r in rows if r["split"] == "test"])
    print(f"{counter.upper()}: Acc±1={m_test.get('macro_acc_pm1', 0)*100:.2f}%  "
          f"MAE={m_test.get('macro_class_mae', 0):.4f}  (n={n_test}) → {out_dir}")


# ---------------------------------------------------------------------------
# Step 0: Derive per-image JSONs from existing per-tree JSONs (no GPU needed)
# ---------------------------------------------------------------------------

def derive_per_image_from_per_tree(per_tree_dir: Path, per_image_dir: Path, detector: str) -> int:
    """Split per-tree JSONs into per-image JSONs (one JSON per image side).

    Per-tree format: images keyed as 'sisi_N' with list of annotations.
    Per-image output: {image, detector, detections, count_per_class}
    """
    per_image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fp in sorted(per_tree_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        tree_id = d.get("tree_name") or d.get("tree_id") or fp.stem
        for side_key, side_data in d.get("images", {}).items():
            # parse side number from key (e.g. 'sisi_1' → 1, 'side_2' → 2)
            m = re.search(r"(\d+)$", side_key)
            if not m:
                continue
            side_num = int(m.group(1))
            img_name = f"{tree_id}_{side_num}.jpg"
            out_path = per_image_dir / f"{tree_id}_{side_num}.json"
            if out_path.exists():
                written += 1
                continue
            anns = side_data.get("annotations", [])
            count_per_class = {c: sum(1 for a in anns if (a.get("class_name") or a.get("class", "")) == c)
                               for c in CLASSES}
            out_path.write_text(json.dumps({
                "image": img_name,
                "detector": detector,
                "detections": anns,
                "count_per_class": count_per_class,
                "side_index": side_num - 1,
            }))
            written += 1
    return written


# ---------------------------------------------------------------------------
# Step 1: Per-image inference from scratch (requires GPU)
# ---------------------------------------------------------------------------

def run_inference_per_image(name: str, weights: Path, data_dir: Path,
                             per_image_dir: Path, conf: float = 0.25) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    per_image_dir.mkdir(parents=True, exist_ok=True)
    images = sorted((data_dir / "images").rglob("*.jpg"))
    print(f"Running inference on {len(images)} images → {per_image_dir}")
    done = 0
    for img_path in images:
        out_path = per_image_dir / f"{img_path.stem}.json"
        if out_path.exists():
            done += 1
            continue
        results = model(str(img_path), verbose=False, conf=conf)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for k in range(len(r.boxes)):
                xywhn = r.boxes.xywhn[k].tolist()
                detections.append({
                    "class_name": CLASS_MAP.get(int(r.boxes.cls[k].item()), "B3"),
                    "bbox_yolo": xywhn,
                    "conf": float(r.boxes.conf[k].item()),
                })
        m = re.match(r"^(.+)_(\d+)$", img_path.stem)
        side_num = int(m.group(2)) if m else 0
        count_per_class = {c: sum(1 for d in detections if d["class_name"] == c) for c in CLASSES}
        out_path.write_text(json.dumps({
            "image": img_path.name,
            "detector": name,
            "detections": detections,
            "count_per_class": count_per_class,
            "side_index": side_num - 1,
        }))
        done += 1
    print(f"Inference complete: {done} images.")


# ---------------------------------------------------------------------------
# Step 2: Group per-image JSONs → per-tree
# ---------------------------------------------------------------------------

def group_per_image_by_tree(per_image_dir: Path, gt_map: dict, splits_map: dict
                             ) -> tuple[dict, list]:
    """Returns (tree_images, rows_template) where tree_images[tid] = list of side dicts."""
    tree_images: dict[str, list[dict]] = defaultdict(list)

    for jp in sorted(per_image_dir.glob("*.json")):
        d = json.loads(jp.read_text())
        img_name = d.get("image", jp.name)
        m = re.match(r"^(.+)_(\d+)(?:\.jpg)?$", Path(img_name).stem)
        if not m:
            continue
        tree_id = m.group(1)
        side_num = int(m.group(2))
        tree_images[tree_id].append({
            "side_num": side_num,
            "side_index": d.get("side_index", side_num - 1),
            "count_per_class": d.get("count_per_class", {c: 0 for c in CLASSES}),
            "detections": d.get("detections", []),
        })

    # Sort sides within each tree
    for tid in tree_images:
        tree_images[tid].sort(key=lambda x: x["side_num"])

    return dict(tree_images)


# ---------------------------------------------------------------------------
# Step 3a: Simple aggregation counters (max / mean / sum)
# ---------------------------------------------------------------------------

def step_aggregation(name: str, agg: str, per_image_dir: Path, gt_dir: Path,
                     data_dir: Path, out_dir: Path) -> None:
    gt_map = _load_gt(gt_dir)
    splits_map = _load_splits(data_dir)
    tree_images = group_per_image_by_tree(per_image_dir, gt_map, splits_map)
    rows = []
    for tid, sides in tree_images.items():
        if tid not in gt_map:
            continue
        counts_list = [s["count_per_class"] for s in sides]
        if agg == "max":
            pred = {c: max(cnt.get(c, 0) for cnt in counts_list) for c in CLASSES}
        elif agg == "mean":
            pred = {c: round(sum(cnt.get(c, 0) for cnt in counts_list) / len(counts_list))
                    for c in CLASSES}
        else:  # sum
            pred = {c: sum(cnt.get(c, 0) for cnt in counts_list) for c in CLASSES}
        entry = gt_map[tid]
        split = splits_map.get(tid, entry["split"])
        row = {"tree_id": tid, "split": split}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"] = entry["gt"].get(c, 0)
        rows.append(row)
    _save_results(rows, out_dir, name, agg)


# ---------------------------------------------------------------------------
# Step 3b: M01 heuristic counter (multi-view dedup per tree)
# ---------------------------------------------------------------------------

def step_m01(name: str, per_image_dir: Path, gt_dir: Path, data_dir: Path, out_dir: Path) -> None:
    from algorithms.M01_selector_b2b3 import predict
    gt_map = _load_gt(gt_dir)
    splits_map = _load_splits(data_dir)
    tree_images = group_per_image_by_tree(per_image_dir, gt_map, splits_map)
    rows = []
    for tid, sides in tree_images.items():
        if tid not in gt_map:
            continue
        dets = []
        for side in sides:
            for ann in side["detections"]:
                bbox = ann.get("bbox_yolo", [0.5, 0.5, 0.1, 0.1])
                dets.append({
                    "class": ann.get("class_name") or ann.get("class", "B3"),
                    "x_norm": float(bbox[0]),
                    "y_norm": float(bbox[1]),
                    "side_index": int(side["side_index"]),
                })
        pred = predict(dets)
        entry = gt_map[tid]
        split = splits_map.get(tid, entry["split"])
        row = {"tree_id": tid, "split": split}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"] = entry["gt"].get(c, 0)
        rows.append(row)
    _save_results(rows, out_dir, name, "m01")


# ---------------------------------------------------------------------------
# Step 3c: ML counters (SVM / RF / LR) using 13-dim features from per-image data
# ---------------------------------------------------------------------------

def step_ml(name: str, counter: str, per_image_dir: Path, gt_dir: Path,
            data_dir: Path, out_dir: Path) -> None:
    from build_counting_features import extract_features, FEATURE_NAMES
    gt_map = _load_gt(gt_dir)
    splits_map = _load_splits(data_dir)
    tree_images = group_per_image_by_tree(per_image_dir, gt_map, splits_map)

    X, y, tree_ids, tree_splits = [], [], [], []
    for tid, sides in sorted(tree_images.items()):
        if tid not in gt_map:
            continue
        # Reconstruct tree-level structure for extract_features
        tree_dict = {
            "images": {
                f"side_{s['side_num']}": {
                    "side_index": s["side_index"],
                    "annotations": [{"class_name": a.get("class_name") or a.get("class", "B3")}
                                    for a in s["detections"]],
                }
                for s in sides
            }
        }
        feats = extract_features(tree_dict)
        entry = gt_map[tid]
        label = np.array([entry["gt"].get(c, 0) for c in CLASSES], dtype=np.float32)
        split = splits_map.get(tid, entry["split"])
        X.append(feats); y.append(label); tree_ids.append(tid); tree_splits.append(split)

    X = np.array(X); y = np.array(y)
    splits = np.array(tree_splits)
    train_m, val_m, test_m = splits == "train", splits == "val", splits == "test"
    X_tr, y_tr = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_te, y_te = X[test_m], y[test_m]
    ids_val = [tree_ids[i] for i in np.where(val_m)[0]]
    ids_te  = [tree_ids[i] for i in np.where(test_m)[0]]

    if counter == "svm":
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        pipe = Pipeline([("sc", StandardScaler()), ("svr", MultiOutputRegressor(SVR(kernel="rbf")))])
        gs = GridSearchCV(pipe, {"svr__estimator__C": [0.1, 1, 10], "svr__estimator__gamma": ["scale", 0.01, 0.1]},
                          cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        predict_fn = gs.predict
        extra = {"best_params": str(gs.best_params_), "cv_mae": float(-gs.best_score_)}
    elif counter == "lr":
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(X_tr, y_tr)
        predict_fn = pipe.predict
        extra = {}
    else:  # rf
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        predict_fn = rf.predict
        extra = {"feature_importance": {n: float(v) for n, v in zip(FEATURE_NAMES, rf.feature_importances_)}}

    def _eval(X_sub, y_sub, ids_sub):
        yp = np.clip(np.round(predict_fn(X_sub)), 0, None).astype(int)
        yt = y_sub.astype(int)
        rows = []
        for i, tid in enumerate(ids_sub):
            row = {"tree_id": tid, "split": splits_map.get(tid, "unknown")}
            for j, c in enumerate(CLASSES):
                row[f"pred_{c}"] = int(yp[i, j])
                row[f"gt_{c}"] = int(yt[i, j])
            rows.append(row)
        return rows

    rows_te = _eval(X_te, y_te, ids_te)
    rows_val = _eval(X_val, y_val, ids_val)
    all_rows = rows_te + rows_val

    out_dir.mkdir(parents=True, exist_ok=True)
    m_te = _compute_metrics(rows_te, "test"); m_te["split"] = "test"
    m_val = _compute_metrics(rows_val, "val"); m_val["split"] = "val"
    metrics = {"test": {**m_te, **extra}, "val": m_val}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(all_rows).to_csv(out_dir / "predictions.csv", index=False)
    if counter == "rf":
        fi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        fi.to_csv(out_dir / "feature_importance.csv", index=False)
    n_test = len(rows_te)
    print(f"{counter.upper()}: Acc±1={m_te.get('macro_acc_pm1', 0)*100:.2f}%  "
          f"MAE={m_te.get('macro_class_mae', 0):.4f}  (n={n_test}) → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Complete E2E per-image pipeline: inference → group by tree → counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Counters:
  max   Simple aggregation: count[c] = max detections in any single image
  mean  Simple aggregation: count[c] = round(mean detections per image)
  sum   Simple aggregation: count[c] = total detections (naive, no dedup)
  m01   M01 heuristic (multi-view dedup, no training)
  svm   SVM with RBF kernel (GridSearchCV, trained on train split)
  rf    Random Forest (n=200, max_depth=10)
  lr    Linear Regression + StandardScaler (interpretable)
        """)
    p.add_argument("--name", required=True, help="Experiment name, e.g. y26n or y26m")
    p.add_argument("--weights", type=Path, required=True, help="Path to YOLO .pt weights")
    p.add_argument("--data", type=Path, default=ROOT / "SawitMVC-YOLO",
                   help="Dataset root (default: ./SawitMVC-YOLO/)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip inference. If per-image JSONs don't exist, derives from per-tree predictions.")
    p.add_argument("--counters", nargs="+",
                   default=["max", "mean", "sum", "m01", "svm", "lr", "rf"],
                   choices=["max", "mean", "sum", "m01", "svm", "rf", "lr"],
                   help="Counters to run (default: all)")
    args = p.parse_args()

    per_image_dir = ROOT / "predictions" / f"{args.name}_per_image"
    per_tree_dir  = ROOT / "predictions" / f"{args.name}_inference"
    gt_dir        = args.data / "json"
    e2e_base      = ROOT / "benchmarks" / "e2e"

    # Step 1: Inference
    if not args.skip_inference:
        print(f"\n[Step 1] Per-image YOLO inference: {args.name}")
        run_inference_per_image(args.name, args.weights, args.data, per_image_dir, args.conf)
    else:
        n_existing = len(list(per_image_dir.glob("*.json"))) if per_image_dir.exists() else 0
        if n_existing > 0:
            print(f"[Step 1] Skipped. Using {n_existing} existing per-image JSONs in {per_image_dir}")
        elif per_tree_dir.exists():
            print(f"[Step 1] Per-image JSONs not found. Deriving from per-tree predictions in {per_tree_dir} ...")
            n = derive_per_image_from_per_tree(per_tree_dir, per_image_dir, args.name)
            print(f"  Derived {n} per-image JSONs → {per_image_dir}")
        else:
            print(f"ERROR: No per-image JSONs at {per_image_dir} and no per-tree at {per_tree_dir}.")
            print("Run without --skip-inference to generate predictions.")
            raise SystemExit(1)

    n_imgs = len(list(per_image_dir.glob("*.json")))
    print(f"  Total per-image JSONs available: {n_imgs}")

    # Steps 2+3: Count with selected counters
    print(f"\n[Steps 2-3] Counters: {args.counters}")
    for ctr in args.counters:
        out_dir = e2e_base / f"e2e_{args.name}_per_image_{ctr}"
        if ctr in ("max", "mean", "sum"):
            step_aggregation(args.name, ctr, per_image_dir, gt_dir, args.data, out_dir)
        elif ctr == "m01":
            step_m01(args.name, per_image_dir, gt_dir, args.data, out_dir)
        else:
            step_ml(args.name, ctr, per_image_dir, gt_dir, args.data, out_dir)

    print(f"\nDone. Results in {e2e_base}/")


if __name__ == "__main__":
    main()
