"""
Unified E2E pipeline — inference + SVM + RF + M01 heuristic for one detector.

Steps:
  1. YOLO inference on all 953 trees (grouped per tree, 4-8 images each)
  2. Extract 13-dim features per tree from inference JSONs
  3. Train + evaluate SVM, RF, Linear Regression, and M01 heuristic counters
  4. Save results to results/e2e_per_tree/{name}_{counter}/

Usage:
    # Skip inference, use pre-computed predictions (default; CPU only)
    python pipeline/run_e2e_pipeline.py --name y26mv2 --skip-inference

    # Full pipeline (requires GPU and the SawitMVC-YOLO/ image folder)
    python pipeline/run_e2e_pipeline.py --name y26mv2 --weights models/yolo/y26mv2.pt \
        --data SawitMVC-YOLO/

    # Custom ground-truth/annotation path
    python pipeline/run_e2e_pipeline.py --name y26mv2 --skip-inference --data ground_truth/
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def _resolve_gt_dir(data_dir: Path) -> Path:
    """Locate the annotations subdir; supports both ground_truth/ and SawitMVC-YOLO/ layouts."""
    for sub in ("annotations", "json"):
        candidate = data_dir / sub
        if candidate.is_dir():
            return candidate
    return data_dir / "annotations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits(data_dir: Path) -> dict[str, str]:
    """Load canonical per-tree splits from split_manifest.csv."""
    manifest = data_dir / "split_manifest.csv"
    splits: dict[str, str] = {}
    if manifest.exists():
        with open(manifest, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                tid = row.get("tree_id", "")
                sp = row.get("new_split") or row.get("split", "train")
                if tid:
                    splits[tid] = sp
        return splits

    # Fallback for incomplete local dataset copies: infer a tree split from image lists.
    for sp in ("train", "val", "test"):
        list_file = data_dir / f"{sp}.txt"
        if not list_file.exists():
            continue
        for line in list_file.read_text(encoding="utf-8").splitlines():
            fname = Path(line.strip()).name
            m = re.match(r"^(.+)_(\d+)\.jpg$", fname)
            if m:
                splits.setdefault(m.group(1), sp)
    return splits


def _load_gt(gt_dir: Path, splits_map: dict[str, str] | None = None) -> dict[str, dict]:
    splits_map = splits_map or {}
    gt: dict = {}
    for fp in sorted(gt_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        tid = d.get("tree_id") or d.get("tree_name") or fp.stem
        by_class = d.get("summary", {}).get("by_class", {})
        split = splits_map.get(tid, d.get("split", "train"))
        gt[tid] = {"gt": {c: int(by_class.get(c, 0)) for c in CLASSES}, "split": split}
    return gt


def _group_by_tree(data_dir: Path, split_map: dict) -> dict:
    trees: dict = defaultdict(list)
    for img_path in sorted((data_dir / "images").rglob("*.jpg")):
        m = re.match(r"^(.+)_(\d+)$", img_path.stem)
        if not m:
            continue
        tree_id, side_num = m.group(1), int(m.group(2))
        trees[tree_id].append((split_map.get(tree_id, "unknown"), side_num - 1, img_path))
    for tid in trees:
        trees[tid].sort(key=lambda x: x[1])
    return dict(trees)


def _build_dets(infer_json: dict) -> list[dict]:
    """Flatten inference JSON into detection list for algorithm input."""
    dets = []
    for side_label, side_data in infer_json.get("images", {}).items():
        # side_index: prefer explicit field, fallback to parsing label
        si = side_data.get("side_index")
        if si is None:
            m = re.search(r"(\d+)$", side_label)
            si = (int(m.group(1)) - 1) if m else 0
        for ann in side_data.get("annotations", []):
            bbox = ann.get("bbox_yolo", [0.5, 0.5, 0.1, 0.1])
            dets.append({
                "class": ann.get("class_name") or ann.get("class", "B3"),
                "x_norm": float(bbox[0]),
                "y_norm": float(bbox[1]),
                "side_index": int(si),
            })
    return dets


def _compute_metrics(rows: list[dict], split: str) -> dict:
    sub = [r for r in rows if r["split"] == split]
    if not sub:
        return {}
    m: dict = {"n_trees": len(sub)}
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


# ---------------------------------------------------------------------------
# Step 1: Inference
# ---------------------------------------------------------------------------

def step_inference(name: str, weights: Path, data_dir: Path, infer_dir: Path, conf: float) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    infer_dir.mkdir(parents=True, exist_ok=True)
    split_map = _load_splits(data_dir)
    trees = _group_by_tree(data_dir, split_map)
    done = 0
    for tree_id, sides in sorted(trees.items()):
        out_path = infer_dir / f"{tree_id}.json"
        if out_path.exists():
            done += 1
            continue
        split = sides[0][0]
        images_data: dict = {}
        for _, side_index, img_path in sides:
            side_label = f"side_{side_index + 1}"
            results = model(str(img_path), verbose=False, conf=conf)
            anns = []
            for r in results:
                if r.boxes is None:
                    continue
                for k in range(len(r.boxes)):
                    xywhn = r.boxes.xywhn[k].tolist()
                    anns.append({
                        "class_name": CLASS_MAP.get(int(r.boxes.cls[k].item()), "B3"),
                        "bbox_yolo": xywhn,
                        "conf": float(r.boxes.conf[k].item()),
                    })
            images_data[side_label] = {"side_index": side_index, "annotations": anns}
        out_path.write_text(json.dumps({
            "tree_name": tree_id, "split": split, "detector": name, "images": images_data
        }))
        done += 1
    print(f"Inference: {done}/{len(trees)} trees -> {infer_dir}")


# ---------------------------------------------------------------------------
# Step 2+3: Count with M01 heuristic
# ---------------------------------------------------------------------------

def step_m01(name: str, infer_dir: Path, gt_dir: Path, data_dir: Path, out_dir: Path) -> None:
    from algorithms.M01_selector_b2b3 import predict
    splits_map = _load_splits(data_dir)
    gt_map = _load_gt(gt_dir, splits_map)
    rows = []
    for fp in sorted(infer_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        tid = d.get("tree_name") or d.get("tree_id") or fp.stem
        if tid not in gt_map:
            continue
        dets = _build_dets(d)
        pred = predict(dets)
        entry = gt_map[tid]
        row = {"tree_id": tid, "split": entry["split"]}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"] = entry["gt"].get(c, 0)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    m_test = _compute_metrics(rows, "test")
    m_val  = _compute_metrics(rows, "val")
    (out_dir / "metrics.json").write_text(json.dumps({"test": m_test, "val": m_val}, indent=2))
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    print(f"M01: Class ±1 Acc={m_test.get('macro_acc_pm1',0)*100:.2f}%  MAE={m_test.get('macro_class_mae',0):.4f} -> {out_dir}")


# ---------------------------------------------------------------------------
# Step 2+3: Count with SVM / RF
# ---------------------------------------------------------------------------

def step_ml(name: str, counter: str, infer_dir: Path, gt_dir: Path, data_dir: Path, out_dir: Path) -> None:
    from build_counting_features import load_dataset, FEATURE_NAMES
    X, y, tree_ids, tree_splits = load_dataset(infer_dir, gt_dir, data_dir)
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
        extra = {}

    def _metrics(X_sub, y_sub, ids_sub, split_label: str):
        yp = np.clip(np.round(predict_fn(X_sub)), 0, None).astype(int)
        yt = y_sub.astype(int)
        m: dict = {"n_trees": int(len(ids_sub))}
        for j, c in enumerate(CLASSES):
            err = np.abs(yp[:, j] - yt[:, j])
            m[f"MAE_{c}"] = float(np.mean(err))
            m[f"bias_{c}"] = float(np.mean(yp[:, j] - yt[:, j]))
            m[f"acc_pm1_{c}"] = float(np.mean(err <= 1))
        m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
        m["macro_acc_pm1"] = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
        te = np.abs(yp.sum(1) - yt.sum(1))
        m["total_count_mae"] = float(np.mean(te))
        m["total_pm1_acc"] = float(np.mean(te <= 1))
        m["exact_profile_acc"] = float(np.mean(np.all(yp == yt, axis=1)))
        df = pd.DataFrame([dict(tree_id=tid, split=split_label,
                                **{f"pred_{c}": yp[i, j] for j, c in enumerate(CLASSES)},
                                **{f"gt_{c}": yt[i, j] for j, c in enumerate(CLASSES)})
                           for i, tid in enumerate(ids_sub)])
        return {**m, **extra}, df

    m_te, df_te = _metrics(X_te, y_te, ids_te, "test")
    m_te["split"] = "test"
    m_val, df_val = _metrics(X_val, y_val, ids_val, "val")
    m_val["split"] = "val"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"test": m_te, "val": m_val}, indent=2))
    pd.concat([df_te, df_val], ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    if counter == "rf":
        fi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        fi.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"{counter.upper()}: Class ±1 Acc={m_te['macro_acc_pm1']*100:.2f}%  MAE={m_te['macro_class_mae']:.4f} -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Unified E2E pipeline: inference -> counting -> evaluation")
    p.add_argument("--name", required=True, help="Experiment name, e.g. y26mv2")
    p.add_argument("--weights", type=Path, default=None,
                   help="Path to YOLO .pt weights (default: models/yolo/{name}.pt; only required for inference)")
    p.add_argument("--data", type=Path, default=ROOT / "ground_truth",
                   help="Annotations + split root (default: ./ground_truth/). "
                        "For full pipeline including YOLO inference, point this at ./SawitMVC-YOLO/ "
                        "so the images/ subdir is also available.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip Step 1 and use existing predictions/{name}_per_tree/")
    p.add_argument("--counters", nargs="+", default=["svm", "rf", "lr", "m01"],
                   choices=["svm", "rf", "lr", "m01"], help="Which counters to run")
    args = p.parse_args()

    infer_dir = ROOT / "predictions" / f"{args.name}_per_tree"
    gt_dir    = _resolve_gt_dir(args.data)
    e2e_base  = ROOT / "results" / "e2e_per_tree"

    # Step 1: Inference
    if not args.skip_inference:
        weights = args.weights or (ROOT / "models" / "yolo" / f"{args.name}.pt")
        if not weights.exists():
            print(f"ERROR: weights {weights} not found.")
            raise SystemExit(1)
        print(f"\n[Step 1] Running YOLO inference: {args.name}")
        step_inference(args.name, weights, args.data, infer_dir, args.conf)
    else:
        if not infer_dir.exists():
            print(f"ERROR: --skip-inference but {infer_dir} not found. Run without --skip-inference first.")
            raise SystemExit(1)
        print(f"[Step 1] Skipped. Using existing {infer_dir} ({len(list(infer_dir.glob('*.json')))} trees)")

    # Steps 2+3: Count
    print(f"\n[Steps 2-3] Counting with: {args.counters}")
    if "m01" in args.counters:
        step_m01(args.name, infer_dir, gt_dir, args.data, e2e_base / f"{args.name}_m01")
    if "svm" in args.counters:
        step_ml(args.name, "svm", infer_dir, gt_dir, args.data, e2e_base / f"{args.name}_svm")
    if "lr" in args.counters:
        step_ml(args.name, "lr",  infer_dir, gt_dir, args.data, e2e_base / f"{args.name}_lr")
    if "rf" in args.counters:
        step_ml(args.name, "rf",  infer_dir, gt_dir, args.data, e2e_base / f"{args.name}_rf")

    print(f"\nDone. Results in {e2e_base}/")


if __name__ == "__main__":
    main()
