"""
Unified E2E pipeline: inference + SVM + RF + M01 untuk satu detektor.

Usage:
  python scripts/run_e2e_pipeline.py --name y26n_vanilla_local \
      --weights ml-track/baseline-run/weights/y26n_vanilla_local.pt

Output:
  ml-track/predictions/{name}_inference/   <- YOLO inference JSONs
  reports/e2e_{name}_svm/                  <- SVM metrics
  reports/e2e_{name}_rf/                   <- RF metrics
  reports/e2e_{name}_m01/                  <- M01 heuristic metrics
"""
import os, sys, json, glob, csv, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO / "Brand-New-Dataset-YOLO"
IMG_ROOT = DATASET_ROOT / "images"
GT_JSON_DIR = DATASET_ROOT / "json"
MANIFEST = DATASET_ROOT / "split_manifest.csv"
CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


def _load_split_map():
    """Build {filename.jpg: split} from train.txt/val.txt/test.txt."""
    split_map = {}
    for sp in ("train", "val", "test"):
        list_file = DATASET_ROOT / f"{sp}.txt"
        if not list_file.is_file():
            continue
        for line in list_file.read_text(encoding="utf-8").splitlines():
            name = Path(line.strip()).name
            if name:
                split_map[name] = sp
    return split_map

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))


# ─── helpers ────────────────────────────────────────────────────────────────

def load_split_manifest():
    splits = {}
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            splits[row["tree_id"]] = row["new_split"]
    return splits


def load_gt_labels():
    labels = {}
    for fp in GT_JSON_DIR.glob("*.json"):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        name = d.get("tree_name") or d.get("tree_id") or fp.stem
        labels[name] = {c: d["summary"]["by_class"].get(c, 0) for c in CLASSES}
    return labels


def compute_metrics_svm_rf(y_true, y_pred, tree_ids):
    y_pred_r = np.clip(np.round(y_pred), 0, None).astype(int)
    y_true_i = y_true.astype(int)
    rows = []
    for i, tid in enumerate(tree_ids):
        rows.append(dict(
            tree_id=tid,
            **{f"pred_{c}": y_pred_r[i, j] for j, c in enumerate(CLASSES)},
            **{f"gt_{c}": y_true_i[i, j] for j, c in enumerate(CLASSES)},
        ))
    df = pd.DataFrame(rows)
    m = {}
    for j, c in enumerate(CLASSES):
        err = y_pred_r[:, j] - y_true_i[:, j]
        m[f"MAE_{c}"] = float(np.mean(np.abs(err)))
        m[f"bias_{c}"] = float(np.mean(err))
        m[f"acc_pm1_{c}"] = float(np.mean(np.abs(err) <= 1))
    m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
    m["macro_acc_pm1"] = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
    m["exact_profile_acc"] = float(np.mean(np.all(y_pred_r == y_true_i, axis=1)))
    total_err = y_pred_r.sum(axis=1) - y_true_i.sum(axis=1)
    m["total_count_mae"] = float(np.mean(np.abs(total_err)))
    m["total_pm1_acc"] = float(np.mean(np.abs(total_err) <= 1))
    return m, df


# ─── Step 1: Inference ──────────────────────────────────────────────────────

def run_inference(name, weights_path):
    from ultralytics import YOLO
    out_dir = REPO / "ml-track" / "predictions" / f"{name}_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = defaultdict(list)
    split_map = _load_split_map()
    for img_path in sorted(IMG_ROOT.glob("*.jpg")):
        split = split_map.get(img_path.name, "unknown")
        parts = img_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            tree_name, side_num = parts[0], parts[1]
            side_label = f"side_{side_num}"
        else:
            tree_name = img_path.stem
            side_label = "side_1"
        trees[tree_name].append((split, side_label, img_path))

    existing = {p.stem for p in out_dir.glob("*.json")}
    remaining = [t for t in sorted(trees) if t not in existing]
    print(f"[inference] {len(existing)} cached, {len(remaining)} to process.")
    if not remaining:
        print("[inference] All done, skipping.")
        return out_dir

    model = YOLO(weights_path)
    for tree_name in remaining:
        sides = trees[tree_name]
        split = sides[0][0]
        images_data = {}
        for _, side_label, img_path in sorted(sides, key=lambda x: x[1]):
            results = model(str(img_path), verbose=False, conf=0.25)
            anns = []
            for r in results:
                if r.boxes is None:
                    continue
                for k in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[k].item())
                    anns.append({
                        "box_index": k,
                        "class_id": cls_id,
                        "class_name": CLASS_MAP.get(cls_id, f"cls{cls_id}"),
                        "bbox_yolo": r.boxes.xywhn[k].tolist(),
                        "conf": float(r.boxes.conf[k].item()),
                    })
            images_data[side_label] = {"annotations": anns}
        out = {"tree_name": tree_name, "split": split, "source": f"{name}_inference",
               "weights": str(weights_path), "images": images_data}
        (out_dir / f"{tree_name}.json").write_text(json.dumps(out))

    print(f"[inference] Done → {out_dir}")
    return out_dir


# ─── Step 2: SVM ────────────────────────────────────────────────────────────

def run_svm(name, infer_dir, report_dir):
    from build_counting_features import extract_features_from_json, FEATURE_NAMES
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    report_dir.mkdir(parents=True, exist_ok=True)
    splits = load_split_manifest()
    gt = load_gt_labels()

    X, y, ids, tree_splits = [], [], [], []
    for fp in sorted(infer_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        tn = d.get("tree_name") or fp.stem
        if tn not in gt:
            continue
        X.append(extract_features_from_json(d))
        y.append([gt[tn][c] for c in CLASSES])
        ids.append(tn)
        tree_splits.append(splits.get(tn, d.get("split", "train")))

    X, y, tree_splits = np.array(X), np.array(y, dtype=np.float32), np.array(tree_splits)
    tr, va, te = X[tree_splits=="train"], X[tree_splits=="val"], X[tree_splits=="test"]
    yt, yv, yte = y[tree_splits=="train"], y[tree_splits=="val"], y[tree_splits=="test"]
    ids_te = [ids[i] for i in np.where(tree_splits=="test")[0]]
    ids_va = [ids[i] for i in np.where(tree_splits=="val")[0]]
    print(f"  [SVM] Train={len(tr)} Val={len(va)} Test={len(te)}")

    pipe = Pipeline([("sc", StandardScaler()), ("svr", MultiOutputRegressor(SVR(kernel="rbf")))])
    gs = GridSearchCV(pipe, {"svr__estimator__C": [0.1, 1, 10], "svr__estimator__gamma": ["scale", 0.01, 0.1]},
                      cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(tr, yt)

    mt, dt = compute_metrics_svm_rf(yte, gs.predict(te), ids_te)
    mv, _ = compute_metrics_svm_rf(yv, gs.predict(va), ids_va)
    mt.update({"best_params": str(gs.best_params_), "cv_mae": float(-gs.best_score_), "split": "test"})
    mv["split"] = "val"

    (report_dir / "metrics.json").write_text(json.dumps({"test": mt, "val": mv}, indent=2))
    dt.to_csv(report_dir / "predictions.csv", index=False)
    print(f"  [SVM] Macro Acc±1={mt['macro_acc_pm1']*100:.1f}%  MAE={mt['macro_class_mae']:.3f}")


# ─── Step 3: RF ─────────────────────────────────────────────────────────────

def run_rf(name, infer_dir, report_dir):
    from build_counting_features import extract_features_from_json, FEATURE_NAMES
    from sklearn.ensemble import RandomForestRegressor

    report_dir.mkdir(parents=True, exist_ok=True)
    splits = load_split_manifest()
    gt = load_gt_labels()

    X, y, ids, tree_splits = [], [], [], []
    for fp in sorted(infer_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        tn = d.get("tree_name") or fp.stem
        if tn not in gt:
            continue
        X.append(extract_features_from_json(d))
        y.append([gt[tn][c] for c in CLASSES])
        ids.append(tn)
        tree_splits.append(splits.get(tn, d.get("split", "train")))

    X, y, tree_splits = np.array(X), np.array(y, dtype=np.float32), np.array(tree_splits)
    tr, va, te = X[tree_splits=="train"], X[tree_splits=="val"], X[tree_splits=="test"]
    yt, yv, yte = y[tree_splits=="train"], y[tree_splits=="val"], y[tree_splits=="test"]
    ids_te = [ids[i] for i in np.where(tree_splits=="test")[0]]
    ids_va = [ids[i] for i in np.where(tree_splits=="val")[0]]
    print(f"  [RF]  Train={len(tr)} Val={len(va)} Test={len(te)}")

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(tr, yt)

    mt, dt = compute_metrics_svm_rf(yte, rf.predict(te), ids_te)
    mv, _ = compute_metrics_svm_rf(yv, rf.predict(va), ids_va)
    mt["split"] = "test"
    mv["split"] = "val"

    fi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    (report_dir / "metrics.json").write_text(json.dumps({"test": mt, "val": mv}, indent=2))
    dt.to_csv(report_dir / "predictions.csv", index=False)
    fi.to_csv(report_dir / "feature_importance.csv", index=False)
    print(f"  [RF]  Macro Acc±1={mt['macro_acc_pm1']*100:.1f}%  MAE={mt['macro_class_mae']:.3f}")


# ─── Step 4: M01 ────────────────────────────────────────────────────────────

def run_m01(name, infer_dir, report_dir):
    from algorithms.M01_selector_b2b3 import predict as m01_predict

    report_dir.mkdir(parents=True, exist_ok=True)
    splits = load_split_manifest()
    gt = load_gt_labels()

    rows = []
    for fp in sorted(infer_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        tn = d.get("tree_name") or fp.stem
        if tn not in gt:
            continue
        dets = []
        for side_label, sd in d["images"].items():
            try:
                si = int(side_label.replace("side_", "").replace("si" + "si_", "")) - 1
            except ValueError:
                si = 0
            for ann in sd.get("annotations", []):
                cx, cy, w, h = ann["bbox_yolo"]
                dets.append({"class": ann["class_name"], "x_norm": cx, "y_norm": cy,
                             "side_index": si, "area_norm": w*h,
                             "aspect_ratio": w/h if h > 0 else 1.0, "side": side_label})
        pred = m01_predict(dets)
        split = splits.get(tn, d.get("split", "train"))
        row = {"tree_name": tn, "split": split}
        for c in CLASSES:
            row[f"pred_{c}"] = pred.get(c, 0)
            row[f"gt_{c}"] = gt[tn].get(c, 0)
        rows.append(row)

    def _metrics(subset):
        if not subset:
            return {}
        m = {}
        for c in CLASSES:
            errs = [abs(r[f"pred_{c}"] - r[f"gt_{c}"]) for r in subset]
            biases = [r[f"pred_{c}"] - r[f"gt_{c}"] for r in subset]
            m[f"MAE_{c}"] = float(np.mean(errs))
            m[f"bias_{c}"] = float(np.mean(biases))
            m[f"acc_pm1_{c}"] = float(np.mean([e <= 1 for e in errs]))
        m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
        m["macro_acc_pm1"] = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
        total_errs = [abs(sum(r[f"pred_{c}"] for c in CLASSES) - sum(r[f"gt_{c}"] for c in CLASSES)) for r in subset]
        m["total_count_mae"] = float(np.mean(total_errs))
        m["total_pm1_acc"] = float(np.mean([e <= 1 for e in total_errs]))
        m["exact_profile_acc"] = float(np.mean([all(r[f"pred_{c}"] == r[f"gt_{c}"] for c in CLASSES) for r in subset]))
        m["n_trees"] = len(subset)
        return m

    mt = _metrics([r for r in rows if r["split"] == "test"])
    mv = _metrics([r for r in rows if r["split"] == "val"])
    mt["split"] = "test"
    mv["split"] = "val"

    (report_dir / "metrics.json").write_text(json.dumps({"test": mt, "val": mv}, indent=2))
    with open(report_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [M01] Macro Acc±1={mt['macro_acc_pm1']*100:.1f}%  MAE={mt['macro_class_mae']:.3f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name, e.g. y26n_vanilla_local")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference step")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"E2E Pipeline: {args.name}")
    print(f"Weights: {weights}")
    print(f"{'='*60}")

    infer_dir = REPO / "ml-track" / "predictions" / f"{args.name}_inference"

    if not args.skip_inference:
        print("\n[Step 1] Running inference...")
        run_inference(args.name, str(weights))
    else:
        print(f"\n[Step 1] Skipping inference (using {infer_dir})")

    if not any(infer_dir.glob("*.json")):
        print(f"ERROR: No inference JSONs found in {infer_dir}")
        sys.exit(1)

    print("\n[Step 2] Running SVM...")
    run_svm(args.name, infer_dir, REPO / "reports" / f"e2e_{args.name}_svm")

    print("\n[Step 3] Running RF...")
    run_rf(args.name, infer_dir, REPO / "reports" / f"e2e_{args.name}_rf")

    print("\n[Step 4] Running M01 heuristic...")
    run_m01(args.name, infer_dir, REPO / "reports" / f"e2e_{args.name}_m01")

    print(f"\nDone: {args.name}")


if __name__ == "__main__":
    main()
