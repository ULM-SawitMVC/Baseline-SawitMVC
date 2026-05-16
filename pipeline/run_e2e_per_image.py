"""
Simplified E2E pipeline — per-image inference, aggregate by tree with max-per-image.

Unlike run_e2e_pipeline.py (which groups all sides into 1 tree-level JSON), this script:
  1. Runs YOLO on each image independently → saves per-image detection counts
  2. Aggregates per tree using max-per-image per class (a simple dedup heuristic)
  3. Evaluates against ground truth

This is the simplest possible E2E baseline:
  - No complex feature engineering
  - No ML training (uses max across images as the count estimate)
  - Runs on a single image at a time — easy to deploy on arbitrary images

Aggregation options (--agg):
  max    : unique_count[c] = max detections of class c in any single image  (default)
  mean   : unique_count[c] = round(mean detections per image)
  sum    : unique_count[c] = total detections (naive sum, no dedup — baseline)

Usage:
    python pipeline/run_e2e_per_image.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt
    python pipeline/run_e2e_per_image.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt --agg mean
    python pipeline/run_e2e_per_image.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt --skip-inference
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


# ---------------------------------------------------------------------------
# Step 1: Per-image inference
# ---------------------------------------------------------------------------

def run_inference_per_image(name: str, weights: Path, data_dir: Path,
                             img_out_dir: Path, conf: float = 0.25) -> None:
    """Run YOLO on each image independently. Saves one JSON per image."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    img_out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((data_dir / "images").rglob("*.jpg"))
    print(f"Running inference on {len(images)} images → {img_out_dir}")

    done = 0
    for img_path in images:
        out_path = img_out_dir / f"{img_path.stem}.json"
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

        # Count detections per class in this image
        count_per_class = {c: sum(1 for d in detections if d["class_name"] == c) for c in CLASSES}

        out_path.write_text(json.dumps({
            "image":      img_path.name,
            "detector":   name,
            "detections": detections,
            "count_per_class": count_per_class,
        }))
        done += 1

    print(f"Inference complete: {done} images.")


# ---------------------------------------------------------------------------
# Step 2: Aggregate per-image counts to per-tree counts
# ---------------------------------------------------------------------------

def aggregate_per_tree(img_out_dir: Path, data_dir: Path, agg: str) -> dict[str, dict]:
    """
    Group per-image JSONs by tree and aggregate class counts.

    Returns:
        {tree_id: {"pred": {B1:int,...}, "split": str}}
    """
    # Build {tree_id: [count_per_class, ...]} across all images of that tree
    tree_counts: dict[str, list[dict[str, int]]] = defaultdict(list)
    tree_splits: dict[str, str] = {}

    # Load split membership
    split_map: dict[str, str] = {}
    for sp in ("train", "val", "test"):
        lf = data_dir / f"{sp}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                fname = Path(line.strip()).name
                if fname:
                    split_map[fname] = sp

    for jp in sorted(img_out_dir.glob("*.json")):
        d = json.loads(jp.read_text())
        img_name = d["image"]
        m = re.match(r"^(.+)_(\d+)\.jpg$", img_name)
        if not m:
            continue
        tree_id = m.group(1)
        tree_counts[tree_id].append(d["count_per_class"])
        tree_splits[tree_id] = split_map.get(img_name, "unknown")

    results: dict[str, dict] = {}
    for tree_id, counts_list in tree_counts.items():
        if agg == "max":
            pred = {c: max(cnt.get(c, 0) for cnt in counts_list) for c in CLASSES}
        elif agg == "mean":
            pred = {c: round(sum(cnt.get(c, 0) for cnt in counts_list) / len(counts_list))
                    for c in CLASSES}
        elif agg == "sum":
            pred = {c: sum(cnt.get(c, 0) for cnt in counts_list) for c in CLASSES}
        else:
            raise ValueError(f"Unknown aggregation: {agg}")
        results[tree_id] = {"pred": pred, "split": tree_splits.get(tree_id, "unknown")}

    return results


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------

def evaluate(preds: dict[str, dict], gt_dir: Path, out_dir: Path, name: str, agg: str) -> None:
    # Load GT
    gt_map: dict[str, dict] = {}
    for fp in sorted(gt_dir.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8-sig"))
        tid = d.get("tree_id") or d.get("tree_name") or fp.stem
        by_class = d.get("summary", {}).get("by_class", {})
        gt_split = d.get("split", "train")
        gt_map[tid] = {"gt": {c: int(by_class.get(c, 0)) for c in CLASSES}, "split": gt_split}

    rows = []
    for tid, info in preds.items():
        if tid not in gt_map:
            continue
        gt = gt_map[tid]["gt"]
        split = gt_map[tid]["split"]
        row = {"tree_id": tid, "split": split}
        for c in CLASSES:
            row[f"pred_{c}"] = info["pred"].get(c, 0)
            row[f"gt_{c}"]   = gt.get(c, 0)
        rows.append(row)

    def _metrics(subset_rows: list[dict], split_name: str) -> dict:
        if not subset_rows:
            return {}
        m: dict = {}
        for c in CLASSES:
            errs = [abs(r[f"pred_{c}"] - r[f"gt_{c}"]) for r in subset_rows]
            m[f"MAE_{c}"]     = float(np.mean(errs))
            m[f"bias_{c}"]    = float(np.mean([r[f"pred_{c}"] - r[f"gt_{c}"] for r in subset_rows]))
            m[f"acc_pm1_{c}"] = float(np.mean([e <= 1 for e in errs]))
        m["macro_class_mae"] = float(np.mean([m[f"MAE_{c}"] for c in CLASSES]))
        m["macro_acc_pm1"]   = float(np.mean([m[f"acc_pm1_{c}"] for c in CLASSES]))
        total_errs = [abs(sum(r[f"pred_{c}"] for c in CLASSES) - sum(r[f"gt_{c}"] for c in CLASSES))
                      for r in subset_rows]
        m["total_count_mae"] = float(np.mean(total_errs))
        m["total_pm1_acc"]   = float(np.mean([e <= 1 for e in total_errs]))
        m["exact_profile_acc"] = float(np.mean(
            [all(r[f"pred_{c}"] == r[f"gt_{c}"] for c in CLASSES) for r in subset_rows]))
        m["split"] = split_name
        m["aggregation"] = agg
        return m

    m_test = _metrics([r for r in rows if r["split"] == "test"], "test")
    m_val  = _metrics([r for r in rows if r["split"] == "val"],  "val")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"test": m_test, "val": m_val}, indent=2))
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n=== {name} → per-image ({agg}) | test n={len([r for r in rows if r['split']=='test'])} ===")
    print(f"  Macro Acc±1 : {m_test.get('macro_acc_pm1', 0)*100:.2f}%")
    print(f"  Macro MAE   : {m_test.get('macro_class_mae', 0):.4f}")
    print(f"  Total MAE   : {m_test.get('total_count_mae', 0):.4f}")
    for c in CLASSES:
        print(f"  {c}: Acc±1={m_test.get(f'acc_pm1_{c}', 0)*100:.1f}%  MAE={m_test.get(f'MAE_{c}', 0):.3f}")
    print(f"\nResults saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Simplified E2E: per-image inference + aggregate by tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aggregation methods (--agg):
  max   count[c] = max detections of class c in any single image  (default)
  mean  count[c] = round(mean detections per image)
  sum   count[c] = total detections across all images (naive sum, no dedup)
        """)
    p.add_argument("--name", required=True, help="Experiment name, e.g. y26n_vanilla_local")
    p.add_argument("--weights", type=Path, required=True, help="Path to YOLO .pt weights")
    p.add_argument("--data", type=Path, default=ROOT / "SawitMVC-YOLO",
                   help="Dataset root (default: ./SawitMVC-YOLO/)")
    p.add_argument("--agg", choices=["max", "mean", "sum"], default="max",
                   help="Aggregation method across images of the same tree")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip inference, use existing per-image JSONs")
    args = p.parse_args()

    img_out_dir = ROOT / "predictions" / f"{args.name}_per_image"
    gt_dir      = args.data / "json"
    out_dir     = ROOT / "benchmarks" / "e2e" / f"e2e_{args.name}_per_image_{args.agg}"

    # Step 1: Inference
    if not args.skip_inference:
        print(f"\n[Step 1] Per-image inference: {args.name}")
        run_inference_per_image(args.name, args.weights, args.data, img_out_dir, args.conf)
    else:
        n = len(list(img_out_dir.glob("*.json"))) if img_out_dir.exists() else 0
        print(f"[Step 1] Skipped. Using {n} existing images in {img_out_dir}")

    # Step 2: Aggregate
    print(f"\n[Step 2] Aggregating per tree using: {args.agg}")
    preds = aggregate_per_tree(img_out_dir, args.data, args.agg)
    print(f"  Aggregated {len(preds)} trees.")

    # Step 3: Evaluate
    print(f"\n[Step 3] Evaluating against GT ...")
    evaluate(preds, gt_dir, out_dir, args.name, args.agg)


if __name__ == "__main__":
    main()
