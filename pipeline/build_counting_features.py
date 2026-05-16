"""
Build 13-dim counting features dari JSON GT atau inference JSON.
Output: X (n_trees, 13), y (n_trees, 4), tree_ids, splits
"""
import json, csv, os, glob
import numpy as np

CLASSES = ["B1", "B2", "B3", "B4"]
JSON_DIR = os.path.join(os.path.dirname(__file__), "..", "Brand-New-Dataset-YOLO", "json")
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "Brand-New-Dataset-YOLO", "split_manifest.csv")


def load_split_manifest(manifest_path=MANIFEST):
    splits = {}
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            splits[row["tree_id"]] = row["new_split"]
    return splits


def extract_features_from_json(data: dict) -> np.ndarray:
    """Extract 13-dim feature vector from one tree JSON (GT or inference)."""
    sides = data["images"]
    n_sides = len(sides)

    per_side_counts = {c: [] for c in CLASSES}
    for side_data in sides.values():
        anns = side_data.get("annotations", [])
        counts = {c: 0 for c in CLASSES}
        for a in anns:
            cn = a.get("class_name") or a.get("class")
            if cn in counts:
                counts[cn] += 1
        for c in CLASSES:
            per_side_counts[c].append(counts[c])

    feats = []
    # naive_sum (4)
    for c in CLASSES:
        feats.append(float(sum(per_side_counts[c])))
    # max_per_side (4)
    for c in CLASSES:
        feats.append(float(max(per_side_counts[c])) if per_side_counts[c] else 0.0)
    # mean_per_side (4)
    for c in CLASSES:
        arr = per_side_counts[c]
        feats.append(float(sum(arr) / len(arr)) if arr else 0.0)
    # n_sides (1)
    feats.append(float(n_sides))
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = (
    [f"naive_sum_{c}" for c in CLASSES]
    + [f"max_per_side_{c}" for c in CLASSES]
    + [f"mean_per_side_{c}" for c in CLASSES]
    + ["n_sides"]
)


def load_dataset(json_dir=JSON_DIR, manifest_path=MANIFEST):
    """Load all 953 trees → returns (X, y, tree_ids, splits)."""
    splits = load_split_manifest(manifest_path)
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    X, y, tree_ids, tree_splits = [], [], [], []
    for fp in files:
        with open(fp, encoding="utf-8-sig") as f:
            data = json.load(f)
        tree_id = data.get("tree_name") or data.get("tree_id") or os.path.splitext(os.path.basename(fp))[0]
        feats = extract_features_from_json(data)
        gt = data["summary"]["by_class"]
        label = np.array([gt.get(c, 0) for c in CLASSES], dtype=np.float32)
        split = splits.get(tree_id, data.get("split", "train"))
        X.append(feats)
        y.append(label)
        tree_ids.append(tree_id)
        tree_splits.append(split)
    return np.array(X), np.array(y), tree_ids, tree_splits


if __name__ == "__main__":
    X, y, ids, splits = load_dataset()
    print(f"Loaded {len(ids)} trees | X: {X.shape} | y: {y.shape}")
    print("Feature names:", FEATURE_NAMES)
    from collections import Counter
    print("Split dist:", Counter(splits))
