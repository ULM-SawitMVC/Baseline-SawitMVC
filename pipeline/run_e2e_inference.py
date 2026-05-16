"""
YOLO inference — per tree (all sides grouped into 1 JSON per tree).

For each tree, runs YOLO on every side image and saves all detections
in a single JSON with this structure:
    {
        "tree_name": "DAMIMAS_A21B_0001",
        "split": "train",
        "images": {
            "side_1": {"side_index": 0, "annotations": [{"class_name": "B3", "bbox_yolo": [...], "conf": 0.87}]},
            "side_2": {"side_index": 1, "annotations": [...]},
            ...
        }
    }

Usage:
    python pipeline/run_e2e_inference.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt
    python pipeline/run_e2e_inference.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt --data ./SawitMVC-YOLO/
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}
ROOT = Path(__file__).resolve().parent.parent


def _load_splits(data_dir: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    for sp in ("train", "val", "test"):
        list_file = data_dir / f"{sp}.txt"
        if not list_file.exists():
            continue
        for line in list_file.read_text(encoding="utf-8").splitlines():
            fname = Path(line.strip()).name
            if fname:
                splits[fname] = sp
    return splits


def _group_by_tree(data_dir: Path, split_map: dict[str, str]) -> dict[str, list[tuple[str, int, Path]]]:
    """
    Returns {tree_id: [(split, side_index, img_path), ...]}
    Image name format: {tree_id}_{side_number}.jpg  e.g. DAMIMAS_A21B_0001_2.jpg
    """
    trees: dict[str, list] = defaultdict(list)
    for img_path in sorted((data_dir / "images").rglob("*.jpg")):
        fname = img_path.name  # e.g. DAMIMAS_A21B_0001_2.jpg
        stem = img_path.stem   # DAMIMAS_A21B_0001_2
        # Split at last underscore to separate tree_id from side number
        m = re.match(r"^(.+)_(\d+)$", stem)
        if not m:
            continue
        tree_id, side_num = m.group(1), int(m.group(2))
        side_index = side_num - 1  # 0-based
        split = split_map.get(fname, "unknown")
        trees[tree_id].append((split, side_index, img_path))
    # sort sides within each tree
    for tid in trees:
        trees[tid].sort(key=lambda x: x[1])
    return dict(trees)


def run_inference(name: str, weights: Path, data_dir: Path, out_dir: Path, conf: float = 0.25) -> None:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = _load_splits(data_dir)
    trees = _group_by_tree(data_dir, split_map)
    print(f"Found {len(trees)} trees | Output → {out_dir}")

    for tree_id, sides in sorted(trees.items()):
        out_path = out_dir / f"{tree_id}.json"
        if out_path.exists():
            continue  # resume-friendly: skip already processed trees

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
                    cls_id = int(r.boxes.cls[k].item())
                    xywhn = r.boxes.xywhn[k].tolist()
                    anns.append({
                        "box_index": k,
                        "class_name": CLASS_MAP.get(cls_id, f"cls{cls_id}"),
                        "bbox_yolo": xywhn,   # [x_center, y_center, width, height] normalized
                        "conf": float(r.boxes.conf[k].item()),
                    })
            images_data[side_label] = {
                "side_index": side_index,
                "annotations": anns,
            }

        out_path.write_text(json.dumps({
            "tree_name": tree_id,
            "split": split,
            "detector": name,
            "weights": str(weights),
            "images": images_data,
        }))

    print(f"Inference complete. {len(list(out_dir.glob('*.json')))} trees saved.")


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO inference — per tree (all sides → 1 JSON)")
    p.add_argument("--name", required=True, help="Experiment name (used for output folder)")
    p.add_argument("--weights", type=Path, required=True, help="Path to .pt weights file")
    p.add_argument("--data", type=Path, default=ROOT / "SawitMVC-YOLO",
                   help="Dataset root (default: ./SawitMVC-YOLO/)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output folder (default: predictions/{name}_inference/)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = p.parse_args()

    out_dir = args.out or ROOT / "predictions" / f"{args.name}_inference"
    run_inference(args.name, args.weights, args.data, out_dir, args.conf)


if __name__ == "__main__":
    main()
