"""
Step 1 untuk Exp #8/#9: jalankan y26s inference pada semua tree images.
Output: ml-track/predictions/y26s_vanilla_local_inference/<tree_name>.json (mirror schema GT)

Usage: python scripts/run_e2e_inference.py --weights <path/to/best.pt>
"""
import os, sys, json, glob, argparse
from pathlib import Path
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(REPO, "Brand-New-Dataset-YOLO")
IMG_ROOT = os.path.join(DATASET_ROOT, "images")
JSON_DIR = os.path.join(DATASET_ROOT, "json")
OUT_DIR = os.path.join(REPO, "ml-track", "predictions", "y26s_vanilla_local_inference")
CLASSES = ["B1", "B2", "B3", "B4"]
CLASS_MAP = {0: "B1", 1: "B2", 2: "B3", 3: "B4"}


def _load_split_map():
    """Build {filename.jpg: split} from train.txt/val.txt/test.txt."""
    split_map = {}
    for sp in ("train", "val", "test"):
        list_file = os.path.join(DATASET_ROOT, f"{sp}.txt")
        if not os.path.isfile(list_file):
            continue
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                fname = os.path.basename(line.strip())
                if fname:
                    split_map[fname] = sp
    return split_map


def get_tree_images():
    """Returns dict: tree_name -> list of (split, side_label, img_path)"""
    trees = defaultdict(list)
    split_map = _load_split_map()
    for img_path in sorted(glob.glob(os.path.join(IMG_ROOT, "*.jpg"))):
        basename = os.path.basename(img_path)
        fname = os.path.splitext(basename)[0]
        split = split_map.get(basename, "unknown")
        # filename format: TREENAME_N  e.g. DAMIMAS_A21B_0001_1
        parts = fname.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            tree_name, side_num = parts[0], parts[1]
            side_label = f"side_{side_num}"
        else:
            tree_name = fname
            side_label = "side_1"
        trees[tree_name].append((split, side_label, img_path))
    return trees


def run_inference(weights_path):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    trees = get_tree_images()
    print(f"Found {len(trees)} trees across splits.")

    for tree_name, sides in sorted(trees.items()):
        out_path = os.path.join(OUT_DIR, f"{tree_name}.json")
        if os.path.exists(out_path):
            continue  # resume-friendly

        images_data = {}
        split = sides[0][0]

        for _, side_label, img_path in sorted(sides, key=lambda x: x[1]):
            results = model(img_path, verbose=False, conf=0.25)
            anns = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for k in range(len(boxes)):
                    cls_id = int(boxes.cls[k].item())
                    cls_name = CLASS_MAP.get(cls_id, f"cls{cls_id}")
                    xywhn = boxes.xywhn[k].tolist()
                    anns.append({
                        "box_index": k,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "bbox_yolo": xywhn,
                        "conf": float(boxes.conf[k].item()),
                    })
            images_data[side_label] = {"annotations": anns}

        tree_json = {
            "tree_name": tree_name,
            "split": split,
            "source": "y26s_vanilla_local_inference",
            "weights": weights_path,
            "images": images_data,
        }
        with open(out_path, "w") as f:
            json.dump(tree_json, f)

    print(f"Inference complete. JSONs saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    args = parser.parse_args()
    run_inference(args.weights)
