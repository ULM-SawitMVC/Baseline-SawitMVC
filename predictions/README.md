# Predictions — Cached YOLO Outputs

This folder mirrors every YOLOv26 inference output used by the end-to-end
tracks. Bundling these JSONs (`~48 MB`) lets contributors reproduce every
metric in [`results/`](../results/) without a GPU and without rerunning
detection.

## Layout

| Folder | Detector | Files | Description |
|--------|----------|------:|-------------|
| [`y26m_per_tree/`](y26m_per_tree/) | YOLOv26 medium | 953 | One JSON per tree; sides grouped under `images.side_*`. |
| [`y26n_per_tree/`](y26n_per_tree/) | YOLOv26 nano | 953 | Same schema. |
| [`y26s_per_tree/`](y26s_per_tree/) | YOLOv26 small | 953 | Same schema. |
| [`y26m_per_image/`](y26m_per_image/) | derived from per-tree | 3,992 | One JSON per image; flat detection list. |
| [`y26n_per_image/`](y26n_per_image/) | derived | 3,992 | Same. |
| [`y26s_per_image/`](y26s_per_image/) | derived | 3,992 | Same. |

The per-image folders are derived programmatically from their per-tree
counterparts by [`pipeline/run_e2e_per_image.py`](../pipeline/run_e2e_per_image.py)
on `--skip-inference`, so they will be repopulated automatically if you
delete them.

## Per-tree JSON schema

```jsonc
{
  "tree_name": "DAMIMAS_A21B_0001",
  "split":     "train",
  "detector":  "y26s",
  "images": {
    "side_1": {
      "side_index": 0,
      "annotations": [
        {
          "class_name": "B3",
          "bbox_yolo":  [0.512, 0.341, 0.08, 0.12],   // normalized cx, cy, w, h
          "conf":       0.87
        }
      ]
    },
    "side_2": { ... }
  }
}
```

This is a strict subset of the ground-truth schema (see
[`ground_truth/README.md`](../ground_truth/README.md)) with a `conf` field
added and the `bunches`, `_confirmedLinks`, and `summary` blocks omitted.

## Per-image JSON schema

```jsonc
{
  "image":      "DAMIMAS_A21B_0001_2.jpg",
  "detector":   "y26s",
  "side_index": 1,
  "detections": [
    {
      "class_name": "B3",
      "bbox_yolo":  [0.512, 0.341, 0.08, 0.12],
      "conf":       0.87
    }
  ],
  "count_per_class": {"B1": 0, "B2": 1, "B3": 5, "B4": 0}
}
```

## Choosing per-tree vs. per-image

| Question | Use |
|----------|-----|
| Are you running an ML or heuristic counter? | Either; both pipelines reconstruct identical 13-dim features. |
| Are you computing simple aggregation (`max`/`mean`/`sum`)? | Per-image. |
| Are you experimenting with new features that need raw per-side counts? | Per-tree (more compact). |

## Regenerating predictions

Required only when you change a detector or want to confirm the cache. The
GPU step is `pipeline/run_e2e_inference.py`; everything downstream is CPU only.

```bash
# Per-tree predictions for one detector (GPU required)
python pipeline/run_e2e_inference.py --name y26s \
    --weights models/yolo/y26s.pt --data ./SawitMVC-YOLO/

# Per-image predictions: derived from per-tree, no GPU needed
python pipeline/run_e2e_per_image.py --name y26s --skip-inference
```
