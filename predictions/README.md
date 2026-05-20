# Predictions: Current Cached YOLO Outputs

This folder contains only the cached predictions used by the latest baseline:
[`y26mv2_per_tree/`](y26mv2_per_tree/). It is the current per-tree output of
the `y26mv2` detector and is sufficient to reproduce Track B without rerunning
YOLO inference.

## Current layout

| Folder | Detector | Files | Description |
|--------|----------|------:|-------------|
| [`y26mv2_per_tree/`](y26mv2_per_tree/) | YOLOv26 medium (`y26mv2`) | 953 | One JSON per tree; camera sides grouped under `images.side_*`. |

## Per-tree JSON schema

```jsonc
{
  "tree_name": "DAMIMAS_A21B_0001",
  "split": "train",
  "detector": "y26mv2",
  "images": {
    "side_1": {
      "side_index": 0,
      "annotations": [
        {
          "class_name": "B3",
          "bbox_yolo": [0.512, 0.341, 0.08, 0.12],
          "conf": 0.87
        }
      ]
    }
  }
}
```

This is a strict subset of the ground-truth schema in
[`ground_truth/README.md`](../ground_truth/README.md): the prediction JSON keeps
the per-side detection lists and confidence scores, but omits the GT-specific
`bunches`, `_confirmedLinks`, and `summary` blocks.

## Reproducing Track B

```bash
python pipeline/run_e2e_pipeline.py --name y26mv2 --skip-inference
```

## Archived prediction sets

Legacy detector outputs (`y26n`, `y26s`, `y26m`) and the per-image experiment
artifacts are kept under [`archive/predictions/`](../archive/predictions/).
They are historical experiments and are not part of the latest baseline
surface.
