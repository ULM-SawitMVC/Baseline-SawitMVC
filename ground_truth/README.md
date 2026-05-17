# Ground Truth — SawitMVC-YOLO Annotations

This folder bundles every artifact needed to reproduce the SawitMVC baseline
without contacting an external dataset host. With this folder committed, both
the heuristic benchmark and the end-to-end pipeline run from a fresh clone.

## Contents

| Path | Description |
|------|-------------|
| [`annotations/`](annotations/) | 953 per-tree JSON files (854 DAMIMAS + 99 LONSUM) with bounding boxes, class labels, and confirmed cross-view bunch links. |
| [`split_manifest.csv`](split_manifest.csv) | Canonical train/val/test assignment for every tree. 763 train, 95 val, 95 test, stratified by `variety` × `dominant_class`. |
| [`data.yaml`](data.yaml) | YOLO dataset descriptor: class names (B1, B2, B3, B4) and image subdirectory layout. |

The corresponding JPEG images are not included; they live on Hugging Face
([`ULM-DS-Lab/SawitMVC-YOLO`](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO))
and are only required when retraining a detector. Heuristic deduplication,
feature extraction, all four counter families, and every metric in
[`results/`](../results/) operate purely on the JSON annotations and the
cached predictions in [`predictions/`](../predictions/).

## Split manifest schema

```
tree_id, variety, dominant_class, strat_key, B1, B2, B3, B4, new_split
```

- `tree_id` — canonical identifier, e.g. `DAMIMAS_A21B_0001`.
- `variety` — plantation source (`DAMIMAS` or `LONSUM`).
- `dominant_class` — most abundant maturity class on the tree, used for stratification.
- `strat_key` — `{variety}_{dominant_class}`.
- `B1…B4` — unique bunch counts per maturity class for that tree.
- `new_split` — one of `train`, `val`, `test`.

Sample rows (first three trees):

```
DAMIMAS_A21B_0001,DAMIMAS,B3,DAMIMAS_B3,1,2,5,0,train
DAMIMAS_A21B_0002,DAMIMAS,B3,DAMIMAS_B3,1,0,7,4,train
DAMIMAS_A21B_0003,DAMIMAS,B3,DAMIMAS_B3,1,2,5,1,train
```

## Annotation schema (per tree)

Each `annotations/{tree_id}.json` file is a self-contained record:

```jsonc
{
  "version": 4,
  "tree_id": "DAMIMAS_A21B_0001",
  "tree_name": "...",
  "split": "train",
  "metadata": {
    "date": "...",
    "session_id": "...",
    "variety": "DAMIMAS"
  },
  "images": {
    "side_1": {
      "filename": "...jpg",
      "label_file": "...txt",
      "side_index": 0,
      "side_label": "Side 1",
      "width": 960,
      "height": 1280,
      "bbox_count": 7,
      "annotations": [
        {
          "box_index": 0,
          "class_id": 2,
          "class_name": "B3",
          "bbox_yolo": [cx, cy, w, h],     // normalized [0, 1]
          "bbox_pixel": [x1, y1, x2, y2]   // pixel coordinates
        }
      ]
    },
    "side_2": { ... },
    "side_3": { ... },
    "side_4": { ... }
  },
  "bunches": [
    {
      "bunch_id": 1,
      "class": "B3",
      "class_mismatch": false,
      "appearance_count": 3,
      "appearances": [
        {"side": "side_1", "side_index": 0, "box_index": 0,
         "class_name": "B3", "bbox_pixel": [...]}
      ]
    }
  ],
  "_confirmedLinks": [ ... ],
  "summary": {
    "total_by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0},
    "total_by_side": {"side_1": 7, ...}
  }
}
```

The `bunches` array is the human-confirmed ground truth: each entry represents
one physical fruit bunch and lists every camera angle in which it appears.
Naive summation of per-side detections overcounts because the same bunch may
appear in up to four `appearances`. The heuristics in
[`algorithms/`](../algorithms/) and the ML counters in
[`pipeline/`](../pipeline/) exist to recover the unique count per class from
the inflated per-side observations.

## Class taxonomy

| Class | Visual cue | Position on tree |
|:-----:|------------|------------------|
| B1 | Red, large, round | Lowest (most mature) |
| B2 | Black with red transition | Above B1 |
| B3 | Solid black, spiky, elongated | Above B2 |
| B4 | Smallest, dark green-black | Highest (newest) |

## Provenance

This bundle is a verbatim copy of the `json/`, `split_manifest.csv`, and
`data.yaml` artifacts of the `ULM-DS-Lab/SawitMVC-YOLO` Hugging Face dataset
release. Annotation counts, schema, and class taxonomy match the upstream
publication. See [`docs/dataset.md`](../docs/dataset.md) for collection
methodology and quality-assurance details.
