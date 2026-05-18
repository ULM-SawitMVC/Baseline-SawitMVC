# Dataset — SawitMVC-YOLO

**SawitMVC-YOLO** is a multi-view oil palm bunch detection and counting dataset
collected from two commercial plantations in Kabupaten Tanah Laut, Kalimantan Selatan Selatan, Indonesia.

---

## Overview

| Property | Value |
|----------|-------|
| Trees | 953 |
| Images | 3,992 JPEG (960 × 1280 px) |
| Camera views per tree | 4 (95.3% of trees) or 8 (4.7%) |
| Unique bunches (ground truth) | 9,823 |
| Raw detections (YOLO labels) | 18,544 |
| Deduplication ratio | 0.53 (naive sum overcounts 1.83×) |
| Maturity classes | 4 (B1, B2, B3, B4) |
| Annotation format | YOLO TXT + JSON with cross-view links |
| License | CC BY-NC 4.0 |

---

## Sources

| Source | Trees | Region |
|--------|------:|--------|
| DAMIMAS | 854 | Kabupaten Tanah Laut, Kalimantan Selatan |
| LONSUM | 99 | Kabupaten Tanah Laut, Kalimantan Selatan |
| **Total** | **953** | Kabupaten Tanah Laut, Kalimantan Selatan, Indonesia |

---

## Maturity Classes

Bunches are classified into 4 ordinal stages from most mature (B1) to least mature (B4).

| Class | Label | Visual Appearance | Physical Position | Fraction |
|-------|-------|-------------------|-------------------|:--------:|
| **B1** | 0 | Red, large, round | Lowest on frond | 9.7% |
| **B2** | 1 | Black with red transition, large | Above B1 | 18.2% |
| **B3** | 2 | Full black, spiky, elongated | Above B2 | 51.6% |
| **B4** | 3 | Smallest, dark green-black | Highest (newest) | 20.5% |

**B2↔B3 ambiguity:** These two classes are visually ambiguous from any single camera
angle. After manual audit (audit-JSON-01), label noise was confirmed to be ≈ 0% —
the ambiguity is a genuine appearance similarity, not annotator error. This is the
primary bottleneck for all algorithms.

---

## Annotation Format

### YOLO TXT labels

Standard YOLO format, one file per image:
```
<class_id> <x_center> <y_center> <width> <height>
```
Class mapping: `0=B1, 1=B2, 2=B3, 3=B4`

### JSON ground truth

The authoritative ground truth is the JSON file per tree in `json/`. It contains
confirmed cross-view links (`_confirmedLinks`) that link the same physical bunch
across different camera sides. Ground truth unique counts are in `summary.by_class`.

```json
{
  "version": 3,
  "tree_id": "DAMIMAS_A21B_0001",
  "split": "train",
  "images": {
    "side_1": {
      "side_index": 0,
      "annotations": [
        {"class_name": "B3", "bbox_yolo": [0.512, 0.341, 0.082, 0.121], "box_index": 0}
      ]
    }
  },
  "_confirmedLinks": [
    {"linkId": "lnk-0", "sideA": 0, "bboxIdA": "b0", "sideB": 1, "bboxIdB": "b2"}
  ],
  "summary": {
    "total_unique_bunches": 8,
    "by_class": {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
  }
}
```

Ground truth unique counts are derived by running Union-Find connected components
over `_confirmedLinks`: boxes connected across sides belong to the same physical bunch.

---

## Splits

| Split | Trees | Images |
|-------|------:|-------:|
| Train | 763 | 3,164 |
| Val | 95 | 416 |
| Test | 95 | 412 |

Canonical tree-level split membership is in `split_manifest.csv`. The YOLO image
split is represented by the `images/train`, `images/val`, and `images/test`
folders (and matching `labels/` folders). E2E counters should use
`split_manifest.csv`, not the `split` value embedded in older JSON files.

Stratification was performed on unique bunch count distribution to ensure similar
class balance across splits.

---

## Ground Truth Quality

As of 2026-05-16, the ground truth is fully validated with zero known violations:

| Invariant | Status |
|-----------|--------|
| Same-side uniqueness (no bunch appears twice on same side) | ✅ 0 violations |
| Geometric adjacency (bunch only visible from adjacent sides) | ✅ 0 violations |
| Image–label–JSON integrity (all 3 files match) | ✅ 100% |
| Link-graph cycles | ✅ 0 |

**Fixes applied:**
- 8 wrap-around trees: over-linked bunches between side 0 and side 3 in 4-side trees
- 9 eight-side trees: geometric rule relaxed from max_dist=2 to max_dist=3 after
  visual validation by research assistants
- 31 four-side trees: auto-healed using the largest-bbox-as-home heuristic

**Net effect:** +62 unique bunches recovered from previously merged annotations.
All pre-fix versions are archived in the research repository.

---

## Folder Structure

```
SawitMVC-YOLO/
├── images/           3,992 JPEG images under train/val/test
├── labels/           3,992 YOLO TXT files under train/val/test
├── json/             953 JSON ground-truth files (one per tree)
├── data.yaml         YOLO data config (class names, split paths)
└── split_manifest.csv  Per-tree split assignment + stratification keys
```

Image naming convention: `{SITE}_{BLOCK}_{TREE_NUMBER}_{SIDE}.jpg`
Example: `DAMIMAS_A21B_0001_1.jpg` (site DAMIMAS, block A21B, tree 0001, side 1)

---

## Downloading the Dataset

The ground-truth annotations needed for every benchmark in this repository are
bundled at [`ground_truth/annotations/`](../ground_truth/annotations/) — no
external download is required for Tracks A, B, B', or C. The corresponding
3,992 JPEG images live on Hugging Face and are needed only when retraining a
detector. If the Hugging Face dataset is gated, log in or pass an approved
token.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ULM-DS-Lab/SawitMVC-YOLO",
    repo_type="dataset",
    local_dir="./SawitMVC-YOLO",
    token="your_hf_token",   # required for access
)
```

Or via CLI:
```bash
huggingface-cli download ULM-DS-Lab/SawitMVC-YOLO \
    --repo-type dataset \
    --local-dir ./SawitMVC-YOLO \
    --token your_hf_token
```
