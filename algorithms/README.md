# Algorithms: Multi-View Deduplication

This folder contains the top-5 deterministic heuristic algorithms for deduplicating
oil palm bunch detections across multiple camera views.

**Constraint:** All algorithms are 100% algorithmic, no training, no embeddings,
no gradient computation. Every parameter is derived from first principles or dataset
medians from a held-out development set.

---

## Performance comparison (953 trees: SawitMVC-YOLO)

| Rank | File | Acc±1 | Macro MAE | Total MAE | Approach |
|:----:|------|------:|----------:|----------:|----------|
| 1 | [`M01_selector_b2b3.py`](M01_selector_b2b3.py) | **87.62%** | 0.375 | 1.331 | Selector plus B2 ↔ B3 correction |
| 2 | [`M02_selector_trifurc.py`](M02_selector_trifurc.py) | 87.62% | 0.376 | 1.331 | Trifurcation selector |
| 3 | [`M03_blend_geometric.py`](M03_blend_geometric.py) | 86.99% | 0.377 | 1.341 | Geometric-mean blend |
| 4 | [`M04_blend_floor_clamped.py`](M04_blend_floor_clamped.py) | 86.99% | 0.385 | 1.342 | Floor-clamped weighted blend |
| 5 | [`M05_blend_vis_divide.py`](M05_blend_vis_divide.py) | 86.99% | 0.388 | 1.346 | Simple weighted blend |
| - | Naive sum (reference) | 3.78% | 2.287 | 9.147 | No deduplication |

Full ranking of all 29 evaluated methods: [`results/heuristics_953/accuracy_full.csv`](../results/heuristics_953/accuracy_full.csv).

- **Acc ±1**: % of trees where every class prediction is within ±1 of ground truth (macro-averaged)
- **Macro MAE**: Unweighted mean of per-class absolute errors across B1/B2/B3/B4
- **Total MAE**: MAE of the sum B1+B2+B3+B4 per tree

---

## Input / Output Specification

### Input: `detections: list[dict]`

A flat list of all bounding-box detections across all camera sides for a single tree.

```python
detections = [
    {
        "class":      "B3",   # str, one of "B1", "B2", "B3", "B4"
        "x_norm":     0.512,  # float, normalized x-center of bbox (0.0–1.0)
        "y_norm":     0.341,  # float, normalized y-center of bbox (0.0–1.0)
        "side_index": 0,      # int  , camera side (0-based, wraps around)
    },
    # ... more detections
]
```

**Where to get `x_norm` and `side_index`:**
- From YOLO labels: column 1 = `x_norm`, column 2 = `y_norm`
- From JSON ground truth: `tree["images"]["side_X"]["side_index"]` and
  `annotation["bbox_yolo"][0]` / `[1]`

### Output: `dict[str, int]`

```python
{"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

Predicted unique bunch count per maturity class after deduplication.

---

## Algorithm Descriptions

### M01: `selector_b2b3` (Champion)

**Strategy:** Route each tree to the best estimator based on its detection profile,
then fix B2↔B3 ambiguity with a post-prediction reallocation.

**Stage 1, Trifurcation selector:**
- Dense B3-dominated tree (`b3_frac ≥ 0.60`, `n_total ≥ 25`) → `median3_floor`
- B1-rich, B3-moderate tree (`naive_B1 ≥ 3`, `b3_frac < 0.45`, `naive_B4 < 10`) → `adaptive_corrected`
- Default → `geometric_mean_blend`

**Stage 2, B2↔B3 split correction:**
Preserve the combined B2+B3 total from Stage 1 but redistribute using the naive B2:B3
ratio from raw detections. This corrects systematic class confusion without changing the
total count.

**Why it works:** B2 and B3 are visually ambiguous from any angle, so getting the split
right requires looking at the raw frequency signal. The total B2+B3 count is more stable
than the individual class predictions.

---

### M02: `selector_trifurc`

Same trifurcation logic as M01 (Stage 1 only). Without the B2↔B3 correction, Acc±1 is
identical but MAE is 0.001 higher. Useful when you need the selector behavior without
post-processing.

---

### M03: `blend_geometric`

**Strategy:** `sqrt(visibility × adaptive)` per class, floored at `max_per_side`.

- `visibility`: Gaussian weighting by horizontal position, detections near the center
  of a frame are more likely unique (weight ≈ 1), edge detections have lower weight.
- `adaptive`: Global divisor that decreases as total detection count increases (dense
  trees have a lower duplication rate per class).
- Geometric mean of the two is robust to outliers from either estimator.
- Floor at `max_per_side` ensures the count is never below the most-detected side's count.

---

### M04: `blend_floor_clamped`

**Strategy:** `round(0.6 × visibility + 0.4 × adaptive)`, floored at `max_per_side`.

Linear weighted blend with the same floor constraint as M03. Slightly simpler than the
geometric mean; useful when interpretability of weighting coefficients matters.

---

### M05: `blend_vis_divide`

**Strategy:** `round(0.6 × visibility + 0.4 × adaptive)`, no floor.

The simplest method in the top-5. No floor constraint, making it the most conservative
estimator. Recommended as a baseline when applying the algorithms to a new dataset or
camera configuration where `max_per_side` semantics may not hold.

---

## Usage Examples

### Run a single algorithm

```python
import json
from algorithms.M01_selector_b2b3 import predict

with open("ground_truth/annotations/DAMIMAS_A21B_0001.json") as f:
    tree = json.load(f)

detections = [
    {
        "class": ann["class_name"],
        "x_norm": ann["bbox_yolo"][0],
        "y_norm": ann["bbox_yolo"][1],
        "side_index": side["side_index"],
    }
    for side_key, side in tree["images"].items()
    for ann in side["annotations"]
]

result = predict(detections)
print(result)  # {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

### Compare all top-5 algorithms

```python
from algorithms import RANKING

results = {}
for name, meta in RANKING.items():
    results[name] = meta["predict"](detections)
    print(f"{name}: {results[name]}")
```

### Use the get_algorithm helper

```python
from algorithms import get_algorithm

predict = get_algorithm("M03_blend_geometric")
result = predict(detections)
```

---

## Adding a New Algorithm

1. Create `algorithms/M{NN}_{family}_{descriptor}.py`
2. Implement `predict(detections: list) -> dict` with signature matching the spec above
3. Add a docstring with benchmark results on 953 trees
4. Run `python benchmarks/run_benchmark.py` to get official numbers
5. Add an entry to `algorithms/__init__.py` RANKING dict
6. Submit a PR with the `.md` checklist filled in

**Naming convention:** `M{NN}` where NN continues from the last registered algorithm.
Families: `selector`, `blend`, `weight`, `divide`, `median`, `entropy`, `stack`.

**Minimum bar:** Acc±1 ≥ 85.00% on the full 953-tree dataset to be considered for
inclusion in the top-tier list.
