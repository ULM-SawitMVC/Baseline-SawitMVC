# SawitMVC Baseline — Multi-View Oil Palm Bunch Counting

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

**SawitMVC Baseline** is the official research baseline for multi-view oil palm fruit bunch
counting and maturity classification. Given 4–8 photos of the same oil palm tree taken from
different angles, the task is to count the **unique number of bunches per maturity class**
(B1 → B4) without double-counting bunches visible from multiple sides.

> **Core problem:** Naive summation overcounts by ~83.4% because the same bunch is
> detected across multiple camera angles. This repository provides 5 deterministic
> heuristic algorithms that reduce the error to ≤13% without any training.

---

## Results at a Glance

### Heuristic Deduplication (953 trees, no training required)

| Rank | Algorithm | Acc ±1 | Macro MAE | Approach |
|:----:|-----------|:------:|:---------:|----------|
| 🥇 1 | `M01_selector_b2b3` | **87.62%** | 0.375 | Trifurcation selector + B2↔B3 correction |
| 🥈 2 | `M02_selector_trifurc` | 87.62% | 0.376 | Trifurcation selector (base) |
| 🥉 3 | `M03_blend_geometric` | 86.99% | 0.377 | Geometric mean blend |
| 4 | `M04_blend_floor_clamped` | 86.99% | 0.385 | Floor-clamped weighted blend |
| 5 | `M05_blend_vis_divide` | 86.99% | 0.388 | Visibility + adaptive divide |
| — | Naive sum (baseline) | 3.78% | 2.287 | No deduplication |

**Acc ±1** = percentage of trees where predicted count per class is within ±1 of ground truth (macro-averaged across 4 classes). Evaluated on 953 trees from `Brand-New-Dataset-YOLO`.

### YOLO26 Detection (mAP50, test split)

| Model | mAP50 | Speed | Size | Notes |
|-------|:-----:|:-----:|:----:|-------|
| `y26n_vanilla_local` | **0.521** | **0.2 ms** | 5.2 MB | **Recommended for production** |
| `y26s_nopretrained` | 0.511 | 0.5 ms | 20 MB | Scratch training matches pretrained |
| `y26m_vanilla_local` | 0.509 | 0.8 ms | 42 MB | Largest; best E2E pipeline |
| `y26s_vanilla_local` | 0.506 | 0.5 ms | 20 MB | Standard small |
| `y26s_noaug` | 0.465 | 0.5 ms | 20 MB | Ablation: no augmentation |

### E2E — Per-Image Approach (simple, no tree grouping)

Each image is processed independently. Detections per class are aggregated across images
of the same tree using `max` (default), `mean`, or `sum`.

> Run: `python pipeline/run_e2e_per_image.py --name y26n_vanilla_local --weights models/y26n_vanilla_local.pt --agg max`

| Aggregation | Acc ±1 | MAE | Description |
|-------------|:------:|:---:|-------------|
| **max** | TBD | TBD | `count[c] = max detections in any single image` |
| **mean** | TBD | TBD | `count[c] = round(mean detections per image)` |
| **sum** | ~3.8% | ~2.3 | Naive sum (no dedup) — known baseline from heuristic bench |

Generate results: `python pipeline/run_e2e_per_image.py --name {model} --weights models/{model}.pt`

---

### E2E — Per-Tree Approach (full pipeline, pre-computed)

All images of a tree are grouped, detections aggregated, then a ML counter is applied.
Results on test split (95 trees). Sorted by Acc±1.

| Rank | Detector | Counter | Acc ±1 ↑ | MAE ↓ | B1 | B2 | B3 | B4 |
|:----:|----------|---------|:--------:|:-----:|:--:|:--:|:--:|:--:|
| 🥇 1 | y26m_vanilla | SVM | **71.6%** | 1.118 | 92.6% | 63.2% | 60.0% | 70.5% |
| 2 | y26s_noaug | SVM | 70.5% | 1.126 | 91.6% | 69.5% | 56.8% | 64.2% |
| 3 | y26n_vanilla | SVM | 70.0% | 1.145 | 90.5% | 68.4% | 56.8% | 64.2% |
| 4 | y26s_nopretrained | M01 | 69.2% | 1.266 | 91.6% | 63.2% | 52.6% | 69.5% |
| 5 | y26s_nopretrained | SVM | 69.0% | 1.145 | 90.5% | 68.4% | 51.6% | 65.3% |
| 6 | y26s_vanilla | SVM | 69.0% | 1.163 | 93.7% | 68.4% | 48.4% | 65.3% |
| 7 | y26s_noaug | RF | 68.4% | 1.184 | 92.6% | 66.3% | 55.8% | 58.9% |
| 8 | y26n_vanilla | RF | 68.2% | 1.218 | 90.5% | 68.4% | 54.7% | 58.9% |
| 9 | y26m_vanilla | RF | 67.9% | 1.211 | 95.8% | 68.4% | 49.5% | 57.9% |
| 10 | y26s_nopretrained | RF | 67.9% | 1.229 | 93.7% | 65.3% | 55.8% | 56.8% |
| 11 | y26n_vanilla | M01 | 67.1% | 1.337 | 87.4% | 65.3% | 51.6% | 64.2% |
| 12 | y26s_noaug | M01 | 66.6% | 1.384 | 90.5% | 68.4% | 43.2% | 64.2% |
| 13 | y26s_vanilla | RF | 66.6% | 1.216 | 96.8% | 68.4% | 48.4% | 52.6% |
| 14 | y26s_vanilla | M01 | 65.5% | 1.403 | 89.5% | 66.3% | 38.9% | 67.4% |
| 15 | y26m_vanilla | M01 | 64.5% | 1.400 | 90.5% | 56.8% | 40.0% | 70.5% |
| — | **LR counter** | all models | TBD | TBD | Run `run_e2e_pipeline.py` | | | |
| — | **M01 on GT** (upper bound) | — | **87.6%** | **0.375** | — | — | — | — |

> Full analysis in [`docs/e2e_pipeline.md`](docs/e2e_pipeline.md).
> All `metrics.json` files in [`benchmarks/e2e/`](benchmarks/e2e/).

The ~16 pp gap between best E2E and heuristic-on-GT represents **detector error
propagation** — improving the detector is the highest-leverage path. See [`docs/findings.md`](docs/findings.md).

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ULM-DS-Lab/SawitMVC-YOLO",
    repo_type="dataset",
    local_dir="./SawitMVC-YOLO",
)
```

### 3. Run the benchmark

```bash
python benchmarks/run_benchmark.py --data ./SawitMVC-YOLO/json/
```

Expected output:

```
SawitMVC Baseline — Benchmark Results (953 trees)
==================================================
Rank  Algorithm               Acc±1    MAE
  1   M01_selector_b2b3      87.62%  0.3746
  2   M02_selector_trifurc   87.62%  0.3757
  3   M03_blend_geometric    86.99%  0.3767
  4   M04_blend_floor_clamped 86.99%  0.3848
  5   M05_blend_vis_divide   86.99%  0.3875
```

### 4. Run inference on a single tree

```python
import json
from algorithms.M01_selector_b2b3 import predict

# Load detections from a JSON ground-truth file (or your own detector output)
with open("SawitMVC-YOLO/json/DAMIMAS_A21B_0001.json") as f:
    tree = json.load(f)

# Build detection list from JSON
detections = []
for side_key, side in tree["images"].items():
    for ann in side["annotations"]:
        x_center = ann["bbox_yolo"][0]
        detections.append({
            "class": ann["class_name"],
            "x_norm": x_center,
            "y_norm": ann["bbox_yolo"][1],
            "side_index": side["side_index"],
        })

result = predict(detections)
print(result)  # {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

---

## Repository Structure

```
Baseline-SawitMVC/
├── README.md                    You are here
├── CONTRIBUTING.md              How to contribute new algorithms or fixes
├── CHANGELOG.md                 Version history
├── LICENSE                      CC BY-NC 4.0
├── requirements.txt             Python dependencies
│
├── algorithms/                  Top-5 heuristic deduplication algorithms
│   ├── README.md                Algorithm guide, comparison table, input/output spec
│   ├── __init__.py              Unified imports + ranking metadata
│   ├── M01_selector_b2b3.py    🥇 Champion: 87.62% Acc±1
│   ├── M02_selector_trifurc.py 🥈 Runner-up: 87.62% Acc±1
│   ├── M03_blend_geometric.py  🥉 86.99% Acc±1
│   ├── M04_blend_floor_clamped.py  86.99% Acc±1
│   └── M05_blend_vis_divide.py     86.99% Acc±1
│
├── models/                      Trained YOLO26 weights (all < 50 MB)
│   ├── README.md                Model comparison, inference guide
│   ├── y26n_vanilla_local.pt   🏆 Best: mAP50=0.521, 5.2 MB
│   ├── y26s_vanilla_local.pt   Standard small, 20 MB
│   ├── y26m_vanilla_local.pt   Medium, 42 MB (best E2E)
│   ├── y26s_nopretrained.pt    Ablation: scratch training
│   └── y26s_noaug.pt           Ablation: no augmentation
│
├── benchmarks/                  Reproducible benchmark suite
│   ├── README.md                How to run, metric definitions
│   ├── run_benchmark.py         Entry point: load data → run → print table
│   ├── results/                 Pre-computed results (953 trees, 29 methods)
│   │   ├── accuracy_953.csv     Full 29-method ranking
│   │   ├── per_tree.csv         Per-tree predictions
│   │   ├── totals.csv           Aggregate counts
│   │   └── mean_per_tree.csv    Mean per-tree statistics
│   └── e2e/                     All 15 E2E results (5 detectors × 3 counters)
│       ├── e2e_y26m_vanilla_local_svm/   metrics.json + predictions.csv
│       ├── e2e_y26n_vanilla_local_svm/
│       └── … (15 folders total)
│
├── pipeline/                    E2E pipeline scripts
│   ├── README.md                Step-by-step replication + ablation guide
│   ├── run_e2e_pipeline.py      Unified harness (inference → features → eval)
│   ├── run_e2e_inference.py     YOLO inference → JSON per tree
│   ├── build_counting_features.py  Extract 13-dim features
│   ├── run_counting_svm.py      SVM counter (train + evaluate)
│   └── run_counting_rf.py       RF counter (train + evaluate)
│
├── predictions/                 Pre-computed YOLO inference (~28 MB)
│   ├── y26n_vanilla_local_inference/  953 JSON files per detector
│   ├── y26s_vanilla_local_inference/
│   ├── y26m_vanilla_local_inference/
│   ├── y26s_nopretrained_inference/
│   └── y26s_noaug_inference/
│
├── figures/                     All visualizations
│   ├── README.md                Figure index and descriptions
│   ├── eda/                     23 EDA plots (dataset characteristics)
│   └── training/                YOLO training artifacts (5 models × 7 plots)
│
└── docs/                        In-depth documentation
    ├── dataset.md               Dataset structure, classes, GT quality
    ├── algorithms.md            How algorithms work, design rationale
    ├── training.md              Reproduce YOLO training experiments
    ├── evaluation.md            Metric definitions, evaluation protocol
    ├── e2e_pipeline.md          Full E2E guide + 15-combo results + ablation recipes
    └── findings.md              Key research findings and future work
```

---

## Dataset

**SawitMVC-YOLO** — 953 oil palm trees from DAMIMAS (854) and LONSUM (99) plantations
in Kalimantan, Indonesia.

| Statistic | Value |
|-----------|-------|
| Trees | 953 |
| Images | 3,992 JPEG (960×1280) |
| Unique bunches | 9,823 |
| Raw detections | 18,544 |
| Deduplication ratio | 0.53 (naive overcounts 1.83×) |
| 4-side trees | 908 (95.3%) |
| 8-side trees | 45 (4.7%) |
| Ground truth format | JSON with confirmed cross-side links |

**Maturity classes (ordinal, most → least mature):**

| Class | Visual | Physical Position |
|-------|--------|-------------------|
| **B1** | Red, large, round | Lowest on tree |
| **B2** | Black with red transition | Above B1 |
| **B3** | Full black, spiky, elongated | Above B2 |
| **B4** | Smallest, dark green-black | Highest (newest) |

Download: [HuggingFace — ULM-DS-Lab/SawitMVC-YOLO](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

---

## How to Contribute

We welcome contributions of:

- **New deduplication algorithms** — must be deterministic, no training, better Acc±1 than M05
- **Bug reports** — incorrect benchmark results, broken scripts
- **Documentation improvements**

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines.

Quick summary:
1. Fork this repository
2. Create a branch: `git checkout -b algo/your-algorithm-name`
3. Add your algorithm as `algorithms/M{NN}_family_descriptor.py`
4. Run `python benchmarks/run_benchmark.py` and include results in your PR
5. Open a pull request using the provided template

---

## Citation

If you use this dataset or baseline in your research, please cite:

```bibtex
@dataset{sawitmvc2026,
  title   = {SawitMVC-YOLO: Multi-View Oil Palm Bunch Counting Dataset},
  author  = {ULM-DS-Lab},
  year    = {2026},
  url     = {https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO},
  license = {CC BY-NC 4.0}
}
```

---

## License

This project is licensed under [CC BY-NC 4.0](LICENSE).
You may use, share, and adapt this work for **non-commercial purposes** with attribution.
