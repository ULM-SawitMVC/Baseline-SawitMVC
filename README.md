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

### End-to-End Pipeline (detection → counting)

| Pipeline | Acc ±1 | Macro MAE | Notes |
|----------|:------:|:---------:|-------|
| `y26m` → SVM | **71.6%** | 1.118 | Best E2E combination |
| Heuristic M01 (GT input) | 87.62% | 0.375 | Upper bound on counting |

The ~16 pp gap between E2E and heuristic represents **detector error propagation** —
the counting algorithm is not the bottleneck. See [`docs/findings.md`](docs/findings.md).

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
│   ├── results/                 Pre-computed results (953 trees)
│   │   ├── accuracy_953.csv     Full 29-method ranking
│   │   ├── per_tree.csv         Per-tree predictions
│   │   ├── totals.csv           Aggregate counts
│   │   └── mean_per_tree.csv    Mean per-tree statistics
│   └── e2e/
│       └── y26m_svm_metrics.json  Best E2E: 71.6% Acc±1
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
